/**
 * k6 Load Test — LLM Serving Endpoint
 *
 * Usage:
 *   Recommended: run via `sml loadtest ...`, which submits this script as a
 *   cluster k6 job and passes RUN_CONFIG_JSON to it.
 *
 *   This script is strict: RUN_CONFIG_JSON must contain one of:
 *   - custom
 *   - realistic
 *   - scenario_definition
 *   If none are present, initialization fails.
 *
 * Config source:
 *   RUN_CONFIG_JSON     - JSON payload from Python launcher (primary source)
 *   PROMPTS_FILE        - JSON prompt corpus path mounted in the k6 container
 *
 * Optional env-only overrides (useful for direct k6 runs):
 *   STREAM_RATIO        - fraction of requests that use streaming (default 0.7)
 *   SYSTEM_PROMPT       - prepend a fixed system prompt to every chat request, enabling
 *                         vLLM prefix caching (APC) to share KV blocks across requests.
 *                         Set to "default" to use the built-in ~200-token prompt, or pass
 *                         your own string.  Empty string (default) disables it.
 *                         NOTE: requires vllm serve --enable-prefix-caching
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Counter, Rate, Gauge } from "k6/metrics";
import { SharedArray } from "k6/data";

// ─── Config ───────────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_REQUEST_TIMEOUT = "120s";
const DEFAULT_STREAM_RATIO = 0.7;
const DEFAULT_THINK_TIME = 2;
const DEFAULT_MAX_VUS = 10;
const DEFAULT_REALISTIC_USERS = 20;
const DEFAULT_RAMP_DOWN = "30s";
const DEFAULT_DURATION = "5m";
const DEFAULT_REALISTIC_DURATION = "15m";
const ESTIMATED_CHARS_PER_TOKEN = 4;
const MS_PER_SECOND = 1000;
const LATENCY_LABEL_PATTERN = /^e2e_latency_ms\{label:(.+)\}$/;
const STATUS_200_CHECK = "status 200";
const SCENARIO_CONSTANT_VUS = "constant-vus";
const SCENARIO_RAMPING_VUS = "ramping-vus";
const LABEL_TAG = "label";
const STAT_AVG = "avg";
const STAT_MED = "med";
const STAT_P95 = "p(95)";
const STAT_P99 = "p(99)";

function parseRunConfig() {
  if (!__ENV.RUN_CONFIG_JSON) {
    throw new Error(
      "Missing RUN_CONFIG_JSON. This script requires launcher-provided config.",
    );
  }
  try {
    return JSON.parse(__ENV.RUN_CONFIG_JSON);
  } catch (error) {
    throw new Error(`Invalid RUN_CONFIG_JSON: ${error?.message ?? error}`);
  }
}

function parseNumber(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parsePositiveInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function randomUnit() {
  return Math.random(); // NOSONAR: load-test sampling and jitter are not security-sensitive.
}

const RUN_CFG = parseRunConfig();
const CFG_PROMPT_LABELS =
  Array.isArray(RUN_CFG.prompt_labels) && RUN_CFG.prompt_labels.length > 0
    ? RUN_CFG.prompt_labels
    : null;

const BASE_URL = RUN_CFG.server_url ?? DEFAULT_BASE_URL;
const STREAM_RATIO = parseNumber(__ENV.STREAM_RATIO, DEFAULT_STREAM_RATIO);
const API_KEY = RUN_CFG.api_key ?? "";
const CHAT_MODE = (RUN_CFG.chat_mode ?? false) === true;
const MODEL = RUN_CFG.model ?? "";
const REQUEST_TIMEOUT =
  RUN_CFG.request_timeout ??
  RUN_CFG.scenario_definition?.request_timeout ??
  DEFAULT_REQUEST_TIMEOUT;
const STREAM_RECORD_USAGE =
  (RUN_CFG.stream_record_usage ??
    RUN_CFG.scenario_definition?.stream_record_usage ??
    false) === true;
const INITIAL_STAGGER =
  (RUN_CFG.initial_stagger ??
    RUN_CFG.scenario_definition?.initial_stagger ??
    false) === true;
// THINK_TIME: max seconds of sleep between requests per VU (uniform [0, THINK_TIME]).
// Lower values → more in-flight requests → higher KV cache fill. 0 = no sleep.
const THINK_TIME = parseNumber(RUN_CFG.think_time, DEFAULT_THINK_TIME);
// MAX_TOKENS: when set, overrides the per-prompt max_tokens.
// KV cache fill is driven by the decode phase — longer outputs hold KV blocks longer.
// Use 1024–4096 with kv_stress to keep requests alive and fill the cache.
const MAX_TOKENS = RUN_CFG.max_tokens
  ? Number.parseInt(RUN_CFG.max_tokens, 10)
  : null;

// ─── Shared system prompt (prefix caching) ────────────────────────────────────
// A fixed prefix shared by all requests lets vLLM's automatic prefix caching
// (APC) keep those KV blocks warm across the entire run, boosting cache fill %.
// Use SYSTEM_PROMPT=default to activate the built-in prompt, or pass your own.

const DEFAULT_SYSTEM_PROMPT =
  "You are a knowledgeable and helpful AI assistant. " +
  "You answer questions clearly and concisely, explain technical concepts at an " +
  "appropriate depth for the user, and always reason step-by-step before giving " +
  "a final answer. When writing code, prefer readability and correctness over " +
  "brevity. If you are unsure about something, say so rather than guessing. " +
  "You follow instructions carefully and stay on topic. Your responses should be " +
  "well-structured: use bullet points or numbered lists when enumerating items, " +
  "and use code blocks for all code snippets regardless of language.";

function resolveSystemPrompt(raw) {
  if (raw === "default") {
    return DEFAULT_SYSTEM_PROMPT;
  }
  return raw ?? "";
}

const SYSTEM_PROMPT = resolveSystemPrompt(__ENV.SYSTEM_PROMPT);

if (SYSTEM_PROMPT) {
  console.log(
    `[init] System prompt enabled (${Math.round(SYSTEM_PROMPT.length / ESTIMATED_CHARS_PER_TOKEN)} tokens est.) — ` +
      "prefix caching will share KV blocks across requests",
  );
}

// ─── Prompts corpus ───────────────────────────────────────────────────────────
// SharedArray stores the prompts once in Go memory, shared across all VUs.
// Without this, each VU gets its own JS heap copy — costly at high concurrency.
//
// PROMPT_LABELS (optional) — comma-separated list of labels to keep, e.g.
//   --env PROMPT_LABELS=short,medium
// Filtering happens here at init time so every VU shares the same subset.
// Omit to use all labels (the weighted mix defined in LABEL_WEIGHTS below).

const PROMPTS = new SharedArray("prompts", function loadPrompts() {
  if (!__ENV.PROMPTS_FILE) {
    throw new Error(
      "Missing PROMPTS_FILE. Provide a cluster-visible prompt corpus path.",
    );
  }
  const all = JSON.parse(open(__ENV.PROMPTS_FILE));
  const filter = CFG_PROMPT_LABELS ? new Set(CFG_PROMPT_LABELS) : null;
  return filter ? all.filter((p) => filter.has(p.label)) : all;
});
if (PROMPTS.length === 0) {
  throw new Error(
    "No prompts available after filtering. Check prompt_labels against the configured prompt corpus labels.",
  );
}
const conversationPromptCount = PROMPTS.filter((p) => p.messages).length;
console.log(
  `[init] Loaded ${PROMPTS.length} prompts` +
    (CFG_PROMPT_LABELS
      ? ` (filtered to: ${CFG_PROMPT_LABELS.join(",")})`
      : "") +
    ` — ${conversationPromptCount} multi-turn, ${PROMPTS.length - conversationPromptCount} single-turn`,
);

// ─── Metrics ──────────────────────────────────────────────────────────────────

const e2eLatency = new Trend("e2e_latency_ms", true);
const tokensPerSecond = new Trend("tokens_per_second");
const promptTokens = new Trend("prompt_tokens");
const completionTokens = new Trend("completion_tokens");
const requestErrors = new Counter("request_errors");
const successRate = new Rate("success_rate");
const activeRequests = new Gauge("active_requests");
const malformedStreamEvents = new Counter("malformed_stream_events");

function parseJsonOrNull(raw, context) {
  try {
    return JSON.parse(raw);
  } catch (error) {
    console.warn(`Could not parse ${context}: ${error?.message ?? error}`);
    return null;
  }
}

// ─── Scenario loading from JSON ───────────────────────────────────────────────

function scenarioToK6(scenario) {
  // Map scenario JSON to k6 scenario config
  if (!scenario) return null;
  // Support both ramping-vus and constant-vus
  if (scenario.executor === SCENARIO_RAMPING_VUS) {
    return {
      executor: SCENARIO_RAMPING_VUS,
      startVUs: scenario.startVUs ?? 0,
      stages: scenario.stages,
      gracefulRampDown: scenario.gracefulRampDown ?? DEFAULT_RAMP_DOWN,
    };
  }
  return {
    executor: SCENARIO_CONSTANT_VUS,
    vus: scenario.vus ?? DEFAULT_MAX_VUS,
    duration: scenario.duration ?? DEFAULT_DURATION,
  };
}

const scenarioName = RUN_CFG.scenario ?? "configured";

function parseCustomStages(raw) {
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error("custom.stages must be a non-empty JSON array");
  }

  for (const stage of parsed) {
    const okDuration =
      typeof stage?.duration === "string" && stage.duration.length > 0;
    const okTarget = Number.isFinite(stage?.target);
    if (!okDuration || !okTarget) {
      throw new Error(
        "each custom stage must include { duration: string, target: number }",
      );
    }
  }

  return parsed;
}

function buildCustomScenario(custom) {
  if (!custom || !custom.executor) return null;
  if (custom.executor === SCENARIO_RAMPING_VUS) {
    const defaultStages = [
      { duration: "2m", target: 10 },
      { duration: DEFAULT_DURATION, target: 10 },
      { duration: DEFAULT_RAMP_DOWN, target: 0 },
    ];
    const stages = custom.stages
      ? parseCustomStages(custom.stages)
      : defaultStages;
    return {
      executor: SCENARIO_RAMPING_VUS,
      startVUs: 0,
      stages,
      gracefulRampDown: custom.ramp_down ?? DEFAULT_RAMP_DOWN,
    };
  }
  return {
    executor: SCENARIO_CONSTANT_VUS,
    vus: parsePositiveInteger(custom.vus, DEFAULT_MAX_VUS),
    duration: custom.duration ?? DEFAULT_DURATION,
  };
}

function buildRealisticScenario(realistic) {
  if (!realistic) return null;
  return {
    executor: SCENARIO_CONSTANT_VUS,
    vus: parsePositiveInteger(realistic.users, DEFAULT_REALISTIC_USERS),
    duration: realistic.duration ?? DEFAULT_REALISTIC_DURATION,
  };
}

const customScenario = RUN_CFG.custom
  ? buildCustomScenario(RUN_CFG.custom)
  : null;
const realisticScenario = RUN_CFG.realistic
  ? buildRealisticScenario(RUN_CFG.realistic)
  : null;
const definedScenario = scenarioToK6(RUN_CFG.scenario_definition);
const scenarioCandidates = [
  customScenario,
  realisticScenario,
  definedScenario,
].filter(Boolean);

if (scenarioCandidates.length === 0) {
  throw new Error(
    "No scenario found in RUN_CONFIG_JSON. Expected one of: custom, realistic, scenario_definition",
  );
}

if (scenarioCandidates.length > 1) {
  throw new Error(
    "Ambiguous RUN_CONFIG_JSON: provide only one of custom, realistic, scenario_definition",
  );
}

const selectedScenario = scenarioCandidates[0];

// ─── k6 options ───────────────────────────────────────────────────────────────

export const options = {
  scenarios: {
    load: selectedScenario,
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],
    success_rate: ["rate>0.95"],
    e2e_latency_ms: ["p(95)<30000"],
  },
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

// ─── Request helpers ──────────────────────────────────────────────────────────

const HEADERS = {
  "Content-Type": "application/json",
  ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
};

const ENDPOINT = CHAT_MODE ? "/v1/chat/completions" : "/v1/completions";

// Build the messages array for a prompt, handling both single-turn and multi-turn formats.
// Prepends the system prompt (if configured) so all requests share a common prefix,
// enabling vLLM APC to reuse the corresponding KV blocks.
function buildMessages(prompt) {
  const body = prompt.messages ?? [{ role: "user", content: prompt.content }];
  if (!SYSTEM_PROMPT) return body;
  return [{ role: "system", content: SYSTEM_PROMPT }, ...body];
}

// Flatten a messages array to a plain text prompt for the completions endpoint.
// Ends with "\nAssistant:" so the model knows to continue as the assistant.
function flattenMessages(messages) {
  const parts = messages.map((m) =>
    m.role === "user" ? `Human: ${m.content}` : `Assistant: ${m.content}`,
  );
  return parts.join("\n") + "\nAssistant:";
}

function payload(prompt, stream) {
  const maxTokens = MAX_TOKENS ?? prompt.max_tokens;
  if (CHAT_MODE) {
    const body = {
      ...(MODEL ? { model: MODEL } : {}),
      messages: buildMessages(prompt),
      max_tokens: maxTokens,
      temperature: 1.0,
      stream,
    };
    if (stream) {
      body.stream_options = { include_usage: true };
    }
    return JSON.stringify(body);
  }
  const base = prompt.messages
    ? flattenMessages(prompt.messages)
    : prompt.content;
  const text = SYSTEM_PROMPT ? `${SYSTEM_PROMPT}\n\n${base}` : base;
  return JSON.stringify({
    ...(MODEL ? { model: MODEL } : {}),
    prompt: text,
    max_tokens: maxTokens,
    temperature: 1.0,
    stream,
  });
}

function extractText(choice) {
  if (CHAT_MODE) {
    return choice?.delta?.content ?? choice?.message?.content ?? "";
  }
  return choice?.text ?? "";
}

function recordUsage(usage, elapsed, label) {
  if (!usage) return;
  const prompt = Number(usage.prompt_tokens);
  const completion = Number(usage.completion_tokens);
  if (Number.isFinite(prompt) && prompt >= 0) {
    promptTokens.add(prompt, { label });
  }
  if (Number.isFinite(completion) && completion >= 0) {
    completionTokens.add(completion, { label });
    if (elapsed > 0) {
      tokensPerSecond.add((completion / elapsed) * MS_PER_SECOND, { label });
    }
  }
}

function parseStreamingUsage(bodyText) {
  if (typeof bodyText !== "string" || bodyText.length === 0) return null;
  let usage = null;
  let completionText = "";

  for (const rawLine of bodyText.split("\n")) {
    if (!rawLine.startsWith("data:")) continue;
    const data = rawLine.slice(5).trim();
    if (!data || data === "[DONE]") continue;

    try {
      const evt = JSON.parse(data);
      if (evt?.usage) usage = evt.usage;
      completionText += extractText(evt?.choices?.[0]);
    } catch {
      malformedStreamEvents.add(1);
    }
  }

  if (usage) return usage;
  if (completionText.length > 0) {
    return {
      prompt_tokens: null,
      completion_tokens: Math.max(
        1,
        Math.round(completionText.length / ESTIMATED_CHARS_PER_TOKEN),
      ),
    };
  }
  return null;
}

// ─── Non-streaming ────────────────────────────────────────────────────────────

function runNonStreaming(prompt) {
  activeRequests.add(1);
  const start = Date.now();

  const res = http.post(`${BASE_URL}${ENDPOINT}`, payload(prompt, false), {
    headers: HEADERS,
    timeout: REQUEST_TIMEOUT,
    tags: { [LABEL_TAG]: prompt.label, stream: "false" },
  });

  activeRequests.add(-1);
  const elapsed = Date.now() - start;
  e2eLatency.add(elapsed, { label: prompt.label });

  const body = parseJsonOrNull(res.body, "non-streaming response body");

  const ok = check(res, {
    [STATUS_200_CHECK]: (r) => r.status === 200,
    "has choices": () => body?.choices?.length > 0,
  });

  successRate.add(ok);

  if (!ok) {
    requestErrors.add(1, { status: res.status, [LABEL_TAG]: prompt.label });
    console.warn(
      `[non-stream] ${prompt.label} failed: ${res.status} ${res.body?.slice(0, 200)}`,
    );
    return;
  }

  recordUsage(body?.usage, elapsed, prompt.label);
}

// ─── Streaming ────────────────────────────────────────────────────────────────

function runStreaming(prompt) {
  activeRequests.add(1);
  const start = Date.now();

  const streamReqOpts = {
    headers: HEADERS,
    timeout: REQUEST_TIMEOUT,
    tags: { [LABEL_TAG]: prompt.label, stream: "true" },
  };
  if (!STREAM_RECORD_USAGE) {
    // Avoid buffering large SSE bodies unless usage accounting is enabled.
    streamReqOpts.responseType = "none";
  }
  const res = http.post(
    `${BASE_URL}${ENDPOINT}`,
    payload(prompt, true),
    streamReqOpts,
  );

  activeRequests.add(-1);
  const elapsed = Date.now() - start;
  e2eLatency.add(elapsed, { label: prompt.label });

  const ok = check(res, { [STATUS_200_CHECK]: (r) => r.status === 200 });

  successRate.add(ok);

  if (!ok) {
    requestErrors.add(1, { status: res.status, [LABEL_TAG]: prompt.label });
    console.warn(`[stream] ${prompt.label} failed: ${res.status}`);
    return;
  }

  if (STREAM_RECORD_USAGE) {
    const usage = parseStreamingUsage(res.body);
    recordUsage(usage, elapsed, prompt.label);
  }
}

// ─── Weighted prompt selection ────────────────────────────────────────────────
// Weights by label mirror a realistic serving workload:
// more medium/long tasks, fewer trivial and very long inputs.
// Unknown labels (e.g. from a custom dataset) get weight 10.
//
// You can override defaults with RUN_CONFIG_JSON.label_weights or
// RUN_CONFIG_JSON.scenario_definition.label_weights.
const DEFAULT_LABEL_WEIGHTS = {
  medium: 20,
  long_output: 30,
  long_input: 20,
  xl_input: 10,
  conv_medium: 10,
  conv_long: 10,
};

function resolveLabelWeights(raw) {
  if (!raw || typeof raw !== "object") return DEFAULT_LABEL_WEIGHTS;
  const merged = { ...DEFAULT_LABEL_WEIGHTS };
  for (const [label, weight] of Object.entries(raw)) {
    if (Number.isFinite(weight) && weight >= 0) {
      merged[label] = weight;
    }
  }
  return merged;
}

const LABEL_WEIGHTS = resolveLabelWeights(
  RUN_CFG.label_weights ?? RUN_CFG.scenario_definition?.label_weights,
);

// Cumulative weight array — avoids duplicating prompt objects into a flat pool.
const CUMULATIVE_WEIGHTS = PROMPTS.reduce((acc, p, i) => {
  const w = LABEL_WEIGHTS[p.label] ?? 10;
  acc.push((acc[i - 1] ?? 0) + w);
  return acc;
}, []);
const TOTAL_WEIGHT = CUMULATIVE_WEIGHTS[CUMULATIVE_WEIGHTS.length - 1];
if (!Number.isFinite(TOTAL_WEIGHT) || TOTAL_WEIGHT <= 0) {
  throw new Error("Invalid label weights: total prompt weight must be > 0");
}

function pickPrompt() {
  const r = randomUnit() * TOTAL_WEIGHT;
  let lo = 0;
  let hi = CUMULATIVE_WEIGHTS.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (CUMULATIVE_WEIGHTS[mid] < r) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return PROMPTS[lo];
}

// ─── VU entrypoint ────────────────────────────────────────────────────────────

// Track whether this VU has run its first iteration yet.
// Used when initial_stagger=true in RUN_CONFIG_JSON.
const vuStarted = { done: false };

export default function runIteration() {
  // Optional initial staggering: offset each VU's first request by a random
  // fraction of THINK_TIME so all VUs don't pile up at t=0.
  if (INITIAL_STAGGER && !vuStarted.done) {
    vuStarted.done = true;
    if (THINK_TIME > 0) sleep(randomUnit() * THINK_TIME);
  }

  const prompt = pickPrompt();

  if (randomUnit() < STREAM_RATIO) {
    runStreaming(prompt);
  } else {
    runNonStreaming(prompt);
  }

  if (THINK_TIME > 0) sleep(randomUnit() * THINK_TIME);
}

// ─── Summary ──────────────────────────────────────────────────────────────────

function formatMilliseconds(metric, stat) {
  return metric ? `${(metric.values[stat] ?? 0).toFixed(0)}ms` : "n/a";
}

function metricValue(metric, stat) {
  return metric?.values[stat] ?? 0;
}

function buildLabelBreakdown(metrics) {
  const labelBreakdown = {};
  for (const [key, metric] of Object.entries(metrics)) {
    const match = key.match(LATENCY_LABEL_PATTERN);
    if (match) {
      labelBreakdown[match[1]] = {
        p50: formatMilliseconds(metric, STAT_MED),
        p95: formatMilliseconds(metric, STAT_P95),
        avg: formatMilliseconds(metric, STAT_AVG),
      };
    }
  }
  return labelBreakdown;
}

function buildThresholdSummary(metrics) {
  return Object.entries(metrics)
    .filter(([, metric]) => metric.thresholds)
    .map(([name, metric]) => ({
      metric: name,
      ok: Object.values(metric.thresholds).every((threshold) => threshold.ok),
    }));
}

export function handleSummary(data) {
  const metrics = data.metrics;

  const summary = {
    scenario: scenarioName,
    thresholds: buildThresholdSummary(metrics),
    latency: {
      e2e_p50: formatMilliseconds(metrics.e2e_latency_ms, STAT_MED),
      e2e_p95: formatMilliseconds(metrics.e2e_latency_ms, STAT_P95),
      e2e_p99: formatMilliseconds(metrics.e2e_latency_ms, STAT_P99),
    },
    latency_by_label: buildLabelBreakdown(metrics),
    throughput: {
      total_requests: metricValue(metrics.http_reqs, "count"),
      req_per_sec: metricValue(metrics.http_reqs, "rate").toFixed(2),
      success_rate:
        (metricValue(metrics.success_rate, "rate") * 100).toFixed(1) + "%",
      error_count: metricValue(metrics.request_errors, "count"),
    },
    tokens: {
      avg_completion_tokens: metricValue(
        metrics.completion_tokens,
        STAT_AVG,
      ).toFixed(1),
      avg_tokens_per_second: metricValue(
        metrics.tokens_per_second,
        STAT_AVG,
      ).toFixed(1),
    },
  };

  return {
    stdout: `\n=== LLM Load Test Summary ===\n${JSON.stringify(summary, null, 2)}\n`,
    "summary.json": JSON.stringify(summary, null, 2),
  };
}
