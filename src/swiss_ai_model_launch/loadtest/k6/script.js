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

function parseRunConfig() {
  if (!__ENV.RUN_CONFIG_JSON) {
    throw new Error("Missing RUN_CONFIG_JSON. This script requires launcher-provided config.");
  }
  try {
    return JSON.parse(__ENV.RUN_CONFIG_JSON);
  } catch (e) {
    throw new Error(`Invalid RUN_CONFIG_JSON: ${e?.message || e}`);
  }
}

const RUN_CFG = parseRunConfig();
const CFG_PROMPT_LABELS = Array.isArray(RUN_CFG.prompt_labels) && RUN_CFG.prompt_labels.length > 0
  ? RUN_CFG.prompt_labels
  : null;

const BASE_URL         = RUN_CFG.server_url    || "http://localhost:8000";
const STREAM_RATIO     = parseFloat(__ENV.STREAM_RATIO || "0.7");
const API_KEY          = RUN_CFG.api_key       || "";
const CHAT_MODE        = (RUN_CFG.chat_mode ?? false) === true;
const MODEL            = RUN_CFG.model         || "";
const REQUEST_TIMEOUT  = RUN_CFG.request_timeout || RUN_CFG.scenario_definition?.request_timeout || "120s";
const STREAM_RECORD_USAGE = (
  RUN_CFG.stream_record_usage ?? RUN_CFG.scenario_definition?.stream_record_usage ?? false
) === true;
const INITIAL_STAGGER  = (
  RUN_CFG.initial_stagger ?? RUN_CFG.scenario_definition?.initial_stagger ?? false
) === true;
// THINK_TIME: max seconds of sleep between requests per VU (uniform [0, THINK_TIME]).
// Lower values → more in-flight requests → higher KV cache fill. 0 = no sleep.
const THINK_TIME        = parseFloat(RUN_CFG.think_time || "2");
// MAX_TOKENS: when set, overrides the per-prompt max_tokens.
// KV cache fill is driven by the decode phase — longer outputs hold KV blocks longer.
// Use 1024–4096 with kv_stress to keep requests alive and fill the cache.
const MAX_TOKENS       = RUN_CFG.max_tokens ? parseInt(RUN_CFG.max_tokens) : null;

// ─── Shared system prompt (prefix caching) ────────────────────────────────────
// A fixed prefix shared by all requests lets vLLM's automatic prefix caching
// (APC) keep those KV blocks warm across the entire run, boosting cache fill %.
// Use SYSTEM_PROMPT=default to activate the built-in prompt, or pass your own.

const _DEFAULT_SYSTEM_PROMPT =
  "You are a knowledgeable and helpful AI assistant. " +
  "You answer questions clearly and concisely, explain technical concepts at an " +
  "appropriate depth for the user, and always reason step-by-step before giving " +
  "a final answer. When writing code, prefer readability and correctness over " +
  "brevity. If you are unsure about something, say so rather than guessing. " +
  "You follow instructions carefully and stay on topic. Your responses should be " +
  "well-structured: use bullet points or numbered lists when enumerating items, " +
  "and use code blocks for all code snippets regardless of language.";

const SYSTEM_PROMPT = __ENV.SYSTEM_PROMPT === "default" ? _DEFAULT_SYSTEM_PROMPT
                    : (__ENV.SYSTEM_PROMPT || "");

if (SYSTEM_PROMPT) {
  console.log(
    `[init] System prompt enabled (${Math.round(SYSTEM_PROMPT.length / 4)} tokens est.) — ` +
    "prefix caching will share KV blocks across requests"
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

const PROMPTS = new SharedArray("prompts", function () {
  if (!__ENV.PROMPTS_FILE) {
    throw new Error("Missing PROMPTS_FILE. Provide a cluster-visible prompt corpus path.");
  }
  const all    = JSON.parse(open(__ENV.PROMPTS_FILE));
  const filter = CFG_PROMPT_LABELS
    ? new Set(CFG_PROMPT_LABELS)
    : null;
  return filter ? all.filter((p) => filter.has(p.label)) : all;
});
if (PROMPTS.length === 0) {
  throw new Error(
    "No prompts available after filtering. Check prompt_labels against the configured prompt corpus labels."
  );
}
const nConv = PROMPTS.filter((p) => p.messages).length;
console.log(
  `[init] Loaded ${PROMPTS.length} prompts` +
  (CFG_PROMPT_LABELS ? ` (filtered to: ${CFG_PROMPT_LABELS.join(",")})` : "") +
  ` — ${nConv} multi-turn, ${PROMPTS.length - nConv} single-turn`
);

// ─── Metrics ──────────────────────────────────────────────────────────────────

const e2eLatency       = new Trend("e2e_latency_ms", true);
const tokensPerSecond  = new Trend("tokens_per_second");
const promptTokens     = new Trend("prompt_tokens");
const completionTokens = new Trend("completion_tokens");
const requestErrors    = new Counter("request_errors");
const successRate      = new Rate("success_rate");
const activeRequests   = new Gauge("active_requests");

// ─── Scenario loading from JSON ───────────────────────────────────────────────

function scenarioToK6(scenario) {
  // Map scenario JSON to k6 scenario config
  if (!scenario) return null;
  // Support both ramping-vus and constant-vus
  if (scenario.executor === "ramping-vus") {
    return {
      executor: "ramping-vus",
      startVUs: scenario.startVUs || 0,
      stages: scenario.stages,
      gracefulRampDown: scenario.gracefulRampDown || "30s",
    };
  } else {
    return {
      executor: "constant-vus",
      vus: scenario.vus || 10,
      duration: scenario.duration || "5m",
    };
  }
}

const scenarioName = RUN_CFG.scenario || "configured";

function parseCustomStages(raw) {
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error("custom.stages must be a non-empty JSON array");
  }

  for (const stage of parsed) {
    const okDuration = typeof stage?.duration === "string" && stage.duration.length > 0;
    const okTarget = Number.isFinite(stage?.target);
    if (!okDuration || !okTarget) {
      throw new Error(
        "each custom stage must include { duration: string, target: number }"
      );
    }
  }

  return parsed;
}

function buildCustomScenario(custom) {
  if (!custom || !custom.executor) return null;
  if (custom.executor === "ramping-vus") {
    const stages = custom.stages ? parseCustomStages(custom.stages) : [
      { duration: "2m", target: 10 },
      { duration: "5m", target: 10 },
      { duration: "30s", target: 0 },
    ];
    return {
      executor: "ramping-vus",
      startVUs: 0,
      stages,
      gracefulRampDown: custom.ramp_down || "30s",
    };
  }
  return {
    executor: "constant-vus",
    vus: Math.max(1, parseInt(custom.vus || "10") || 10),
    duration: custom.duration || "5m",
  };
}

function buildRealisticScenario(realistic) {
  if (!realistic) return null;
  return {
    executor: "constant-vus",
    vus: Math.max(1, parseInt(realistic.users || "20") || 20),
    duration: realistic.duration || "15m",
  };
}

const _customScenario = RUN_CFG.custom ? buildCustomScenario(RUN_CFG.custom) : null;
const _realisticScenario = RUN_CFG.realistic ? buildRealisticScenario(RUN_CFG.realistic) : null;
const _definedScenario = scenarioToK6(RUN_CFG.scenario_definition);
const _scenarioCandidates = [_customScenario, _realisticScenario, _definedScenario].filter(Boolean);

if (_scenarioCandidates.length === 0) {
  throw new Error(
    "No scenario found in RUN_CONFIG_JSON. Expected one of: custom, realistic, scenario_definition"
  );
}

if (_scenarioCandidates.length > 1) {
  throw new Error(
    "Ambiguous RUN_CONFIG_JSON: provide only one of custom, realistic, scenario_definition"
  );
}

const selectedScenario = _scenarioCandidates[0];

// ─── k6 options ───────────────────────────────────────────────────────────────

export const options = {
  scenarios: {
    load: selectedScenario,
  },
  thresholds: {
    http_req_failed:  ["rate<0.05"],
    success_rate:     ["rate>0.95"],
    e2e_latency_ms:   ["p(95)<30000"],
  },
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

// ─── Request helpers ──────────────────────────────────────────────────────────

const HEADERS = {
  "Content-Type": "application/json",
  ...(API_KEY ? { "Authorization": `Bearer ${API_KEY}` } : {}),
};

const ENDPOINT = CHAT_MODE ? "/v1/chat/completions" : "/v1/completions";

// Build the messages array for a prompt, handling both single-turn and multi-turn formats.
// Prepends the system prompt (if configured) so all requests share a common prefix,
// enabling vLLM APC to reuse the corresponding KV blocks.
function buildMessages(prompt) {
  const body = prompt.messages || [{ role: "user", content: prompt.content }];
  if (!SYSTEM_PROMPT) return body;
  return [{ role: "system", content: SYSTEM_PROMPT }, ...body];
}

// Flatten a messages array to a plain text prompt for the completions endpoint.
// Ends with "\nAssistant:" so the model knows to continue as the assistant.
function flattenMessages(messages) {
  const parts = messages.map((m) =>
    m.role === "user" ? `Human: ${m.content}` : `Assistant: ${m.content}`
  );
  return parts.join("\n") + "\nAssistant:";
}

function payload(prompt, stream) {
  const max_tokens = MAX_TOKENS ?? prompt.max_tokens;
  if (CHAT_MODE) {
    const body = {
      ...(MODEL ? { model: MODEL } : {}),
      messages: buildMessages(prompt),
      max_tokens,
      temperature: 1.0,
      stream,
    };
    if (stream) {
      body.stream_options = { include_usage: true };
    }
    return JSON.stringify(body);
  }
  const base = prompt.messages ? flattenMessages(prompt.messages) : prompt.content;
  const text  = SYSTEM_PROMPT ? `${SYSTEM_PROMPT}\n\n${base}` : base;
  return JSON.stringify({
    prompt: text,
    max_tokens,
    temperature: 1.0,
    stream,
  });
}

function extractText(choice) {
  return CHAT_MODE ? choice?.delta?.content || choice?.message?.content || ""
                   : choice?.text || "";
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
      tokensPerSecond.add((completion / elapsed) * 1000, { label });
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
    } catch (_) {
      // Ignore malformed SSE lines and continue.
    }
  }

  if (usage) return usage;
  if (completionText.length > 0) {
    return {
      prompt_tokens: null,
      completion_tokens: Math.max(1, Math.round(completionText.length / 4)),
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
    tags: { label: prompt.label, stream: "false" },
  });

  activeRequests.add(-1);
  const elapsed = Date.now() - start;
  e2eLatency.add(elapsed, { label: prompt.label });

  let body = null;
  try { body = JSON.parse(res.body); } catch (_) {}

  const ok = check(res, {
    "status 200":  (r) => r.status === 200,
    "has choices": ()  => body?.choices?.length > 0,
  });

  successRate.add(ok);

  if (!ok) {
    requestErrors.add(1, { status: res.status, label: prompt.label });
    console.warn(`[non-stream] ${prompt.label} failed: ${res.status} ${res.body?.slice(0, 200)}`);
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
    tags: { label: prompt.label, stream: "true" },
  };
  if (!STREAM_RECORD_USAGE) {
    // Avoid buffering large SSE bodies unless usage accounting is enabled.
    streamReqOpts.responseType = "none";
  }
  const res = http.post(`${BASE_URL}${ENDPOINT}`, payload(prompt, true), streamReqOpts);

  activeRequests.add(-1);
  const elapsed = Date.now() - start;
  e2eLatency.add(elapsed, { label: prompt.label });

  const ok = check(res, { "status 200": (r) => r.status === 200 });

  successRate.add(ok);

  if (!ok) {
    requestErrors.add(1, { status: res.status, label: prompt.label });
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
  medium:      20,
  long_output: 30,
  long_input:  20,
  xl_input:    10,
  conv_medium: 10,
  conv_long:   10,
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
  RUN_CFG.label_weights || RUN_CFG.scenario_definition?.label_weights
);

// Cumulative weight array — avoids duplicating prompt objects into a flat pool.
const _CUM_WEIGHTS = PROMPTS.reduce((acc, p, i) => {
  const w = LABEL_WEIGHTS[p.label] ?? 10;
  acc.push((acc[i - 1] || 0) + w);
  return acc;
}, []);
const _TOTAL_WEIGHT = _CUM_WEIGHTS[_CUM_WEIGHTS.length - 1];
if (!Number.isFinite(_TOTAL_WEIGHT) || _TOTAL_WEIGHT <= 0) {
  throw new Error("Invalid label weights: total prompt weight must be > 0");
}

function pickPrompt() {
  const r = Math.random() * _TOTAL_WEIGHT;
  let lo = 0, hi = _CUM_WEIGHTS.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (_CUM_WEIGHTS[mid] < r) lo = mid + 1;
    else hi = mid;
  }
  return PROMPTS[lo];
}

// ─── VU entrypoint ────────────────────────────────────────────────────────────

// Track whether this VU has run its first iteration yet.
// Used when initial_stagger=true in RUN_CONFIG_JSON.
const _vuStarted = { done: false };

export default function () {
  // Optional initial staggering: offset each VU's first request by a random
  // fraction of THINK_TIME so all VUs don't pile up at t=0.
  if (INITIAL_STAGGER && !_vuStarted.done) {
    _vuStarted.done = true;
    if (THINK_TIME > 0) sleep(Math.random() * THINK_TIME);
  }

  const prompt = pickPrompt();

  if (Math.random() < STREAM_RATIO) {
    runStreaming(prompt);
  } else {
    runNonStreaming(prompt);
  }

  if (THINK_TIME > 0) sleep(Math.random() * THINK_TIME);
}

// ─── Summary ──────────────────────────────────────────────────────────────────

export function handleSummary(data) {
  const m = data.metrics;

  const ms  = (metric, stat) => metric ? `${(metric.values[stat] ?? 0).toFixed(0)}ms` : "n/a";
  const num = (metric, stat) => metric?.values[stat] ?? 0;

  // Per-label latency breakdown from tagged sub-metrics
  const labelBreakdown = {};
  for (const [key, metric] of Object.entries(m)) {
    const match = key.match(/^e2e_latency_ms\{label:(.+)\}$/);
    if (match) {
      labelBreakdown[match[1]] = {
        p50: ms(metric, "med"),
        p95: ms(metric, "p(95)"),
        avg: ms(metric, "avg"),
      };
    }
  }

  const summary = {
    scenario: scenarioName,
    thresholds: Object.entries(m)
      .filter(([, v]) => v.thresholds)
      .map(([name, v]) => ({
        metric: name,
        ok: Object.values(v.thresholds).every((t) => t.ok),
      })),
    latency: {
      e2e_p50: ms(m.e2e_latency_ms, "med"),
      e2e_p95: ms(m.e2e_latency_ms, "p(95)"),
      e2e_p99: ms(m.e2e_latency_ms, "p(99)"),
    },
    latency_by_label: labelBreakdown,
    throughput: {
      total_requests: num(m.http_reqs, "count"),
      req_per_sec:    num(m.http_reqs, "rate").toFixed(2),
      success_rate:   (num(m.success_rate, "rate") * 100).toFixed(1) + "%",
      error_count:    num(m.request_errors, "count"),
    },
    tokens: {
      avg_completion_tokens: num(m.completion_tokens, "avg").toFixed(1),
      avg_tokens_per_second: num(m.tokens_per_second, "avg").toFixed(1),
    },
  };

  return {
    stdout: `\n=== LLM Load Test Summary ===\n${JSON.stringify(summary, null, 2)}\n`,
    "summary.json": JSON.stringify(summary, null, 2),
  };
}
