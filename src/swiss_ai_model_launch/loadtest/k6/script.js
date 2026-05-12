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
 *   SYSTEM_PROMPT       - prepend a fixed system prompt to every chat request, enabling
 *                         vLLM prefix caching (APC) to share KV blocks across requests.
 *                         Set to "default" to use the built-in ~200-token prompt, or pass
 *                         your own string.  Empty string (default) disables it.
 *                         NOTE: requires vllm serve --enable-prefix-caching
 */

import http from "k6/http";
import { check, sleep } from "k6";
import exec from "k6/execution";
import { Trend, Counter, Rate, Gauge } from "k6/metrics";
import { SharedArray } from "k6/data";

// ─── Config ───────────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_REQUEST_TIMEOUT = "120s";
const DEFAULT_PROMPT_SEED = 1;
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
const SCENARIO_CONSTANT_ARRIVAL_RATE = "constant-arrival-rate";
const SCENARIO_RAMPING_ARRIVAL_RATE = "ramping-arrival-rate";
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

function parseInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
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
const API_KEY = RUN_CFG.api_key ?? "";
const MODEL = RUN_CFG.model ?? "";
const REQUEST_TIMEOUT =
  RUN_CFG.request_timeout ??
  RUN_CFG.scenario_definition?.request_timeout ??
  DEFAULT_REQUEST_TIMEOUT;
const INITIAL_STAGGER =
  (RUN_CFG.initial_stagger ??
    RUN_CFG.scenario_definition?.initial_stagger ??
    false) === true;
const IGNORE_EOS =
  (RUN_CFG.ignore_eos ?? RUN_CFG.scenario_definition?.ignore_eos ?? false) ===
  true;
const PROMPT_SEED = parseInteger(
  __ENV.PROMPT_SEED ?? RUN_CFG.prompt_seed ?? RUN_CFG.scenario_definition?.prompt_seed,
  DEFAULT_PROMPT_SEED,
);
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
  if (scenario.executor === SCENARIO_RAMPING_VUS) {
    return {
      executor: SCENARIO_RAMPING_VUS,
      startVUs: scenario.startVUs ?? 0,
      stages: scenario.stages,
      gracefulRampDown: scenario.gracefulRampDown ?? DEFAULT_RAMP_DOWN,
    };
  }
  if (scenario.executor === SCENARIO_CONSTANT_ARRIVAL_RATE) {
    return {
      executor: SCENARIO_CONSTANT_ARRIVAL_RATE,
      rate: parsePositiveInteger(scenario.rate, 1),
      timeUnit: scenario.timeUnit ?? "1s",
      duration: scenario.duration ?? DEFAULT_DURATION,
      preAllocatedVUs: parsePositiveInteger(
        scenario.preAllocatedVUs,
        DEFAULT_MAX_VUS,
      ),
      maxVUs: parsePositiveInteger(
        scenario.maxVUs,
        scenario.preAllocatedVUs ?? DEFAULT_MAX_VUS,
      ),
    };
  }
  if (scenario.executor === SCENARIO_RAMPING_ARRIVAL_RATE) {
    return {
      executor: SCENARIO_RAMPING_ARRIVAL_RATE,
      startRate: parsePositiveInteger(scenario.startRate, 1),
      timeUnit: scenario.timeUnit ?? "1s",
      stages: scenario.stages,
      preAllocatedVUs: parsePositiveInteger(
        scenario.preAllocatedVUs,
        DEFAULT_MAX_VUS,
      ),
      maxVUs: parsePositiveInteger(
        scenario.maxVUs,
        scenario.preAllocatedVUs ?? DEFAULT_MAX_VUS,
      ),
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
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

// ─── Request helpers ──────────────────────────────────────────────────────────

const HEADERS = {
  "Content-Type": "application/json",
  ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
};

const ENDPOINT = "/v1/chat/completions";

// Build the messages array for a prompt, handling both single-turn and multi-turn formats.
// Prepends the system prompt (if configured) so all requests share a common prefix,
// enabling vLLM APC to reuse the corresponding KV blocks.
function buildMessages(prompt) {
  const body = prompt.messages ?? [{ role: "user", content: prompt.content }];
  if (!SYSTEM_PROMPT) return body;
  return [{ role: "system", content: SYSTEM_PROMPT }, ...body];
}

function payload(prompt) {
  const maxTokens = MAX_TOKENS ?? prompt.max_tokens;
  const body = {
    ...(MODEL ? { model: MODEL } : {}),
    messages: buildMessages(prompt),
    max_tokens: maxTokens,
    temperature: 1.0,
    stream: false,
  };
  if (IGNORE_EOS) {
    body.ignore_eos = true;
  }
  return JSON.stringify(body);
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

// ─── Non-streaming ────────────────────────────────────────────────────────────

function runNonStreaming(prompt) {
  activeRequests.add(1);
  const start = Date.now();

  const res = http.post(`${BASE_URL}${ENDPOINT}`, payload(prompt), {
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
    const errorDetail = res.error
      ? ` error=${res.error}`
      : res.error_code
        ? ` error_code=${res.error_code}`
        : "";
    const durationDetail = Number.isFinite(res.timings?.duration)
      ? ` duration_ms=${res.timings.duration.toFixed(0)}`
      : "";
    console.warn(
      `[non-stream] ${prompt.label} failed: status=${res.status}${errorDetail}${durationDetail} body=${res.body?.slice(0, 200)}`,
    );
    return;
  }

  recordUsage(body?.usage, elapsed, prompt.label);
}

// ─── Prompt selection ────────────────────────────────────────────────────────

function seededRandom(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function shuffledPromptIndices(length, seed) {
  const indices = Array.from({ length }, (_, i) => i);
  const random = seededRandom(seed);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

const PROMPT_ORDER = shuffledPromptIndices(PROMPTS.length, PROMPT_SEED);
console.log(`[init] Prompt order shuffled with seed ${PROMPT_SEED}`);

function pickPrompt() {
  const promptIndex =
    PROMPT_ORDER[exec.scenario.iterationInTest % PROMPT_ORDER.length];
  return PROMPTS[promptIndex];
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

  runNonStreaming(prompt);

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
