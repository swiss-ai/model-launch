# How to Size a Model for a GPU / Node

Picking the right replica count and nodes-per-replica boils down to: **does the model fit in VRAM, and how much spare VRAM do I want for the KV cache?**

## Step 1 — VRAM the weights need

Rough formula:

```text
weights_bytes ≈ params × bytes_per_param
```

| Precision | Bytes / param | Example: 70B model |
| --------- | ------------- | ------------------ |
| FP32      | 4             | 280 GB             |
| BF16/FP16 | 2             | 140 GB             |
| FP8       | 1             | 70 GB              |
| INT4      | 0.5           | 35 GB              |

Add **~20% overhead** for activations, framework buffers, and CUDA workspaces.

## Step 2 — VRAM the KV cache needs

KV cache scales with concurrent sequences and context length. For a transformer:

```text
kv_bytes_per_token ≈ 2 × num_layers × hidden_dim × kv_heads/heads × bytes_per_param
```

Then:

```text
kv_total ≈ kv_bytes_per_token × max_concurrent_tokens
```

Where `max_concurrent_tokens` is roughly `max_batch × max_seq_len`. If you're not sure, start by reserving **30–50% of VRAM** for the KV cache — both sglang and vLLM size their cache to fill what's left after weights.

## Step 3 — pick a GPU layout

CSCS GH200 nodes have 4 GPUs at ~96 GB each (~384 GB per node).

| Model size (BF16) | Fits where                                | Layout                                                                        |
| ----------------- | ----------------------------------------- | ----------------------------------------------------------------------------- |
| ≤ 30 B            | 1 GPU                                     | `--slurm-replicas N --slurm-nodes-per-replica 1`, set framework `--tp-size 1` |
| 30–80 B           | 1 node (4-way TP)                         | 1 replica per node, framework `--tp-size 4`                                   |
| 80–250 B          | 1 node (4-way TP) at FP8, or 2 nodes BF16 | quantize, or `--slurm-nodes-per-replica 2` + matching TP                      |
| 250 B+            | Multiple nodes                            | `--slurm-nodes-per-replica 2+`, expect tensor + pipeline parallelism          |

## Parallelism: DP / TP / PP / EP — and why DP is replicas

Four flavors of parallelism show up when serving large models:

| Term                          | What it splits across GPUs                                        | Where SML expresses it                                                                                                            |
| ----------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **TP** (tensor parallelism)   | A single matmul, sharded across GPUs within a layer               | Framework flag (e.g. sglang/vLLM `--tp-size`) inside `--framework-args`. Stays inside one replica.                                |
| **PP** (pipeline parallelism) | Layers, sharded across GPUs (or nodes) end-to-end                 | Framework flag (e.g. `--pp-size`) inside `--framework-args`. Spans nodes within one replica when `--slurm-nodes-per-replica > 1`. |
| **EP** (expert parallelism)   | MoE experts, sharded across GPUs — only meaningful for MoE models | Framework flag (e.g. vLLM/sglang `--ep-size` or `--enable-expert-parallel`) inside `--framework-args`. Stays inside one replica.  |
| **DP** (data parallelism)     | Independent copies serving different requests in parallel         | **`--slurm-replicas N`** — N copies of the model, optionally fronted by `--use-router`.                                           |

In short: **a "replica" in SML is a DP unit.** TP, PP, and EP are framework-internal — they affect how one replica is laid out across its allocated GPUs/nodes. DP is just "how many replicas".

### A note on dense models in Kubernetes

For dense models (one weight matrix per layer, no MoE routing), DP isn't usually expressed inside the inference framework — you don't tell the framework "give me 4 data-parallel copies on these 4 GPUs". You just request a single GPU per replica and let the **autoscaler** add more replicas when load grows. The orchestrator (k8s, or here, SLURM + `--slurm-replicas`) provides DP naturally; the framework only handles TP (and PP when needed).

This shapes the rule below: bump `--slurm-replicas` for throughput, not the framework's DP flags.

### MoE models change the picture

For Mixture-of-Experts models (Mixtral, DeepSeek-V3, GLM-4.5/5, Qwen-MoE, …), the choice between TP and EP matters:

- **TP** shards each expert's weight matrices across GPUs. Communication is on the critical path of every token.
- **EP** keeps each expert whole on one GPU and routes tokens to the GPU that owns the expert they were assigned to. Communication is one all-to-all per MoE layer, but per-expert matmuls stay local.

Rule of thumb: for MoE models with many experts and modest expert size, **prefer EP over TP within a replica** — it's typically faster on multi-GPU nodes. Use TP for the dense (attention) parts and EP for the MoE feed-forward parts when the framework supports it (most modern serving stacks do).

DP across replicas still applies the same way for throughput: more concurrent requests → bump `--slurm-replicas`.

## Step 4 — replicas vs. nodes-per-replica

These two flags set very different things:

- **`--slurm-replicas N`** — N independent copies of the model. Use for **throughput**: more concurrent requests, optionally fronted by `--use-router` for load balancing.
- **`--slurm-nodes-per-replica K`** — each replica spans K nodes. Use when **one replica doesn't fit on a single node** (large models, long context, more KV cache).

Total nodes = `replicas × nodes-per-replica`.

Rule of thumb:

- Model fits on 1 node, want more throughput? → bump `--slurm-replicas`.
- Model doesn't fit on 1 node? → bump `--slurm-nodes-per-replica` first, then add replicas if you still need throughput.

## Step 5 — sanity-check before submitting

- Time limit (`--slurm-time`) covers warm-up + your workload + a margin. Cold start of a multi-node deployment can take sometimes up to 40 minutes (e.g. Kimi-k2.5 1T params).
- Partition matches the GPU layout you're asking for.
- KV cache leaves room for your max sequence length × max batch.

## Latency tuning

Use this when **a single user is waiting for a response** — chat, interactive demos, copilot-style autocomplete. The metric to optimize is TTFT and per-token latency at low concurrency.

| Knob | Recommended for low latency |
| --- | --- |
| Model size | The smallest model that meets your quality bar. A well-tuned 8B is faster than a clumsily-tuned 70B. |
| Precision | FP8 or INT4 if accuracy holds. Less VRAM read per token = faster. |
| Replicas | **1.** More replicas help throughput, not single-request latency. |
| Router | **Off** (`--use-router` not set). The router adds a hop. |
| Framework batching | Keep `--max-num-seqs` low (e.g. 8) so requests don't queue behind a giant batch. |
| Context length | Cap `--max-model-len` to what you actually need. Smaller KV cache = faster prefill. |
| TP | Just enough to fit the model. Past that, TP communication starts costing more than it saves. |
| OCF/OpenTela | If you're driving load directly from another job on the cluster, `--disable-opentela` removes the mesh hop. For end-user traffic via the public gateway, keep it on. |

Measure TTFT and P50/P99 at concurrency = 1 and concurrency = your realistic ceiling — they will tell different stories. See [Benchmarking](benchmarking.md).

## Throughput tuning

Use this when **you have a lot of work to push through** — batch eval, dataset processing, offline scoring. The metric to optimize is tokens/sec aggregated across all requests.

| Knob | Recommended for high throughput |
| --- | --- |
| Replicas | **More.** Bump `--slurm-replicas` until you hit a partition or budget cap. DP scales linearly. |
| Router | **On** (`--use-router`). Spreads load across replicas; without it you have to load-balance yourself. |
| Framework batching | Crank `--max-num-seqs` (e.g. 256+) so the framework can group requests into fat batches. |
| KV cache headroom | Leave more VRAM for the cache. Bigger cache = more concurrent sequences = more batching opportunity. |
| Precision | FP8 if quality allows — smaller weights leave more room for KV cache and increase batch size. |
| Context length | Cap `--max-model-len` to the longest request you'll actually send. Wasted KV cache = lost batch slots. |
| Concurrency at the client | Don't ramp slower than the server can absorb — keep ≥ `replicas × max-num-seqs` requests in flight. |

If you're benchmarking, **disable OpenTela** to take the mesh hop out of the measurement (see [When to disable OpenTela](usage-advanced.md#when-to-disable-opentela)).

## When in doubt

Start with one replica on one node at the lowest precision your accuracy budget tolerates. Measure (see [Benchmarking](benchmarking.md)). Scale from there.

## Next

- [Benchmarking](benchmarking.md) — measure before scaling
- [Advanced Usage](usage-advanced.md) — the flags above in context
