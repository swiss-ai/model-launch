# ADR-0001: The replica is the orchestration unit

**Status**: Accepted (2026-04-26)

## Context

SML schedules inference jobs on SLURM. A request needs to land on a process that holds the model weights. The orchestration layer needs a clear unit to reason about — what gets scheduled, what gets load-balanced, what scales.

The frameworks (sglang, vLLM) have several internal scaling dimensions:

- **TP** (tensor parallelism) — splits one model across N GPUs in one process
- **PP** (pipeline parallelism) — splits layers across devices in one process
- **DP** (data parallelism) — replicates the model N times within one process; one HTTP endpoint, internal scheduling
- **EP** (expert parallelism) — splits MoE experts across devices

All of these run **inside a single framework process** that exposes one HTTP endpoint.

We could expose any of these as orchestration concepts, or none. We chose to expose none and call the framework process itself a *replica*.

## Decision

A **replica** is one independent inference engine instance — one framework process exposing one HTTP endpoint at port 8080. It is the unit of orchestration:

- A SLURM job is `replicas × nodes_per_replica` nodes.
- A replica spans a contiguous set of nodes (`nodes_per_replica`) and is wrapped in OCF on its head node.
- The router (sglang_router today) load-balances **across replicas**, not within them.
- Internal sharding (TP/PP/DP/EP) is the user's concern, configured via free-form `framework_args`. SML does not infer or inject parallelism flags.

## Consequences

### Supported

- Multi-replica fanout (`replicas >= 2`): N independent processes, each on its own node-set, router fans out across them. This is the canonical multi-tenant throughput pattern.
- Multi-node single-replica (`replicas=1, nodes_per_replica >= 2`): one framework process spread across multiple nodes via the framework's own distributed-init mechanism (sglang `--dist-init-addr`, vLLM Ray cluster).

### Deliberately not supported

- **Multiple framework processes on a single node.** A user might want to run 4×TP=1 sglang processes on a 4-GPU node and load-balance across them (via the router or OCF, both of which have the plumbing). To support this we would need:
  - A new topology dimension (`processes_per_node` or similar).
  - Per-process port allocation, undoing the hardcoded `FRAMEWORK_PORT=8080` (see ADR-0002 if/when written).
  - Per-process OCF port allocation (8092/8093/43905 currently collide between OCF instances on a host).
  - GPU pinning via `CUDA_VISIBLE_DEVICES`.
  - User-level mental model: replicas-per-node vs replicas-per-job.
  
  This is real complexity bringing back exactly the kind of multi-source-of-truth port management we just collapsed. The optimisation it unlocks (DP fanout on a single host) is mostly already handled by sglang's internal scheduler with `--dp-size N`. We defer this until there's concrete demand.

- **`use_router=True` with `replicas=1`.** Validates as an error: there's nothing to load-balance across. (sglang's internal DP gives one HTTP endpoint regardless of how many DP workers it runs.)

## Alternatives considered

1. **Expose TP/PP/DP/EP as topology fields.** Rejected: parallelism choices are hardware/model-specific, easy to misconfigure into OOM, and frameworks evolve their own flags. SML staying out of this lets users follow framework docs without translation.

2. **Allow `replicas=1` with `use_router=True` as a no-op or stable-URL pattern.** Rejected: router with one backend is pure overhead with no load-balancing value, and silently no-op'ing surprises users.

3. **Process-per-node as a first-class topology dimension.** Deferred (see "deliberately not supported" above).

## Related

- The framework HTTP port is hardcoded to 8080 across the system to make `curl http://<replica-ip>:8080/...` work predictably regardless of framework or job. This decision underlies the multi-process-per-node restriction above.
