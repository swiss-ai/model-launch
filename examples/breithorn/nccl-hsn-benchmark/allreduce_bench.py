#!/usr/bin/env python3
"""Inter-node NCCL collective bandwidth sweep: all-reduce AND all-to-all.

Runs one process per GPU across >=2 nodes and measures algbw / busbw for a
range of message sizes. Rank 0 prints a table per collective. The ONLY thing
that differs between the HSN and TCP runs is the NCCL transport (chosen by env
vars in the pod spec); this script is transport-agnostic.

Two collectives, because dense and MoE models stress the fabric differently:

  all_reduce    — the dense collective (tensor-parallel, DP gradient sync).
                  Structured ring/tree traffic, bandwidth-bound.
                  busbw factor = 2*(n-1)/n   (nccl-tests convention)

  all_to_all    — the collective that DEFINES MoE (expert-parallel dispatch +
                  combine, 2 per MoE layer). N^2 pairwise flows, incast-heavy,
                  latency- and bisection-bandwidth-bound. This is where a good
                  fabric (Slingshot: adaptive routing + congestion control)
                  pulls ahead of TCP the most.
                  busbw factor = (n-1)/n

Note on scale: at world_size=2 an all-to-all is just a pairwise swap, so the
incast advantage is muted. To show the real MoE story, scale to 8 GPUs/node
(torchrun --nproc_per_node=8, WORLD_SIZE=16) so every rank fans out to 15
peers simultaneously.

Env (set by the launcher / pod spec):
  RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT   torch rendezvous
  LOCAL_RANK                                    GPU index on this node
"""

import os
import time

import torch
import torch.distributed as dist


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.0f}TiB"


def time_op(fn, iters: int) -> float:
    """Mean seconds per call, with a barrier so all ranks start together."""
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def sweep(name: str, make_op, busbw_factor: float, rank: int, world: int, dev) -> None:
    if rank == 0:
        print(f"\n## {name}  (busbw factor {busbw_factor:.3f})", flush=True)
        print(
            f"{'size':>10} {'iters':>6} {'time_ms':>10} {'algbw_GB/s':>12} {'busbw_GB/s':>12}",
            flush=True,
        )
    # 1 KiB -> 8 GiB, x4 per step.
    for nbytes in [2**e for e in range(10, 34, 2)]:
        fn, teardown = make_op(nbytes, world, dev)
        if fn is None:  # size not valid for this collective at this world size
            continue
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()

        iters = 20 if nbytes < 2**26 else 8
        dt = time_op(fn, iters)

        algbw = nbytes / dt / 1e9
        busbw = algbw * busbw_factor
        if rank == 0:
            print(
                f"{human(nbytes):>10} {iters:>6} {dt * 1e3:>10.3f} {algbw:>12.2f} {busbw:>12.2f}",
                flush=True,
            )
        teardown()


def make_all_reduce(nbytes, world, dev):
    x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    return (lambda: dist.all_reduce(x)), (lambda: None)


def make_all_to_all(nbytes, world, dev):
    # nbytes is the total buffer per rank; it must divide evenly into `world`
    # equal shards (one destined for each peer).
    n = nbytes // 4
    if n % world != 0:
        return None, None
    inp = torch.ones(n, dtype=torch.float32, device=dev)
    out = torch.empty_like(inp)
    return (lambda: dist.all_to_all_single(out, inp)), (lambda: None)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"# world_size={world}  gpu={torch.cuda.get_device_name(dev)}", flush=True)
        print(f"# torch={torch.__version__}  nccl={torch.cuda.nccl.version()}", flush=True)

    # Dense collective: tensor-parallel / DP.
    sweep("all_reduce", make_all_reduce, 2 * (world - 1) / world, rank, world, dev)
    # MoE collective: expert-parallel dispatch/combine.
    sweep("all_to_all", make_all_to_all, (world - 1) / world, rank, world, dev)

    if rank == 0:
        print("\n# done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
