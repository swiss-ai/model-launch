# Proving Slingshot (HSN) beats TCP for NCCL

A minimal, apples-to-apples benchmark that shows the inter-node fabric change
is real and measurable. Same image (`ghcr.io/swiss-ai/vllm_cxi`), same GPUs,
same all-reduce workload — the **only** difference is the NCCL transport.

## What it measures

`allreduce_bench.py` runs over 2 GPUs on 2 different nodes and reports `busbw`
(bus bandwidth, the nccl-tests definition) across message sizes 1 KiB → 8 GiB
for **two collectives**:

- **all_reduce** — the dense collective (tensor-parallel, DP gradient sync).
- **all_to_all** — the collective that defines **MoE** (expert-parallel
  dispatch/combine). Incast-heavy and latency-sensitive — the case where
  Slingshot's adaptive routing + congestion control should pull _furthest_
  ahead of TCP. At `world_size=2` it's only a pairwise swap; scale to 8
  GPUs/node (`torchrun --nproc_per_node=8`, `WORLD_SIZE=16`) to expose real
  all-to-all incast.

Two arms:

| Arm       | Manifest         | Transport             | Key env                                                           |
| --------- | ---------------- | --------------------- | ----------------------------------------------------------------- |
| Control   | `bench-tcp.yaml` | TCP sockets (pod net) | `NCCL_NET=Socket`, `NCCL_NET_PLUGIN=none`                         |
| Treatment | `bench-hsn.yaml` | Slingshot via CXI     | `FI_PROVIDER=cxi`, aws-ofi-nccl loaded, hostNetwork + `/dev/cxi*` |

## Run it

Cluster: **breithorn** (kube context `breithorn-oidc`, namespace `rob-poc`).

```bash
CTX="--context breithorn-oidc -n rob-poc"
kubectl $CTX create configmap nccl-bench-src --from-file=allreduce_bench.py

# Control (TCP)
kubectl $CTX apply -f bench-tcp.yaml
kubectl $CTX logs -f bench-tcp-master
kubectl $CTX delete -f bench-tcp.yaml

# Treatment (Slingshot)
kubectl $CTX apply -f bench-hsn.yaml
kubectl $CTX logs -f bench-hsn-master
kubectl $CTX delete -f bench-hsn.yaml
```

## Two independent proofs in the output

1. **Quantitative** — the `busbw_GB/s` column. Expect TCP to plateau at a few
   GB/s while HSN climbs into the tens of GB/s per rail (higher with multi-rail,
   see below), and HSN wins latency at small sizes too.

2. **Qualitative** — with `NCCL_DEBUG=INFO` the log states the transport it
   picked. That is an unambiguous receipt that the fabric actually changed:
   - HSN: a `NET/OFI ... Selected provider is cxi` line
   - TCP: a `NET/Socket` line (no OFI/cxi line)

## Notes / knobs before you trust a number

- **Different nodes.** `podAntiAffinity` forces the two ranks onto different
  hosts, otherwise NCCL would use NVLink/SHM and you'd measure nothing about
  the fabric. Confirm with `kubectl get pod -o wide`.
- **HSN nodes + paths.** `bench-hsn.yaml` needs nodes that expose `/dev/cxi*`.
  Set the `nodeSelector` and fix the libfabric version in `LD_LIBRARY_PATH`
  (`ls /opt/cray/libfabric/` on the node — the `1.15.2.0` there is a placeholder).
  Mount `/dev/cxi1..3` as extra volumes if the nodes have >1 rail.
- **This is 1 GPU/node (single rail).** Already enough to show HSN >> TCP. For
  the _full_ multi-rail story (`NCCL_CROSS_NIC=1` striping over hsn0–3), scale to
  8 GPUs/node with `torchrun --nproc_per_node=8` and `WORLD_SIZE=16`.
- **Both collectives run.** all-reduce covers tensor-/data-parallel traffic;
  all-to-all covers MoE expert-parallel. The MoE (all-to-all) advantage grows
  with world size — the 2-GPU run understates it, so scale up before quoting a
  number to MoE skeptics.
