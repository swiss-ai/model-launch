"""Render the SLURM job submission scripts from Python.

The output is **multi-file**:

- ``master.sh`` — the SBATCH-submitted entrypoint. Telemetry, arch detection,
  node mapping, per-replica IP discovery, then srun calls dispatching to the
  rank scripts.
- ``head.sh`` — what runs on rank 0 of every replica (always present).
- ``follower.sh`` — what runs on ranks 1..N-1 of multi-node replicas (only
  present when ``nodes_per_replica > 1``).
- ``router.sh`` — only present when ``use_router=True`` and ``replicas > 1``.

Each rank script is a normal idiomatic bash file: shellcheckable on its own,
no nested quoting concerns. The master invokes them via ``bash <path> args...``
with positional args for the runtime-determined values (``$replica_N_head_ip``,
``$node_rank``).

This shape keeps the worst pre-existing pain — bash text embedded inside a
``bash -c "..."`` arg whose outer-shell processing forces manual escape
juggling — entirely out of the picture.
"""

from __future__ import annotations

from typing import ClassVar

from swiss_ai_model_launch.launchers.launch_args import (
    FRAMEWORK_PORT,
    LaunchArgs,
)

SGLANG_ROUTER_PORT = 30000
OCF_BOOTSTRAP_ADDR = "/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"
RAY_PORT = 6379
NUM_GPUS_PER_NODE = 4
SGLANG_DIST_INIT_PORT = 5757

_VMAGENT_SCRAPE_CONFIG = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent-scrape.yaml"


# ── frameworks ──────────────────────────────────────────────────────────────


class Framework:
    name: ClassVar[str]
    entrypoint: ClassVar[str]
    env_exports: ClassVar[list[str]]


class Sglang(Framework):
    name = "sglang"
    entrypoint = "python3 -m sglang.launch_server"
    env_exports = [
        'export no_proxy="0.0.0.0,$no_proxy"',
        'export NO_PROXY="0.0.0.0,$NO_PROXY"',
        # JIT DeepGEMM can be unstable on some GPU/model combos. SGL_* is the
        # historical upstream env-var name; SGLANG_* is the newer one. Both
        # are exported during the upstream transition.
        'export SGL_ENABLE_JIT_DEEPGEMM="false"',
        'export SGLANG_ENABLE_JIT_DEEPGEMM="false"',
    ]


class Vllm(Framework):
    name = "vllm"
    entrypoint = "python3 -m vllm.entrypoints.openai.api_server"
    env_exports = [
        "export RAY_CGRAPH_get_timeout=1800",
        'export no_proxy="0.0.0.0,$no_proxy"',
        'export NO_PROXY="0.0.0.0,$NO_PROXY"',
    ]


_FRAMEWORKS: dict[str, type[Framework]] = {"sglang": Sglang, "vllm": Vllm}


def _make_framework(name: str) -> Framework:
    try:
        return _FRAMEWORKS[name]()
    except KeyError:
        known = ", ".join(_FRAMEWORKS)
        raise ValueError(f"Unknown framework: {name!r}. Known: {known}") from None


def _compose_framework_args(launch_args: LaunchArgs) -> str:
    return f"--port {FRAMEWORK_PORT} {launch_args.framework_args}".strip()


def _ocf_wrap(inner_cmd: str) -> str:
    """Wrap a command in OCF's ``--subprocess`` invocation."""
    return (
        f"$OCF_BIN start \\\n"
        f'    --bootstrap.addr "{OCF_BOOTSTRAP_ADDR}" \\\n'
        f"    --service.name llm \\\n"
        f"    --service.port {FRAMEWORK_PORT} \\\n"
        f'    --subprocess "{inner_cmd}"'
    )


def _shebang_and_setup(framework: Framework, pre_launch_cmds: str) -> str:
    """Common header for every rank script: shebang, set -ex, env exports,
    optional pre-launch hook."""
    lines = [
        "#!/bin/bash",
        # SC2046/SC2086: user-supplied framework_args is inlined bare on the
        # python3 -m ... command line. Constructs like ``$(whoami)`` in the
        # args are intentional (and safe in practice since usernames don't
        # contain spaces).
        "# shellcheck disable=SC2046,SC2086",
        "set -ex",
        "",
    ]
    lines.extend(framework.env_exports)
    if pre_launch_cmds:
        lines += [
            "",
            "# User-supplied pre-launch commands",
            "echo 'Running pre-launch commands...'",
            pre_launch_cmds,
        ]
    return "\n".join(lines)


# ── rank script renderers ──────────────────────────────────────────────────


def _render_sglang_head(launch_args: LaunchArgs, framework: Framework) -> str:
    args = _compose_framework_args(launch_args)
    npr = launch_args.topology.nodes_per_replica
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    use_ocf = not launch_args.disable_ocf

    if npr == 1:
        # Singular: one rank per replica, the head IS the only rank.
        # ``$1`` is replica_head_ip (passed by master, unused here but kept
        # for signature symmetry with the multi-node case).
        body_args = '# shellcheck disable=SC2034\nreplica_head_ip="$1"\n'
        cmd = f"{framework.entrypoint} {args}"
    else:
        body_args = 'replica_head_ip="$1"\n# Multi-node head: --node-rank is always 0\n'
        cmd = (
            f"{framework.entrypoint} \\\n"
            f'    --dist-init-addr "$replica_head_ip:{SGLANG_DIST_INIT_PORT}" \\\n'
            f"    --nnodes {npr} \\\n"
            f"    --node-rank 0 \\\n"
            f"    {args}"
        )

    if use_ocf:
        # OCF spawns the launch as a subprocess so it can be advertised on
        # the OpenTela network at $service.port.
        launch = _ocf_wrap(cmd)
    else:
        launch = cmd
    return f"{pre}\n\n{body_args}\n{launch}\n"


def _render_sglang_follower(launch_args: LaunchArgs, framework: Framework) -> str:
    args = _compose_framework_args(launch_args)
    npr = launch_args.topology.nodes_per_replica
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    return (
        f"{pre}\n\n"
        f'node_rank="$1"\n'
        f'replica_head_ip="$2"\n'
        f"\n"
        f"{framework.entrypoint} \\\n"
        f'    --dist-init-addr "$replica_head_ip:{SGLANG_DIST_INIT_PORT}" \\\n'
        f"    --nnodes {npr} \\\n"
        f'    --node-rank "$node_rank" \\\n'
        f"    {args}\n"
    )


def _render_vllm_head(launch_args: LaunchArgs, framework: Framework) -> str:
    args = _compose_framework_args(launch_args)
    npr = launch_args.topology.nodes_per_replica
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    use_ocf = not launch_args.disable_ocf

    if npr == 1:
        # Singular: just run the API server directly, no Ray bootstrap.
        body_args = '# shellcheck disable=SC2034\nreplica_head_ip="$1"\n'
        cmd = f"{framework.entrypoint} {args}"
        if use_ocf:
            launch = _ocf_wrap(cmd)
        else:
            launch = cmd
        return f"{pre}\n\n{body_args}\n{launch}\n"

    # Multi-node head: stage the Ray bootstrap + API server invocation as a
    # script on /tmp (single-quoted heredoc keeps $-constructs literal in
    # the file), then either run it directly or via OCF's --subprocess.
    # PR #124 introduced the on-disk pattern to dodge OCF's subprocess
    # re-evaluation; we keep it because it's still correct.
    expected_gpus = npr * NUM_GPUS_PER_NODE
    body_args = (
        "# shellcheck disable=SC2034  # unused on the head but kept for signature symmetry\n"
        'replica_head_ip="$1"\n'
        'ray_head_script="/tmp/sml-ray-head-${SLURM_JOB_ID}.sh"\n'
    )
    head_script_body = (
        f"cat > \"$ray_head_script\" <<'__SML_RAY_HEAD_EOF__'\n"
        f"ray start --head --port={RAY_PORT} --num-gpus={NUM_GPUS_PER_NODE} --block &\n"
        f"echo 'Waiting for all Ray nodes to connect...'\n"
        f"while true; do\n"
        f'    AVAILABLE_GPUS=$(python3 -c \'import ray; ray.init(address="auto"); '
        f'print(int(ray.available_resources().get("GPU", 0)))\' 2>/dev/null || echo 0)\n'
        f'    echo "Available GPUs: $AVAILABLE_GPUS / {expected_gpus}"\n'
        f'    if [[ "$AVAILABLE_GPUS" -ge {expected_gpus} ]]; then\n'
        f"        echo 'All Ray nodes connected!'\n"
        f"        break\n"
        f"    fi\n"
        f"    sleep 5\n"
        f"done\n"
        f"{framework.entrypoint} --distributed-executor-backend ray {args}\n"
        f"__SML_RAY_HEAD_EOF__"
    )
    if use_ocf:
        # No nested double-quotes inside the --subprocess arg — the path
        # has no spaces (it's our own /tmp/sml-... naming) and the dq
        # surrounding ``--subprocess "..."`` already provides the quoting.
        launch = _ocf_wrap("bash $ray_head_script")
    else:
        launch = 'bash "$ray_head_script"'
    return f"{pre}\n\n{body_args}\n{head_script_body}\n\n{launch}\n"


def _render_vllm_follower(launch_args: LaunchArgs, framework: Framework) -> str:
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    return (
        f"{pre}\n\n"
        f"# shellcheck disable=SC2034  # unused — Ray followers are symmetric\n"
        f'node_rank="$1"\n'
        f'replica_head_ip="$2"\n'
        f"\n"
        f'ray start --address="$replica_head_ip:{RAY_PORT}" '
        f"--num-gpus={NUM_GPUS_PER_NODE} --block\n"
    )


def _render_router(launch_args: LaunchArgs) -> str:
    """Router rank script. Receives all replica head IPs as positional args,
    health-checks each one, then launches the sglang_router."""
    router_args = launch_args.router_args
    return (
        "#!/bin/bash\n"
        "set -ex\n"
        "# Positional args: replica_head_ip_0 replica_head_ip_1 ...\n"
        "\n"
        "# Bypass proxy — the Rust router does not honour it and hangs if set.\n"
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY\n"
        "\n"
        "echo 'Waiting for all replicas to fully initialize the GPU engine before starting router...'\n"
        'for ip in "$@"; do\n'
        '    echo "Checking replica at $ip..."\n'
        f'    while [[ "$(curl --noproxy "*" -s -o /dev/null '
        f"-w '%{{http_code}}' "
        f'"http://$ip:{FRAMEWORK_PORT}/health")" != "200" ]]; do\n'
        "        sleep 10\n"
        "    done\n"
        '    echo "Replica at $ip is fully ready!"\n'
        "done\n"
        "echo 'All replicas are ready! Launching router...'\n"
        "\n"
        "# Build worker-urls arg from all positional args\n"
        'worker_urls=""\n'
        'for ip in "$@"; do\n'
        f'    worker_urls="$worker_urls http://$ip:{FRAMEWORK_PORT}"\n'
        "done\n"
        "\n"
        "# shellcheck disable=SC2086  # intentional word-splitting for --worker-urls\n"
        f"python3 -m sglang_router.launch_router \\\n"
        f"    --host 0.0.0.0 \\\n"
        f"    --port {SGLANG_ROUTER_PORT} \\\n"
        f"    --worker-urls $worker_urls" + (f" \\\n    {router_args}" if router_args else "") + "\n"
    )


# ── master script renderer ─────────────────────────────────────────────────


def _render_telemetry(launch_args: LaunchArgs) -> str:
    if not launch_args.telemetry_endpoint:
        return ""
    topology = launch_args.topology
    use_router = "true" if launch_args.use_router else "false"
    use_ocf = "false" if launch_args.disable_ocf else "true"
    fa = _compose_framework_args(launch_args)
    payload = (
        "{"
        '"user": "\'"${SLURM_JOB_USER}"\'", '
        '"job_id": "\'"${SLURM_JOB_ID}"\'", '
        '"slurm_nodes": \'"${SLURM_NNODES}"\', '
        '"slurm_job_name": "\'"${SLURM_JOB_NAME}"\'", '
        '"slurm_partition": "\'"${SLURM_JOB_PARTITION}"\'", '
        f'"slurm_time": "{launch_args.time}", '
        '"slurm_account": "\'"${SLURM_JOB_ACCOUNT}"\'", '
        f'"slurm_environment": "{launch_args.environment}", '
        '"interactive": false, '
        f'"serving_framework": "{launch_args.framework}", '
        f'"framework_args": "{fa}", '
        f'"pre_launch_cmds": "{launch_args.pre_launch_cmds}", '
        f'"model_name": "{launch_args.served_model_name}", '
        f'"replicas": {topology.replicas}, '
        f'"nodes_per_replica": {topology.nodes_per_replica}, '
        f'"framework_port": {FRAMEWORK_PORT}, '
        f'"use_router": {use_router}, '
        f'"router_environment": "{launch_args.environment}", '
        f'"router_port": {SGLANG_ROUTER_PORT}, '
        f'"router_args": "{launch_args.router_args}", '
        f'"ocf_enabled": {use_ocf}, '
        f'"ocf_bootstrap_addr": "{OCF_BOOTSTRAP_ADDR}", '
        '"ocf_service_name": "llm", '
        f'"ocf_service_port": {FRAMEWORK_PORT}'
        "}"
    )
    return (
        f'curl -sf -X POST "{launch_args.telemetry_endpoint}" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f"    -d '{payload}' || true"
    )


def _render_arch_detection(launch_args: LaunchArgs) -> str:
    base = launch_args.metrics_agent_binary
    return (
        "unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE\n"
        "\n"
        "ARCH=$(uname -m)\n"
        'if [[ "$ARCH" == "aarch64" ]]; then\n'
        '    echo "Running on ARM64 (aarch64)"\n'
        "    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/\n"
        "    export OCF_BIN=/ocfbin/ocf-arm\n"
        f'    metrics_agent_bin="{base}-arm64"\n'
        'elif [[ "$ARCH" == "x86_64" ]]; then\n'
        '    echo "Running on x86_64"\n'
        "    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/\n"
        "    export OCF_BIN=/ocfbin/ocf-amd64\n"
        f'    metrics_agent_bin="{base}-amd64"\n'
        "else\n"
        '    echo "Unknown architecture: $ARCH" >&2\n'
        "    exit 1\n"
        "fi"
    )


def _render_node_mapping() -> str:
    return (
        'mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")\n'
        "TOTAL_NODES=${#nodes[@]}\n"
        "\n"
        'echo "Total nodes allocated: $TOTAL_NODES"\n'
        'for i in "${!nodes[@]}"; do\n'
        '    echo "Node $i: ${nodes[$i]}"\n'
        "done"
    )


def _render_replica_head_ip_discovery(replicas: int, nodes_per_replica: int) -> str:
    blocks = []
    for r in range(replicas):
        start_node = r * nodes_per_replica
        blocks.append(
            f"# ── replica {r} head IP ─────────────────────────────────────────────\n"
            f"replica_{r}_head_node=${{nodes[{start_node}]}}\n"
            f'replica_{r}_head_ip=$(srun --nodes=1 --ntasks=1 -w "$replica_{r}_head_node" hostname -i)\n'
            f'if [[ -z "$replica_{r}_head_ip" ]]; then\n'
            f'    echo "Error: Could not retrieve IP for replica {r} host $replica_{r}_head_node" >&2\n'
            f"    exit 1\n"
            f"fi\n"
            f'echo "Replica {r} head IP: $replica_{r}_head_ip"'
        )
    summary_urls = " ".join(f"http://$replica_{r}_head_ip:{FRAMEWORK_PORT}" for r in range(replicas))
    blocks.append(f'echo "All replica URLs: {summary_urls}"  # NOSONAR')
    return "\n\n".join(blocks)


def _render_replica_launches(launch_args: LaunchArgs) -> str:
    topology = launch_args.topology
    npr = topology.nodes_per_replica
    env = launch_args.environment

    def srun_call(node_index: int, script: str, args: str, comment: str) -> str:
        return (
            f"# {comment}\n"
            f'srun --nodes=1 --ntasks=1 --nodelist="${{nodes[{node_index}]}}" \\\n'
            f"    --container-writable \\\n"
            f'    --environment="{env}" \\\n'
            f'    bash "$RANKS_DIR/{script}" {args} &'
        )

    blocks = []
    for r in range(topology.replicas):
        blocks.append(
            srun_call(
                r * npr,
                "head.sh",
                f'"$replica_{r}_head_ip"',
                f"replica {r}, rank 0 (head)",
            )
        )
        for k in range(1, npr):
            blocks.append(
                srun_call(
                    r * npr + k,
                    "follower.sh",
                    f'{k} "$replica_{r}_head_ip"',
                    f"replica {r}, rank {k} (follower)",
                )
            )
    return "\n\n".join(blocks)


def _render_vmagent(launch_args: LaunchArgs) -> str:
    if launch_args.disable_metrics or not launch_args.metrics_remote_write_url:
        return ""
    # NOTE: this is the pre-DCGM single-vmagent shape preserved from the old
    # script.sh. Main's DCGM exporter PR (#98) added per-node vmagent + DCGM
    # only to template.jinja — porting that to Python is a follow-up.
    url = launch_args.metrics_remote_write_url
    served = launch_args.served_model_name
    fw = launch_args.framework
    return (
        "# vmagent runs on the batch node; pyxis containers share the host network\n"
        "# namespace so the framework API server is reachable at localhost:8080.\n"
        'if [[ -x "$metrics_agent_bin" ]]; then\n'
        '    "$metrics_agent_bin" \\\n'
        f"        -promscrape.config={_VMAGENT_SCRAPE_CONFIG} \\\n"
        f'        -remoteWrite.url="{url}" \\\n'
        '        -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \\\n'
        f'        -remoteWrite.label="model={served}" \\\n'
        f'        -remoteWrite.label="framework={fw}" \\\n'
        '        -remoteWrite.label="user=${SLURM_JOB_USER}" \\\n'
        '        "-remoteWrite.tmpDataPath=/tmp/vmagent-data-${SLURM_JOB_ID}" \\\n'
        '        > "/tmp/vmagent-${SLURM_JOB_ID}.log" 2>&1 &\n'
        "else\n"
        '    echo "metrics: $metrics_agent_bin not found, skipping push" >&2\n'
        "fi"
    )


def _render_router_launch(launch_args: LaunchArgs) -> str:
    topology = launch_args.topology
    if not launch_args.use_router or topology.replicas <= 1:
        return ""
    # Pass all replica head IPs to router.sh as positional args.
    ip_args = " ".join(f'"$replica_{r}_head_ip"' for r in range(topology.replicas))
    return (
        "# ── router ─────────────────────────────────────────────────────────────\n"
        'router_host_node="${nodes[0]}"\n'
        'router_host_ip="$replica_0_head_ip"\n'
        'srun --nodes=1 --ntasks=1 --nodelist="$router_host_node" \\\n'
        "    --container-writable \\\n"
        f'    --environment="{launch_args.environment}" \\\n'
        "    --overlap \\\n"
        f'    bash "$RANKS_DIR/router.sh" {ip_args} &\n'
        "\n"
        "echo\n"
        f'echo "Router URL: http://$router_host_ip:{SGLANG_ROUTER_PORT}"  # NOSONAR'
    )


def _render_footer() -> str:
    return (
        "echo\n"
        'echo "To connect to the host node:"\n'
        'echo "srun --jobid $SLURM_JOB_ID -w ${nodes[0]} --overlap --pty bash"\n'
        "\n"
        "echo\n"
        'echo "Make sure to cancel the job at the end:"\n'
        'echo "scancel $SLURM_JOB_ID"\n'
        "\n"
        "wait\n"
        'echo "Script finished at $(date)"'
    )


# ── public API ─────────────────────────────────────────────────────────────


MASTER_FILENAME = "master.sh"


def _render_self_extracting_ranks(rank_scripts: dict[str, str]) -> str:
    """Bash that materialises rank scripts to a per-job dir on shared FS.

    Used by launchers (firecrest) that submit a single script_str rather
    than a directory of files. Each script is laid down via a single-quoted
    heredoc so its body lands on disk verbatim.

    The location must be on a filesystem visible to every compute node — the
    master runs `cat` on the batch node but later srun-dispatches to compute
    nodes which need to see the same files. ``$HOME/.sml/...`` is shared
    across batch and compute nodes on CSCS systems; ``/tmp`` is per-node and
    would break.
    """
    blocks = [
        "# Self-extract rank scripts: this master.sh was submitted standalone\n"
        "# (no sibling files), so we materialise the rank scripts under HOME\n"
        "# (shared FS, visible to all compute nodes) at job start time. The\n"
        "# single-quoted heredoc keeps each body literal.",
        'RANKS_DIR="$HOME/.sml/job-${SLURM_JOB_ID}"',
        'mkdir -p "$RANKS_DIR"',
    ]
    for filename, content in rank_scripts.items():
        delim = f"__SML_{filename.replace('.sh', '').upper()}_EOF__"
        blocks.append(f"cat > \"$RANKS_DIR/{filename}\" <<'{delim}'\n{content.rstrip()}\n{delim}")
        blocks.append(f'chmod +x "$RANKS_DIR/{filename}"')
    return "\n\n".join(blocks)


def render_master(launch_args: LaunchArgs, *, embed_rank_scripts: bool = False) -> str:
    """Render ``master.sh`` content (without ``#SBATCH`` header — launchers
    attach those via CLI args or :func:`render_sbatch_header`).

    When ``embed_rank_scripts=True`` the rank scripts are inlined as
    self-extracting cat-heredocs at the top, materialised to ``/tmp`` at
    job start. Use this for launchers (firecrest) that submit a single
    script string; for slurm we write files to disk and reference them.
    """
    sections: list[str] = [
        "# shellcheck shell=bash",
        "set -euo pipefail",
    ]
    if embed_rank_scripts:
        sections.append(_render_self_extracting_ranks(render_rank_scripts(launch_args)))
    else:
        sections.append(
            "# Rank scripts live next to this file. master.sh is dispatched\n"
            "# by SLURM and the rank scripts are siblings in the same dir.\n"
            'RANKS_DIR="$(dirname "$(readlink -f "$0")")"'
        )

    telemetry = _render_telemetry(launch_args)
    if telemetry:
        sections.append(telemetry)

    sections.append(_render_arch_detection(launch_args))
    sections.append(_render_node_mapping())

    topology = launch_args.topology
    sections.append(_render_replica_head_ip_discovery(topology.replicas, topology.nodes_per_replica))

    sections.append(_render_replica_launches(launch_args))

    vmagent = _render_vmagent(launch_args)
    if vmagent:
        sections.append(vmagent)

    router_launch = _render_router_launch(launch_args)
    if router_launch:
        sections.append(router_launch)

    sections.append(_render_footer())
    return "\n\n".join(sections) + "\n"


def render_rank_scripts(launch_args: LaunchArgs) -> dict[str, str]:
    """Render all rank scripts needed for ``launch_args``.

    Returns a dict mapping filename (e.g. ``"head.sh"``) to bash content.
    Each script is independently shellcheckable.
    """
    framework = _make_framework(launch_args.framework)
    npr = launch_args.topology.nodes_per_replica

    scripts: dict[str, str] = {}

    if framework.name == "sglang":
        scripts["head.sh"] = _render_sglang_head(launch_args, framework)
        if npr > 1:
            scripts["follower.sh"] = _render_sglang_follower(launch_args, framework)
    elif framework.name == "vllm":
        scripts["head.sh"] = _render_vllm_head(launch_args, framework)
        if npr > 1:
            scripts["follower.sh"] = _render_vllm_follower(launch_args, framework)

    if launch_args.use_router and launch_args.topology.replicas > 1:
        scripts["router.sh"] = _render_router(launch_args)

    return scripts


def render_all(launch_args: LaunchArgs) -> dict[str, str]:
    """Convenience: returns ``{master.sh: ..., head.sh: ..., ...}``."""
    out = {MASTER_FILENAME: render_master(launch_args)}
    out.update(render_rank_scripts(launch_args))
    return out
