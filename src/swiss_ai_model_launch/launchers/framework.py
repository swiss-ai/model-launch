from __future__ import annotations

import importlib.metadata
import shlex
from typing import ClassVar

from swiss_ai_model_launch.launchers.launch_args import (
    FRAMEWORK_PORT,
    LaunchArgs,
    time_str_to_seconds,
)

SGLANG_ROUTER_PORT = 30000
# Default (prod) OCF bootstrap address. The dev datacenter peer differs only
# in the IP. Override per-launch via LaunchArgs.ocf_bootstrap_addr (CLI:
# `--otela-bootstrap-addr <multiaddr>` or shorthand `--dev`).
OCF_BOOTSTRAP_ADDR = "/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"
OCF_BOOTSTRAP_ADDR_DEV = "/ip4/148.187.108.177/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"
RAY_PORT = 6379
NUM_GPUS_PER_NODE = 4
SGLANG_DIST_INIT_PORT = 5757

_METRICS_CONFIG_DIR = "/capstor/store/cscs/swissai/infra01/ocf-share"
_VMAGENT_SCRAPE_CONFIG = f"{_METRICS_CONFIG_DIR}/vmagent-scrape.yaml"
_VMAGENT_SCRAPE_CONFIG_NO_DCGM = f"{_METRICS_CONFIG_DIR}/vmagent-scrape-no-dcgm.yaml"
_VMAGENT_SCRAPE_CONFIG_DCGM_ONLY = f"{_METRICS_CONFIG_DIR}/vmagent-scrape-dcgm-only.yaml"
_DCGM_EXPORTER_PORT = 9400
_DCGM_COUNTERS = f"{_METRICS_CONFIG_DIR}/default-counters.csv"


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


def _ocf_labels(launch_args: LaunchArgs) -> str:
    # Users often write framework_args with bash line-continuations + indented
    # follow-on lines, which collapse to runs of whitespace inside the quoted
    # string. Normalise here so the on-mesh label is the canonical single-space
    # form ("--a 1 --b 2") rather than the as-typed "--a 1     --b 2".
    framework_args_normalised = " ".join(_compose_framework_args(launch_args).split())
    user_input = [
        f"framework={launch_args.framework}",
        f"served_model_name={launch_args.served_model_name}",
        f"framework_args={framework_args_normalised}",
    ]
    quoted = " \\\n".join(f"    --label {shlex.quote(kv)}" for kv in user_input)
    seconds = time_str_to_seconds(launch_args.time)
    return (
        "    --label launched_by=$USER \\\n"
        "    --label slurm_job_id=$SLURM_JOB_ID \\\n"
        "    --label slurm_partition=${SLURM_JOB_PARTITION:-unknown} \\\n"
        "    --label worker_group_id=$SLURM_JOB_ID \\\n"
        f"{quoted} \\\n"
        "    --label started_at=$(date -u +%FT%TZ) \\\n"
        f'    --label expires_at=$(date -u -d "+{seconds} seconds" +%FT%TZ) \\\n'
    )


def _resolve_ocf_bootstrap_addr(launch_args: LaunchArgs) -> str:
    return launch_args.ocf_bootstrap_addr or OCF_BOOTSTRAP_ADDR


def _ocf_wrap(inner_cmd: str, launch_args: LaunchArgs) -> str:
    bootstrap_addr = _resolve_ocf_bootstrap_addr(launch_args)
    return (
        f"$OCF_BIN start \\\n"
        f'    --bootstrap.addr "{bootstrap_addr}" \\\n'
        f"    --service.name llm \\\n"
        f"    --service.port {FRAMEWORK_PORT} \\\n"
        f"{_ocf_labels(launch_args)}"
        f'    --subprocess "{inner_cmd}"'
    )


def _ocf_wrap_metrics_only(inner_cmd: str, launch_args: LaunchArgs) -> str:
    bootstrap_addr = _resolve_ocf_bootstrap_addr(launch_args)
    return (
        f"$OCF_BIN start \\\n"
        f'    --bootstrap.addr "{bootstrap_addr}" \\\n'
        f"{_ocf_labels(launch_args)}"
        f'    --subprocess "{inner_cmd}"'
    )


def _shebang_and_setup(framework: Framework, pre_launch_cmds: str) -> str:
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
        launch = _ocf_wrap(cmd, launch_args)
    else:
        launch = cmd
    return f"{pre}\n\n{body_args}\n{launch}\n"


def _render_sglang_follower(launch_args: LaunchArgs, framework: Framework) -> str:
    args = _compose_framework_args(launch_args)
    npr = launch_args.topology.nodes_per_replica
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    use_ocf = not launch_args.disable_ocf
    # node_rank is $1 (small int) and replica_head_ip is $2 (IPv4 from master).
    # Both are word-split-safe and intentionally left unquoted here so the same
    # cmd string works both directly (disable_ocf path) and inside the OCF
    # --subprocess "..." wrap without nested-quote shellcheck warnings.
    cmd = (
        f"{framework.entrypoint} \\\n"
        f"    --dist-init-addr $replica_head_ip:{SGLANG_DIST_INIT_PORT} \\\n"
        f"    --nnodes {npr} \\\n"
        f"    --node-rank $node_rank \\\n"
        f"    {args}"
    )
    if use_ocf:
        # Followers join DNT in metrics-only mode so the full multi-node
        # topology of a replica is visible (grouped by worker_group_id).
        launch = _ocf_wrap_metrics_only(cmd, launch_args)
    else:
        launch = cmd
    return f'{pre}\n\nnode_rank="$1"\nreplica_head_ip="$2"\n\n{launch}\n'


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
            launch = _ocf_wrap(cmd, launch_args)
        else:
            launch = cmd
        return f"{pre}\n\n{body_args}\n{launch}\n"

    # Multi-node head: stage the Ray bootstrap + API server invocation as a
    # script on /tmp (single-quoted heredoc keeps $-constructs literal in
    # the file), then either run it directly or via OCF's --subprocess.
    # On-disk staging dodges OCF's subprocess re-evaluation.
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
        launch = _ocf_wrap("bash $ray_head_script", launch_args)
    else:
        launch = 'bash "$ray_head_script"'
    return f"{pre}\n\n{body_args}\n{head_script_body}\n\n{launch}\n"


def _render_vllm_follower(launch_args: LaunchArgs, framework: Framework) -> str:
    pre = _shebang_and_setup(framework, launch_args.pre_launch_cmds)
    use_ocf = not launch_args.disable_ocf
    # replica_head_ip is $2 (IPv4 from master), word-split-safe, left unquoted
    # so the cmd is reusable inside the OCF --subprocess "..." wrap without
    # nested-quote shellcheck warnings.
    cmd = f"ray start --address=$replica_head_ip:{RAY_PORT} --num-gpus={NUM_GPUS_PER_NODE} --block"
    if use_ocf:
        # Followers join DNT in metrics-only mode so the full multi-node
        # topology of a replica is visible (grouped by worker_group_id).
        launch = _ocf_wrap_metrics_only(cmd, launch_args)
    else:
        launch = cmd
    return (
        f"{pre}\n\n"
        f"# shellcheck disable=SC2034  # unused — Ray followers are symmetric\n"
        f'node_rank="$1"\n'
        f'replica_head_ip="$2"\n'
        f"\n"
        f"{launch}\n"
    )


def _render_router(launch_args: LaunchArgs) -> str:
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


def _render_telemetry(launch_args: LaunchArgs) -> str:
    if not launch_args.telemetry_endpoint:
        return ""
    topology = launch_args.topology
    use_router = "true" if launch_args.use_router else "false"
    use_ocf = "false" if launch_args.disable_ocf else "true"
    fa = _compose_framework_args(launch_args)
    sml_version = importlib.metadata.version("swiss-ai-model-launch")
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
        f'"ocf_bootstrap_addr": "{_resolve_ocf_bootstrap_addr(launch_args)}", '
        '"ocf_service_name": "llm", '
        f'"ocf_service_port": {FRAMEWORK_PORT}, '
        f'"model_launch_version": "{sml_version}"'
        "}"
    )
    return (
        f'curl -sf -X POST "{launch_args.telemetry_endpoint}" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f"    -d '{payload}' || true"
    )


def _dcgm_enabled(launch_args: LaunchArgs) -> bool:
    return not launch_args.disable_metrics and not launch_args.disable_dcgm_exporter


def _render_arch_detection(launch_args: LaunchArgs) -> str:
    base = launch_args.metrics_agent_binary
    dcgm_base = launch_args.dcgm_exporter_binary
    # Only emit metrics_agent_bin / dcgm_exporter_bin assignments when something
    # downstream consumes them — otherwise shellcheck flags SC2034 (unused var).
    needs_metrics_bin = not launch_args.disable_metrics
    needs_dcgm_bin = _dcgm_enabled(launch_args)
    # /ocfbin/{prod,dev}/otela-<arch> are stable symlinks maintained by
    # OpenTela's release / deploy-dev workflows; they point at versioned
    # files in the same directory. --dev (LaunchArgs.dev) flips the channel.
    ocf_bin_channel = "dev" if launch_args.dev else "prod"
    arm_lines = [
        '    echo "Running on ARM64 (aarch64)"',
        "    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/",
        f"    export OCF_BIN=/ocfbin/{ocf_bin_channel}/otela-arm64",
    ]
    x86_lines = [
        '    echo "Running on x86_64"',
        "    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/",
        f"    export OCF_BIN=/ocfbin/{ocf_bin_channel}/otela-amd64",
    ]
    if needs_metrics_bin:
        arm_lines.append(f'    metrics_agent_bin="{base}-arm64"')
        x86_lines.append(f'    metrics_agent_bin="{base}-amd64"')
    if needs_dcgm_bin:
        arm_lines.append(f'    dcgm_exporter_bin="{dcgm_base}-arm64"')
        x86_lines.append(f'    dcgm_exporter_bin="{dcgm_base}-amd64"')
    arm_block = "\n".join(arm_lines)
    x86_block = "\n".join(x86_lines)
    return (
        "unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE\n"
        "\n"
        "ARCH=$(uname -m)\n"
        f'if [[ "$ARCH" == "aarch64" ]]; then\n{arm_block}\n'
        f'elif [[ "$ARCH" == "x86_64" ]]; then\n{x86_block}\n'
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
            # Bind RANKS_DIR into the container so the rank script (on the
            # host's shared FS) is visible to the bash invocation inside the
            # pyxis container. Attached per-srun rather than via the env toml's
            # static mount list, which is being narrowed and read-only-ed.
            f'    --container-mounts="$RANKS_DIR:$RANKS_DIR" \\\n'
            f'    --environment="{env}" \\\n'
            f'    bash "$RANKS_DIR/{script}" {args} &\n'
            # Track this srun's PID so the footer's `wait -n` exits as soon
            # as the first critical bg job dies (and the trap kills the rest).
            f"critical_pids+=($!)"
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
    if launch_args.disable_metrics:
        return ""
    url = launch_args.metrics_remote_write_url
    served = launch_args.served_model_name
    fw = launch_args.framework
    dcgm_on = _dcgm_enabled(launch_args)
    batch_scrape_config = _VMAGENT_SCRAPE_CONFIG if dcgm_on else _VMAGENT_SCRAPE_CONFIG_NO_DCGM

    # Common vmagent remoteWrite labels — shared between the batch node (rank 0)
    # and per-worker invocations via `srun --overlap`.
    common_labels = (
        '        -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \\\n'
        f'        -remoteWrite.label="model={served}" \\\n'
        f'        -remoteWrite.label="framework={fw}" \\\n'
        '        -remoteWrite.label="user=${SLURM_JOB_USER}" \\\n'
    )

    batch_block = (
        "# vmagent runs on the batch node; pyxis containers share the host network\n"
        "# namespace so the framework API server is reachable at localhost:8080.\n"
        "# vmagent is non-critical: disowned so it's not in `wait -n`'s scope, and\n"
        "# the EXIT trap in the footer kills it when master.sh terminates so the\n"
        "# allocation can be released as soon as the framework process is gone.\n"
        'if [[ -x "$metrics_agent_bin" ]]; then\n'
    )
    if dcgm_on:
        batch_block += (
            '    if [[ -e /dev/nvidia0 && -x "$dcgm_exporter_bin" ]]; then\n'
            '        "$dcgm_exporter_bin" \\\n'
            f"            --address 0.0.0.0:{_DCGM_EXPORTER_PORT} \\\n"
            f"            -f {_DCGM_COUNTERS} \\\n"
            '            > "/tmp/dcgm-exporter-${SLURM_JOB_ID}.log" 2>&1 &\n'
            "        disown $!\n"
            "    else\n"
            '        echo "dcgm-exporter: no NVIDIA GPU or binary not found, skipping" >&2\n'
            "    fi\n"
        )
    batch_block += (
        '    "$metrics_agent_bin" \\\n'
        f"        -promscrape.config={batch_scrape_config} \\\n"
        f'        -remoteWrite.url="{url}" \\\n'
        f"{common_labels}"
        '        -remoteWrite.label="node=$(hostname)" \\\n'
        '        "-remoteWrite.tmpDataPath=/tmp/vmagent-data-${SLURM_JOB_ID}" \\\n'
        '        > "/tmp/vmagent-${SLURM_JOB_ID}.log" 2>&1 &\n'
        "    vmagent_pid=$!\n"
        '    disown "$vmagent_pid"\n'
        "else\n"
        '    echo "metrics: $metrics_agent_bin not found, skipping push" >&2\n'
        "fi"
    )

    if not dcgm_on or launch_args.total_nodes <= 1:
        return batch_block

    # Per-worker dcgm + vmagent. The batch node (index 0) already runs both
    # directly; remaining nodes need an `srun --overlap` so the exporter
    # publishes GPU telemetry from each compute node and vmagent ships it.
    # ${dcgm_exporter_bin} / ${metrics_agent_bin} are master-shell vars (set by
    # arch detection) so they're expanded here at submission time; SLURM_*
    # and $(hostname) are deferred to the worker via \$ / \"..\$..\".
    worker_block = (
        'for i in "${!nodes[@]}"; do\n'
        '    if [[ "$i" -eq 0 ]]; then continue; fi\n'
        '    node="${nodes[$i]}"\n'
        '    srun --nodes=1 --ntasks=1 --nodelist="$node" --overlap \\\n'
        f'        bash -c "\n'
        f'            if [[ -e /dev/nvidia0 && -x \\"${{dcgm_exporter_bin}}\\" ]]; then\n'
        f'                \\"${{dcgm_exporter_bin}}\\" \\\n'
        f"                    --address 0.0.0.0:{_DCGM_EXPORTER_PORT} \\\n"
        f"                    -f {_DCGM_COUNTERS} \\\n"
        f"                    > /tmp/dcgm-exporter-\\${{SLURM_JOB_ID}}.log 2>&1 &\n"
        f'                \\"${{metrics_agent_bin}}\\" \\\n'
        f"                    -promscrape.config={_VMAGENT_SCRAPE_CONFIG_DCGM_ONLY} \\\n"
        f'                    -remoteWrite.url=\\"{url}\\" \\\n'
        f'                    -remoteWrite.label=\\"slurm_job_id=\\${{SLURM_JOB_ID}}\\" \\\n'
        f'                    -remoteWrite.label=\\"model={served}\\" \\\n'
        f'                    -remoteWrite.label=\\"framework={fw}\\" \\\n'
        f'                    -remoteWrite.label=\\"user=\\${{SLURM_JOB_USER}}\\" \\\n'
        f'                    -remoteWrite.label=\\"node=\\$(hostname)\\" \\\n'
        f"                    -remoteWrite.tmpDataPath=/tmp/vmagent-data-\\${{SLURM_JOB_ID}} \\\n"
        f"                    > /tmp/vmagent-\\${{SLURM_JOB_ID}}.log 2>&1 &\n"
        f"                wait\n"
        f"            else\n"
        f'                echo \\"dcgm-exporter: no NVIDIA GPU or binary not found on \\$(hostname), skipping\\" >&2\n'
        f"            fi\n"
        f'        " &\n'
        f"    disown $!\n"
        f"done"
    )
    return f"{batch_block}\n\n{worker_block}"


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
        '    --container-mounts="$RANKS_DIR:$RANKS_DIR" \\\n'
        f'    --environment="{launch_args.environment}" \\\n'
        "    --overlap \\\n"
        f'    bash "$RANKS_DIR/router.sh" {ip_args} &\n'
        "critical_pids+=($!)\n"
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
        # Tear down as soon as the first critical bg job (head / follower /
        # router srun) exits. A healthy launch keeps those running until the
        # SLURM time limit; any exit means the inference server is gone, so
        # vmagent has nothing to scrape and SLURM should release the nodes.
        "cleanup() {\n"
        '    if [[ -n "$vmagent_pid" ]]; then\n'
        '        kill "$vmagent_pid" 2>/dev/null || true\n'
        "    fi\n"
        "    if (( ${#critical_pids[@]} > 0 )); then\n"
        '        kill "${critical_pids[@]}" 2>/dev/null || true\n'
        "    fi\n"
        "}\n"
        "trap cleanup EXIT\n"
        "trap 'exit 143' TERM\n"
        "trap 'exit 130' INT\n"
        "\n"
        "rc=0\n"
        "wait -n || rc=$?\n"
        'echo "Master finished at $(date) with code $rc"\n'
        'exit "$rc"'
    )


MASTER_FILENAME = "master.sh"


def _render_self_extracting_ranks(rank_scripts: dict[str, str]) -> str:
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


def render_master(launch_args: LaunchArgs) -> str:
    sections: list[str] = [
        "# shellcheck shell=bash",
        "set -euo pipefail",
        # Lifecycle tracking. critical_pids collects the head / follower /
        # router srun PIDs; the footer's `wait -n` exits as soon as the first
        # one dies. vmagent_pid (if metrics are enabled) is held separately
        # so it stays out of `wait -n`'s scope but is still killed by the
        # EXIT trap. Initialised here so `set -u` is happy even when no
        # vmagent is rendered, or if cleanup runs before launches start.
        'critical_pids=()\nvmagent_pid=""',
        _render_self_extracting_ranks(render_rank_scripts(launch_args)),
    ]

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
    out = {MASTER_FILENAME: render_master(launch_args)}
    out.update(render_rank_scripts(launch_args))
    return out
