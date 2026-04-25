# shellcheck shell=bash

ROUTER_PORT=30000
OCF_SERVICE_NAME="llm"
OCF_SERVICE_PORT=8080
OCF_BOOTSTRAP_ADDR="/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9"

if [ -n "$TELEMETRY_ENDPOINT" ]; then
    curl -sf -X POST "$TELEMETRY_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d '{"user": "'"${USER}"'", "job_id": "'"${SLURM_JOB_ID}"'", "slurm_nodes": '"${SLURM_NNODES}"', "slurm_job_name": "'"${SLURM_JOB_NAME}"'", "slurm_partition": "'"${SLURM_JOB_PARTITION}"'", "slurm_time": "'"${SML_TIME}"'", "slurm_account": "'"${SLURM_JOB_ACCOUNT}"'", "slurm_environment": "'"${SML_ENVIRONMENT}"'", "interactive": false, "serving_framework": "'"${FRAMEWORK}"'", "framework_args": "'"${FRAMEWORK_ARGS}"'", "pre_launch_cmds": "'"${PRE_LAUNCH_CMDS}"'", "model_name": "'"${SERVED_MODEL_NAME}"'", "workers": '"${WORKERS}"', "nodes_per_worker": '"${NODES_PER_WORKER}"', "worker_port": '"${WORKER_PORT}"', "use_router": '"${USE_ROUTER}"', "router_environment": "'"${ROUTER_ENVIRONMENT}"'", "router_port": 30000, "router_args": "'"${ROUTER_ARGS}"'", "ocf_enabled": '"${USE_OCF}"', "ocf_bootstrap_addr": "'"${OCF_BOOTSTRAP_ADDR}"'", "ocf_service_name": "llm", "ocf_service_port": 8080}' || true
fi

unset SLURM_CPU_BIND SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_LIST SLURM_CPU_BIND_VERBOSE

ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Running on ARM64 (aarch64)"
    export SP_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
    export OCF_BIN=/ocfbin/ocf-arm
    METRICS_AGENT_BIN="${METRICS_AGENT_BIN}-arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "Running on x86_64"
    export SP_NCCL_SO_PATH=/usr/lib/x86_64-linux-gnu/
    export OCF_BIN=/ocfbin/ocf-amd64
    METRICS_AGENT_BIN="${METRICS_AGENT_BIN}-amd64"
else
    echo "Unknown architecture: $ARCH"
    exit 1
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")
TOTAL_NODES=${#nodes[@]}

echo "Total nodes allocated: $TOTAL_NODES"
for i in "${!nodes[@]}"; do
    echo "Node $i: ${nodes[$i]}"
done

case "$FRAMEWORK" in
    sglang)
        FRAMEWORK_ENV_SETUP="export no_proxy=\"0.0.0.0,\$no_proxy\"; export NO_PROXY=\"0.0.0.0,\$NO_PROXY\"; export SGL_ENABLE_JIT_DEEPGEMM=\"false\""
        FRAMEWORK_LAUNCH="python3 -m sglang.launch_server"
        ;;
    vllm)
        FRAMEWORK_ENV_SETUP="export RAY_CGRAPH_get_timeout=1800; export no_proxy=\"0.0.0.0,\$no_proxy\"; export NO_PROXY=\"0.0.0.0,\$NO_PROXY\""
        FRAMEWORK_LAUNCH="python3 -m vllm.entrypoints.openai.api_server"
        ;;
esac

ROUTER_LAUNCH="python3 -m sglang_router.launch_router"

EXPECTED_NODES=$((WORKERS * NODES_PER_WORKER))
if [ "$TOTAL_NODES" -ne "$EXPECTED_NODES" ]; then
    echo "Warning: Total nodes ($TOTAL_NODES) doesn't match WORKERS($WORKERS) * NODES_PER_WORKER($NODES_PER_WORKER) = $EXPECTED_NODES"
    echo "Adjusting to use all available nodes with WORKERS workers"
    NODES_PER_WORKER=$((TOTAL_NODES / WORKERS))
fi

worker_head_ips=()
worker_urls=()

for worker_id in $(seq 0 $((WORKERS - 1))); do
    start_node=$((worker_id * NODES_PER_WORKER))
    worker_host_node=${nodes[$start_node]}
    worker_host_ip=$(srun --nodes=1 --ntasks=1 -w "${worker_host_node}" hostname -i)

    if [ -z "$worker_host_ip" ]; then
        echo "Error: Could not retrieve IP address for worker $worker_id host ${worker_host_node}"
        exit 1
    fi

    echo "Worker $worker_id host IP: $worker_host_ip"
    worker_head_ips+=("$worker_host_ip")
    worker_urls+=("http://${worker_host_ip}:${WORKER_PORT}")
done

echo "All worker URLs: ${worker_urls[*]}"

for worker_id in $(seq 0 $((WORKERS - 1))); do
    echo "Launching worker $worker_id"
    start_node=$((worker_id * NODES_PER_WORKER))
    worker_host_ip=${worker_head_ips[$worker_id]}

    for local_rank in $(seq 0 $((NODES_PER_WORKER - 1))); do
        global_node_idx=$((start_node + local_rank))
        node=${nodes[$global_node_idx]}

        case "$FRAMEWORK" in
            sglang)
                if [ "$NODES_PER_WORKER" -gt 1 ]; then
                    FRAMEWORK_DIST_ARGS_EXPANDED="--dist-init-addr ${worker_host_ip}:5757 --nnodes ${NODES_PER_WORKER} --node-rank ${local_rank}"
                else
                    FRAMEWORK_DIST_ARGS_EXPANDED=""
                fi
                ;;
            vllm)
                if [ "$NODES_PER_WORKER" -gt 1 ]; then
                    FRAMEWORK_DIST_ARGS_EXPANDED="--nnodes ${NODES_PER_WORKER} --node-rank ${local_rank} --distributed-executor-backend mp --master-addr ${worker_host_ip} --master-port 5757"
                else
                    FRAMEWORK_DIST_ARGS_EXPANDED=""
                fi
                ;;
        esac

        # For vLLM multi-node: only the head node runs the API server via Ray;
        # follower nodes join the Ray cluster and block.
        if [ "$FRAMEWORK" = "vllm" ] && [ "$NODES_PER_WORKER" -gt 1 ]; then
            RAY_PORT=6379
            NUM_GPUS=4

            if [ "$local_rank" -eq 0 ]; then
                # Head node: start Ray head, wait for all workers, then launch vLLM
                FRAMEWORK_CMD_OVERRIDE="ray start --head --port=${RAY_PORT} --num-gpus=${NUM_GPUS} --block &

echo 'Waiting for all Ray worker nodes to connect...'
EXPECTED_GPUS=\$((${NODES_PER_WORKER} * ${NUM_GPUS}))
while true; do
    AVAILABLE_GPUS=\$(python3 -c 'import ray; ray.init(address=\"auto\"); print(int(ray.available_resources().get(\"GPU\", 0)))' 2>/dev/null || echo 0)
    echo \"Available GPUs: \${AVAILABLE_GPUS} / \${EXPECTED_GPUS}\"
    if [ \"\${AVAILABLE_GPUS}\" -ge \"\${EXPECTED_GPUS}\" ]; then
        echo 'All Ray workers connected!'
        break
    fi
    sleep 5
done

$FRAMEWORK_LAUNCH --distributed-executor-backend ray $FRAMEWORK_ARGS"
            else
                # Follower node: join Ray cluster and block
                FRAMEWORK_CMD_OVERRIDE="ray start --address=${worker_host_ip}:${RAY_PORT} --num-gpus=${NUM_GPUS} --block"
            fi
        fi

        if [ -n "${FRAMEWORK_CMD_OVERRIDE:-}" ]; then
            FRAMEWORK_CMD="$FRAMEWORK_CMD_OVERRIDE"
        else
            FRAMEWORK_CMD="$FRAMEWORK_LAUNCH $FRAMEWORK_DIST_ARGS_EXPANDED $FRAMEWORK_ARGS"
        fi

        if [ "$USE_OCF" = "true" ] && [ "$local_rank" -eq 0 ]; then
            FRAMEWORK_CMD="\$OCF_BIN start --bootstrap.addr \"$OCF_BOOTSTRAP_ADDR\" --service.name $OCF_SERVICE_NAME --service.port $OCF_SERVICE_PORT --subprocess \"$FRAMEWORK_CMD\""
        fi

        srun --nodes=1 --ntasks=1 --nodelist="$node" \
            --container-writable \
            --environment="$SML_ENVIRONMENT" \
            bash --norc --noprofile -c "\
set -ex
$FRAMEWORK_ENV_SETUP
if [ -n \"$PRE_LAUNCH_CMDS\" ]; then
    echo \"Running pre-launch commands...\"
    eval \"$PRE_LAUNCH_CMDS\"
fi
$FRAMEWORK_CMD" &

        FRAMEWORK_CMD_OVERRIDE=""
    done
done

# vmagent runs on the batch node rather than inside a container: pyxis containers
# share the host network namespace, so the framework's API server is reachable
# at localhost:8080 from here without any extra networking.
if [ -n "$METRICS_REMOTE_WRITE_URL" ] && [ -x "$METRICS_AGENT_BIN" ]; then
    "$METRICS_AGENT_BIN" \
        -promscrape.config=/capstor/store/cscs/swissai/infra01/ocf-share/vmagent-scrape.yaml \
        -remoteWrite.url="${METRICS_REMOTE_WRITE_URL}" \
        -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \
        -remoteWrite.label="model=${SERVED_MODEL_NAME}" \
        -remoteWrite.label="framework=${FRAMEWORK}" \
        -remoteWrite.label="user=${USER}" \
        "-remoteWrite.tmpDataPath=/tmp/vmagent-data-${SLURM_JOB_ID}" \
        > "/tmp/vmagent-${SLURM_JOB_ID}.log" 2>&1 &
elif [ -n "$METRICS_REMOTE_WRITE_URL" ]; then
    echo "metrics: $METRICS_AGENT_BIN not found, skipping push" >&2
fi

if [ "$USE_ROUTER" = "true" ] && [ "$WORKERS" -gt 1 ]; then
    router_host_node=${nodes[0]}
    router_host_ip=${worker_head_ips[0]}
    worker_urls_str="${worker_urls[*]}"

    echo "Starting router on ${router_host_node} (${router_host_ip}:${ROUTER_PORT})"
    echo "Router worker URLs: ${worker_urls_str}"

    srun --nodes=1 --ntasks=1 --nodelist="$router_host_node" \
        --container-writable \
        --environment="$ROUTER_ENVIRONMENT" \
        --overlap \
        bash --norc --noprofile -c "\
set -ex
# bypass proxy — the Rust router does not honour it and hangs if set
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

echo \"Waiting for all workers to fully initialize the GPU engine before starting router...\"
for worker_ip in ${worker_head_ips[*]}; do
    echo \"Checking worker at \$worker_ip...\"
    while [ \"\$(curl --noproxy \"*\" -s -o /dev/null -w '%{http_code}' http://\${worker_ip}:${WORKER_PORT}/health)\" != \"200\" ]; do
        sleep 10
    done
    echo \"Worker \$worker_ip is fully ready!\"
done
echo \"All workers are ready! Launching router...\"

$ROUTER_LAUNCH --host 0.0.0.0 --port ${ROUTER_PORT} --worker-urls ${worker_urls_str} $ROUTER_ARGS" &

    echo ""
    echo "Router URL: http://${router_host_ip}:${ROUTER_PORT}"
fi

echo ""
echo "To connect to the host node:"
echo "srun --jobid $SLURM_JOB_ID -w ${nodes[0]} --overlap --pty bash"

echo ""
echo "Make sure to cancel the job at the end:"
echo "scancel $SLURM_JOB_ID"

wait
echo "Script finished at $(date)"
