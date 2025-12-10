import os
import sys
import logging
import argparse
import tempfile

from utils import nanoid, extract_model_name, setup_logging, generate_job_script, submit_job


def parse_args():
    parser = argparse.ArgumentParser()
    # SLURM-specific parameters (prefixed to avoid collisions)
    parser.add_argument("--slurm-job-name", type=str, default=None)
    parser.add_argument("--slurm-nodes", type=int, required=True, help="Total number of nodes to allocate")
    parser.add_argument("--slurm-partition", type=str, default="normal")
    parser.add_argument("--slurm-time", type=str, default="04:00:00", help="Job time limit (default: 04:00:00)")
    parser.add_argument("--slurm-account", type=str, default="infra01", help="SLURM account (default: infra01)")
    parser.add_argument("--slurm-environment", type=str, help="SLURM environment name (default: {framework})")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell instead of batch job")

    # Framework selection
    parser.add_argument("--serving-framework", type=str, choices=["sglang", "vllm"], required=True, help="Serving framework to use")

    # Framework arguments as a single string
    parser.add_argument("--framework-args", type=str, default="", help="Arguments to pass to the serving framework")

    # Pre-launch setup
    parser.add_argument("--pre-launch-cmds", type=str, default="", help="Commands to run before launching framework (e.g., 'pip install blobfile; pip install package2')")

    # Optional orchestration parameters
    parser.add_argument("--workers", type=int, default=1, help="Number of independent workers")
    parser.add_argument("--nodes-per-worker", type=int, help="Nodes per worker (default: all nodes / workers)")
    parser.add_argument("--worker-port", type=int, default=5000, help="Port for workers")

    # Router parameters
    parser.add_argument("--use-router", action="store_true", help="Enable router (only if workers > 1)")
    parser.add_argument("--router-environment", type=str, help="SLURM environment for router (default: same as worker)")
    parser.add_argument("--router-port", type=int, default=30000, help="Router port")
    parser.add_argument("--router-args", type=str, default="", help="Arguments to pass to the router")

    # OCF (Open Compute Framework) parameters
    parser.add_argument("--disable-ocf", action="store_true", help="Disable OCF wrapper (OCF is enabled by default)")
    parser.add_argument("--ocf-bootstrap-addr", type=str, default="/ip4/148.187.108.172/tcp/43905/p2p/QmQsNxJVa2rnidp998qAz4FCutgmjBsuZqtrxUUy5YfgBu", help="OCF bootstrap address")
    parser.add_argument("--ocf-service-name", type=str, default="llm", help="OCF service name")
    parser.add_argument("--ocf-service-port", type=int, default=8080, help="OCF service port")

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Generate job name if not provided
    if not args.slurm_job_name:
        model_name = extract_model_name(args.framework_args)
        args.slurm_job_name = nanoid(model_name=model_name)

    # Log the full command
    logging.info("=" * 80)
    logging.info("Full command:")
    logging.info(" ".join(sys.argv))
    logging.info("=" * 80)
    logging.info("")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.jinja")

    # Determine environment name - default to framework name
    environment = args.slurm_environment if args.slurm_environment else args.serving_framework
    router_environment = args.router_environment if args.router_environment else environment

    # Calculate nodes_per_worker if not specified
    nodes_per_worker = args.nodes_per_worker if args.nodes_per_worker else args.slurm_nodes // args.workers

    # Build template args
    template_args = {
        "job_name": args.slurm_job_name,
        "account": args.slurm_account,
        "nodes": args.slurm_nodes,
        "partition": args.slurm_partition,
        "time": args.slurm_time,
        "environment": environment,
        "framework": args.serving_framework,
        "framework_args": args.framework_args,
        "pre_launch_cmds": args.pre_launch_cmds,
        "workers": args.workers,
        "nodes_per_worker": nodes_per_worker,
        "worker_port": args.worker_port,
        "use_router": args.use_router,
        "router_environment": router_environment,
        "router_port": args.router_port,
        "router_args": args.router_args,
        "use_ocf": not args.disable_ocf,
        "ocf_bootstrap_addr": args.ocf_bootstrap_addr,
        "ocf_service_name": args.ocf_service_name,
        "ocf_service_port": args.ocf_service_port,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_file:
        generate_job_script(template_path, temp_file.name, **template_args)
        job_id = submit_job(
            temp_file.name,
            interactive=args.interactive,
            nodes=args.slurm_nodes,
            partition=args.slurm_partition,
            time=args.slurm_time,
            account=args.slurm_account,
            environment=environment
        )

        # Only show batch-specific info if not in interactive mode
        if not args.interactive and job_id:
            log_dir = f"logs/{job_id}"

            logging.info("")
            logging.info(f"Root job output will be available in: {log_dir}/log.out")
            logging.info("")
            logging.info(f"To view CSCS Dashboard:")
            logging.info(" https://console.mlp.cscs.ch/")
            logging.info("")
            logging.info("To tail all logs (from all nodes):")
            logging.info(f"  tail -f {log_dir}/*")
            logging.info("")
            logging.info("To cancel job:")
            logging.info(f"  scancel {job_id}")


if __name__ == "__main__":
    main()
