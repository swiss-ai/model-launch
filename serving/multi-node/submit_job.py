import os
import random
import string
import logging
import argparse
import tempfile
import subprocess

from jinja2 import Template


def nanoid(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)


def generate_job_script(template_path, output_path, **kwargs):
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)


def submit_job(job_script_path):
    try:
        result = subprocess.run(
            ["sbatch", job_script_path], capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split("\n")

        job_id = output_lines[-1].split()[-1]
        logging.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    # SLURM-specific parameters (prefixed to avoid collisions)
    parser.add_argument("--slurm-job-name", type=str, default=nanoid())
    parser.add_argument("--slurm-nodes", type=int, required=True, help="Total number of nodes to allocate")
    parser.add_argument("--slurm-partition", type=str, default="normal")
    parser.add_argument("--slurm-environment", type=str, help="SLURM environment name (default: {framework})")

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
        "nodes": args.slurm_nodes,
        "partition": args.slurm_partition,
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
        job_id = submit_job(temp_file.name)
        logging.info(f"Job output will be available in: logs/{job_id}/log.out")


if __name__ == "__main__":
    main()
