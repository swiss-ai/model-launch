import json
import logging
import random
import shlex
import string
import subprocess
from urllib.error import URLError
from urllib.request import urlopen

from jinja2 import Template


def fetch_bootstrap_addresses(bootstrap_api_url="http://148.187.108.172:8092/v1/dnt/bootstraps", timeout=10):
    """Fetch bootstrap addresses from the OCF bootstrap API.

    Args:
        bootstrap_api_url: URL to fetch bootstrap addresses from
        timeout: Request timeout in seconds

    Returns:
        str: Bootstrap multiaddr (e.g., /ip4/148.187.108.172/tcp/43905/p2p/QmQs...)
        None: If fetch fails
    """
    try:
        logging.info(f"Fetching bootstrap addresses from {bootstrap_api_url}")
        with urlopen(bootstrap_api_url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if not data:
            logging.warning("No bootstrap nodes found in API response")
            return None

        # Handle format 1: {"bootstraps": ["/ip4/148.187.108.172/tcp/43905/p2p/QmPf4..."]}
        bootstraps = data.get("bootstraps")
        if bootstraps and isinstance(bootstraps, list) and len(bootstraps) > 0:
            bootstrap_addr = bootstraps[0]
            logging.info(f"Using bootstrap address: {bootstrap_addr}")
            return bootstrap_addr

        # Handle format 2: {"/QmPf4...": {"id": "QmPf4...", "public_address": "148.187.108.172", ...}}
        for peer_id_key, node_info in data.items():
            if not isinstance(node_info, dict):
                continue

            public_address = node_info.get("public_address")
            if not public_address:
                continue

            # Extract peer ID (remove leading slash)
            peer_id = peer_id_key.lstrip("/")

            # Construct multiaddr: /ip4/{ip}/tcp/{port}/p2p/{peer_id}
            # Using port 43905 as standard libp2p port
            bootstrap_addr = f"/ip4/{public_address}/tcp/43905/p2p/{peer_id}"
            logging.info(f"Using bootstrap address: {bootstrap_addr}")
            return bootstrap_addr

        logging.warning("No valid bootstrap addresses found in API response")
        return None

    except URLError as e:
        logging.warning(f"Failed to fetch bootstrap addresses: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse bootstrap API response: {e}")
        return None
    except (KeyError, IndexError, TypeError) as e:
        logging.warning(f"Unexpected API response format: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error fetching bootstrap addresses: {e}")
        return None


def extract_model_name(framework_args: str) -> str:
    """Extract model name from framework args (--served-model-name or --model-path)."""
    if not framework_args:
        return ""

    try:
        args = shlex.split(framework_args)

        # First priority: look for --served-model-name
        for i, arg in enumerate(args):
            if arg == "--served-model-name" and i + 1 < len(args):
                return args[i + 1]

        # Second priority: look for --model-path
        for i, arg in enumerate(args):
            if arg == "--model-path" and i + 1 < len(args):
                model_path = args[i + 1].rstrip("/")
                return model_path.split("/")[-1]
    except Exception:
        pass

    return ""


def nanoid(length: int = 4, model_name: str = "") -> str:
    """Generate a short ID, optionally incorporating model name.

    Args:
        length: Length of random suffix
        model_name: Optional model name to suffix (e.g., "Apertus-8B-Instruct-2509")

    Returns:
        A short identifier string, e.g., "AbCd-Apertus-8B-Instruct-2509" or just "AbCd"
    """
    random_id = "".join(random.choices(string.ascii_letters, k=length))

    if model_name:
        # Extract last part after / if present, sanitize, and limit length
        sanitized = model_name.split("/")[-1]
        sanitized = "".join(c if c.isalnum() or c in "-_." else "-" for c in sanitized)
        sanitized = sanitized[:40].strip("-")
        return f"{random_id}-{sanitized}"

    return random_id


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)


def generate_job_script(template_path, output_path, **kwargs):
    with open(template_path) as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)


def submit_job(
    job_script_path,
    interactive=False,
    nodes=1,
    partition="normal",
    time="04:00:00",
    account="infra01",
    environment=None,
):
    if interactive:
        # Build srun command for interactive session
        cmd = [
            "srun",
            f"--nodes={nodes}",
            f"--partition={partition}",
            f"--time={time}",
            f"--account={account}",
            "--exclusive",
            "--pty",
        ]

        if environment:
            cmd.extend(["--container-writable", f"--environment={environment}"])

        cmd.append("bash")

        logging.info(f"Launching interactive session with command: {' '.join(cmd)}")
        logging.info("Press Ctrl+D or type 'exit' to end the session")

        # Run interactively (blocking, no capture)
        try:
            subprocess.run(cmd, check=True)
            logging.info("Interactive session ended")
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"Error launching interactive session: {e}")
            raise
        except KeyboardInterrupt:
            logging.info("\nInteractive session interrupted")
            return None
    else:
        # Original batch submission logic
        try:
            result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True, check=True)
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
