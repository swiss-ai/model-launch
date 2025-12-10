import random
import string
import logging
import subprocess
import shlex

from jinja2 import Template


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
