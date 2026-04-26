import secrets
import string
from importlib.resources import files
from pathlib import Path

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_SCRIPT_PATH = files("swiss_ai_model_launch.assets").joinpath("script.sh")


def resolve_model_path(model: str, registry: Path, model_path: str | None = None) -> str:
    """Return the filesystem path for a model.

    If *model_path* is provided it is returned as-is, allowing callers to
    point at an arbitrary local copy.  Otherwise the path is constructed by
    joining *registry* with *model* (e.g. ``registry / "vendor/name"``),
    which is the standard layout used by the CSCS model registry.
    """
    if model_path is not None:
        return model_path
    return str(registry / model)


def create_salt(length: int) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))


def get_job_script() -> str:
    return _SCRIPT_PATH.read_text()


def render_sbatch_header(launch_args: LaunchArgs) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={launch_args.job_name}",
        f"#SBATCH --account={launch_args.account}",
        f"#SBATCH --time={launch_args.time}",
        "#SBATCH --exclusive",
        f"#SBATCH --nodes={launch_args.nodes}",
        f"#SBATCH --partition={launch_args.partition}",
    ]
    if launch_args.reservation:
        lines.append(f"#SBATCH --reservation={launch_args.reservation}")
    lines += [
        "#SBATCH --output=logs/%j/log.out",
        "#SBATCH --error=logs/%j/log.out",
    ]
    return "\n".join(lines) + "\n"


def decode_log(data: bytes) -> str:
    """Decode log bytes to string, tolerating partial UTF-8 sequences at the tail.

    Log files may be read while the writer is mid-flush, leaving an incomplete
    multi-byte UTF-8 sequence at the end. This function strips any such trailing
    incomplete sequence before decoding, then falls back to errors="replace" for
    any other malformed bytes in the content.
    """
    # Strip trailing incomplete multi-byte UTF-8 sequence.
    # Walk backwards past continuation bytes (10xxxxxx), then check whether the
    # leading byte at that position expects more continuation bytes than are present.
    i = len(data) - 1
    num_continuation = 0
    while i >= 0 and (data[i] & 0xC0) == 0x80:
        num_continuation += 1
        i -= 1
    if i >= 0:
        lead = data[i]
        if lead & 0xE0 == 0xC0:
            expected = 1
        elif lead & 0xF0 == 0xE0:
            expected = 2
        elif lead & 0xF8 == 0xF0:
            expected = 3
        else:
            expected = num_continuation  # single-byte or already complete
        if num_continuation < expected:
            data = data[:i]
    return data.decode("utf-8", errors="replace")
