import asyncio
import secrets
import string
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import firecrest as f7t
import httpx

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_T = TypeVar("_T")

_FIRECREST_RETRY_DELAYS_SEC: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)


def _is_firecrest_retryable(exc: BaseException) -> bool:
    if isinstance(exc, f7t.UnexpectedStatusException):
        try:
            status = exc.responses[-1].status_code
        except (AttributeError, IndexError):
            return False
        return bool(500 <= status < 600)
    return isinstance(exc, httpx.HTTPError)


async def call_with_firecrest_retry(make_call: Callable[[], Awaitable[_T]]) -> _T:
    """Invoke ``make_call()`` with retries on transient FirecREST errors.

    Retries up to 5 times with exponential backoff (1s, 2s, 4s, 8s, 16s) on
    FirecREST 5xx responses and httpx transport errors. Other exceptions
    propagate immediately.
    """
    for attempt in range(len(_FIRECREST_RETRY_DELAYS_SEC) + 1):
        try:
            return await make_call()
        except Exception as exc:
            if not _is_firecrest_retryable(exc) or attempt == len(_FIRECREST_RETRY_DELAYS_SEC):
                raise
            await asyncio.sleep(_FIRECREST_RETRY_DELAYS_SEC[attempt])
    raise AssertionError("unreachable")  # pragma: no cover


def resolve_model_path(model: str, registry: Path, model_path: str | None = None) -> str:
    if model_path is not None:
        return model_path
    return str(registry / model)


def create_salt(length: int) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))


def render_sbatch_header(launch_args: LaunchArgs, *, reservation: str | None = None) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={launch_args.job_name}",
        f"#SBATCH --account={launch_args.account}",
        f"#SBATCH --time={launch_args.time}",
        "#SBATCH --exclusive",
        f"#SBATCH --nodes={launch_args.total_nodes}",
        f"#SBATCH --partition={launch_args.partition}",
    ]
    if reservation:
        lines.append(f"#SBATCH --reservation={reservation}")
    if launch_args.begin:
        lines.append(f"#SBATCH --begin={launch_args.begin}")
    if launch_args.dependency:
        lines.append(f"#SBATCH --dependency={launch_args.dependency}")
    lines += [
        "#SBATCH --output=logs/%j/log.out",
        "#SBATCH --error=logs/%j/log.err",
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
