import secrets
import string
from importlib.resources import files

from jinja2 import Template

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_TEMPLATE_PATH = files("swiss_ai_model_launch.assets").joinpath("template.jinja")


def create_salt(length: int) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))


def render_job_script(launch_args: LaunchArgs) -> str:
    template = Template(_TEMPLATE_PATH.read_text())
    return str(template.render(**launch_args.model_dump()))


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
