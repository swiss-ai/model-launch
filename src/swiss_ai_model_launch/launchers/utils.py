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
