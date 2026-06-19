from swiss_ai_model_launch.cli.main import (
    _build_parser,
    build_launch_args_from_advanced,
)
from swiss_ai_model_launch.launchers import FirecRESTLauncher
from swiss_ai_model_launch.launchers.framework import OCF_BOOTSTRAP_ADDR_DEV


def _minimal_advanced_args(*extra: str):
    parser = _build_parser()
    tokens = [
        "advanced",
        "--firecrest-system",
        "clariden",
        "--partition",
        "normal",
        "--serving-framework",
        "sglang",
        "--slurm-environment",
        "/path/to/env.toml",
        "--framework-args",
        "--served-model-name vendor/model-abc",
        *extra,
    ]
    return parser.parse_args(tokens)


def test_advanced_default_leaves_bootstrap_addr_unset():
    args = _minimal_advanced_args()
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.ocf_bootstrap_addr is None


def test_advanced_dev_flag_selects_dev_bootstrap_addr():
    args = _minimal_advanced_args("--dev")
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.ocf_bootstrap_addr == OCF_BOOTSTRAP_ADDR_DEV


def test_advanced_router_defaults_to_ocf():
    # Default routing strategy is OCF (mesh load-balancing) -> no SGLang router.
    args = _minimal_advanced_args()
    assert args.router == "OCF"
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.router == "OCF"


def test_advanced_router_sgl_enables_router():
    args = _minimal_advanced_args("--router", "SGL")
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.router == "SGL"


def test_advanced_router_is_case_insensitive():
    args = _minimal_advanced_args("--router", "sgl")
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.router == "SGL"


def test_advanced_explicit_addr_overrides_dev():
    custom = "/ip4/10.0.0.42/tcp/43905/p2p/QmCustomPeerXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    args = _minimal_advanced_args("--dev", "--otela-bootstrap-addr", custom)
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.ocf_bootstrap_addr == custom


def test_advanced_slurm_account_is_parsed():
    args = _minimal_advanced_args("--slurm-account", "proj99")
    assert args.slurm_account == "proj99"


def test_preconfigured_slurm_account_is_parsed():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "preconfigured",
            "--firecrest-system",
            "clariden",
            "--partition",
            "normal",
            "--slurm-account",
            "proj99",
            "--model",
            "vendor/model-abc",
            "--framework",
            "sglang",
            "--replicas",
            "1",
            "--router",
            "OCF",
            "--time",
            "02:00:00",
        ]
    )
    assert args.slurm_account == "proj99"


async def test_firecrest_from_client_uses_account():
    class FakeClient:
        async def userinfo(self, system_name):
            return {"user": {"name": "user"}, "group": {"name": "default"}}

    launcher = await FirecRESTLauncher.from_client(
        client=FakeClient(),
        system_name="clariden",
        partition="normal",
        account="proj99",
    )

    assert launcher.account == "proj99"
