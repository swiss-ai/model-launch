from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
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


def test_advanced_explicit_addr_overrides_dev():
    custom = "/ip4/10.0.0.42/tcp/43905/p2p/QmCustomPeerXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    args = _minimal_advanced_args("--dev", "--otela-bootstrap-addr", custom)
    la = build_launch_args_from_advanced(args, account="proj01", partition="normal")
    assert la.ocf_bootstrap_addr == custom
