from swiss_ai_model_launch.launchers.framework import Sglang, Vllm


def test_sglang_does_not_force_jit_deepgemm():
    # JIT DeepGEMM must NOT be hard-set in the launch script: a shell export
    # there would override the container env from the environment .toml, so
    # DeepEP models (which require it on) could never enable it. The default
    # lives in each env .toml instead.
    exports = "\n".join(Sglang.env_exports)
    assert "JIT_DEEPGEMM" not in exports


def test_sglang_exports_no_proxy():
    exports = "\n".join(Sglang.env_exports)
    assert 'export no_proxy="0.0.0.0,$no_proxy"' in exports


def test_vllm_exports_ray_cgraph_timeout():
    assert "export RAY_CGRAPH_get_timeout=1800" in Vllm.env_exports
