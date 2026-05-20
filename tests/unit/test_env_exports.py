from swiss_ai_model_launch.launchers.framework import Sglang, Vllm


def test_sglang_exports_both_jit_deepgemm_names():
    exports = "\n".join(Sglang.env_exports)
    assert 'SGL_ENABLE_JIT_DEEPGEMM="false"' in exports
    assert 'SGLANG_ENABLE_JIT_DEEPGEMM="false"' in exports


def test_vllm_exports_ray_cgraph_timeout():
    assert "export RAY_CGRAPH_get_timeout=1800" in Vllm.env_exports
