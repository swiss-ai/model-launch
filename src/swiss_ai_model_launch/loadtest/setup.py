from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
K6_SCRIPT = PACKAGE_ROOT / "k6" / "script.js"
DEFAULT_CLUSTER_CONTAINER_IMAGE = Path("/capstor/store/cscs/swissai/infra01/container-images/ci/k6.sqsh")
DEFAULT_CLUSTER_PROMPTS_FILE = Path("/capstor/scratch/cscs/bsezen/loadtest/prompts.json")


def resolve_prompts_file(explicit: str | Path | None = None) -> Path:
    candidates: list[str | Path | None] = [
        explicit,
        os.environ.get("SML_LOADTEST_PROMPTS_FILE"),
        DEFAULT_CLUSTER_PROMPTS_FILE,
    ]
    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        return Path(candidate).expanduser()
    return DEFAULT_CLUSTER_PROMPTS_FILE
