# CLAUDE.md

## Project Overview

**model-launch** is Swiss AI's framework-agnostic system for launching large language models on CSCS HPC clusters (Clariden/Bristen) using SLURM. It abstracts distributed inference setup for SGLang and vLLM frameworks with OCF (Open Compute Framework, formerly OpenTela) integration for service discovery.

## Repository Structure

```
model-launch/
├── download/              # Model download utilities
│   ├── download_model.py  # Downloads HuggingFace models to shared cluster storage
│   └── README.md
├── serving/               # Core job submission system
│   ├── submit_job.py      # Main entry point — CLI for SLURM job submission
│   ├── utils.py           # Helpers: bootstrap fetch, nanoid, job script gen, logging
│   ├── template.jinja     # SLURM batch script template (Jinja2)
│   ├── README.md
│   └── envs/              # TOML environment configs per framework/model
│       ├── sglang.toml
│       ├── vllm.toml
│       ├── sglang_glm.toml
│       └── sglang_kimi.toml
├── images/                # Custom container Dockerfiles
│   ├── sglang_glm5/Dockerfile
│   └── sglang_kimi_k2.5/Dockerfile
├── tests/
│   ├── unit.py            # Unit tests (bootstrap address parsing)
│   └── bootstrap_payload.json
├── requirements.txt       # Only dependency: Jinja2>=3.1.6
├── README.md
└── RUNNING_OPENCODE.md    # Instructions for OpenCode + Swiss AI API
```

## Key Workflows

### Launching a model
```bash
python serving/submit_job.py \
  --slurm-nodes N \
  --serving-framework {sglang|vllm} \
  --framework-args "..." \
  --slurm-environment serving/envs/<env>.toml
```

### Downloading a model
```bash
python download/download_model.py --model <huggingface-repo-id>
```
Models are stored at `/capstor/store/cscs/swissai/infra01/hf_models/models/{model}`.

### Running tests
```bash
python -m pytest tests/unit.py
```

## Architecture & Design Decisions

- **Framework agnostic**: Framework-specific args are passed through via `--framework-args`; the template handles SGLang vs vLLM differences
- **TOML environment configs**: Each config defines the container image, bind mounts, environment variables, and annotations for a specific framework/model combination
- **Jinja2 templating**: The SLURM script is generated from `serving/template.jinja` — supports architecture detection (ARM64/x86_64), multi-node/multi-worker orchestration, optional router, and OCF wrapping
- **OCF integration**: Service discovery via OCF binary wrapping the framework process on rank 0 of each worker
- **Multi-worker support**: Multiple independent model instances behind an optional load-balancing router

## Conventions

- **Job naming**: 4-char nanoid prefix + model name suffix (e.g., `AbCd-Apertus-8B`)
- **Ports**: Workers default to 5000, router to 30000, OCF service to 8080
- **Logs**: Written to `logs/{job_id}/worker{id}_node{rank}_{hostname}.{out|err}`
- **Python**: No formatter/linter configured; standard library style
- **Dependencies**: Minimal — only Jinja2; no virtual environment tooling enforced (README suggests `uv`)
- **No type checking or CI**: Tests are basic unit tests run manually

## Important Paths (Cluster)

- Models: `/capstor/store/cscs/swissai/infra01/hf_models/models/`
- OCF binary: `/ocfbin/ocf-{amd64|arm}`
- Container images: `/iopsstor/scratch/cscs/` and `/capstor/store/cscs/swissai/infra01/container-images/`
- Swiss AI API: `https://serving.swissai.cscs.ch`

## When Modifying This Codebase

- **Adding a new model**: Create a TOML env config in `serving/envs/`, optionally add a Dockerfile in `images/` if a custom container is needed
- **Changing job submission logic**: Edit `serving/submit_job.py` (arg parsing) and `serving/template.jinja` (generated script)
- **Changing helper functions**: Edit `serving/utils.py` and update `tests/unit.py` accordingly
- Run `python -m pytest tests/unit.py` after changes to verify nothing is broken
