# Model Download Instructions

It's often useful to download the model once into a directory so that the serving framework doesn't have to do this. This is important when the model is very big e.g. 1T params as all nodes may attempt to do this individually.

## Setup
This script requires that you install UV since all dependencies are includeded in the `# /// script` tag so it's completely self-contained.

On the cluster it's a good idea (especially if you switch between x86 Bristen and Arm Clariden frequently) to install UV under each of these target architectures (or so that you can use a symbolic link to point to the correct one).

1. Install UV for each architecture:

**On Bristen (x86_64):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mv /users/$(whoami)/.local/bin/uv /users/$(whoami)/.local/bin/uv-amd64
mv /users/$(whoami)/.local/bin/uvx /users/$(whoami)/.local/bin/uvx-amd64
```

**On Clariden (ARM64):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mv /users/$(whoami)/.local/bin/uv /users/$(whoami)/.local/bin/uv-arm64
mv /users/$(whoami)/.local/bin/uvx /users/$(whoami)/.local/bin/uvx-arm64
```

2. Create symbolic links for automatic architecture detection:

```bash
# Detect architecture and create symlinks
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    ln -sf /users/$(whoami)/.local/bin/uv-arm64 /users/$(whoami)/.local/bin/uv
    ln -sf /users/$(whoami)/.local/bin/uvx-arm64 /users/$(whoami)/.local/bin/uvx
elif [[ "$ARCH" == "x86_64" ]]; then
    ln -sf /users/$(whoami)/.local/bin/uv-amd64 /users/$(whoami)/.local/bin/uv
    ln -sf /users/$(whoami)/.local/bin/uvx-amd64 /users/$(whoami)/.local/bin/uvx
fi
```

After creating the symlinks, you can use `uv` and `uvx` directly, and they will automatically point to the correct architecture-specific binary.

## Run

Once the symlinks are set up, you can use `uv` directly on any architecture:
```bash
uv run download_model.py --model deepseek-ai/DeepSeek-V3.1
```

Or if you haven't set up symlinks, use the architecture-specific command:
```bash
uv-arm64 run download_model.py --model deepseek-ai/DeepSeek-V3.1  # On ARM64
uv-amd64 run download_model.py --model deepseek-ai/DeepSeek-V3.1  # On x86_64
```
