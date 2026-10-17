# Building Container Images

Models run inside containers built and published by [swiss-ai/model-launch-images](https://github.com/swiss-ai/model-launch-images). That repository holds the Dockerfiles (`images/<name>/`) and the pipeline that builds them on the cluster, scans them and publishes:

| Artifact | Where | Used for |
| --- | --- | --- |
| squashfs | `/capstor/store/cscs/swissai/infra01/container-images/ci/<name>-<arch>.sqsh` | What pyxis mounts at launch |
| OCI image | `ghcr.io/swiss-ai/<name>:<channel>` | Inspection, secret scanning, provenance |

A pull request there publishes to its own `pr-<N>` channel (`.../ci/pr-<N>/<name>-<arch>.sqsh`) that is deleted when it closes. To add or change an image, open a pull request in that repository; its docs cover the Dockerfile conventions, testing a PR build and iterating locally.

## Using an image here

**1. Add an env toml** under `src/swiss_ai_model_launch/assets/envs/`:

```toml
image = "/capstor/store/cscs/swissai/infra01/container-images/ci/my_image-{arch}.sqsh"
mounts = ["/capstor", "/iopsstor"]
workdir = "/workspace/"
```

`{arch}` is substituted on the batch host from `uname -m` — the launcher can't know the target arch. A pinned path (`-arm64.sqsh`) is passed through untouched.

**2. Use it:** `sml advanced --environment src/.../envs/my_image.toml ...`. See [Adding a new model recipe](development.md#adding-a-new-model-recipe).

To try an image before it is merged, point the env toml at its `pr-<N>` path, then revert before merging.

Images are updated in place (a vLLM upgrade is a change to `vllm_base`, not a new image), so every env toml that names one gets the new build when it merges, with no rollback.
