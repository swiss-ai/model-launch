# Container Images

Models run inside container images that are built, scanned and published by the private `swiss-ai/model-launch-images` repository — not by this one. To add or change an image, open a pull request there.

Each image is published as two artifacts:

| Artifact | Where | Used for |
| --- | --- | --- |
| OCI image | `ghcr.io/swiss-ai/<name>:<channel>` | Inspection, secret scanning, provenance |
| squashfs | `/capstor/store/cscs/swissai/infra01/container-images/ci/<name>-<arch>.sqsh` | What pyxis mounts at launch |

A pull request in the images repository publishes to its own `pr-<N>` channel (`.../ci/pr-<N>/<name>-<arch>.sqsh`); merging it republishes under `latest`, the flat path above.

## Using an image

Point an env toml under `src/swiss_ai_model_launch/assets/envs/` at the squashfs:

```toml
image = "/capstor/store/cscs/swissai/infra01/container-images/ci/my_image-{arch}.sqsh"
mounts = ["/capstor", "/iopsstor"]
workdir = "/workspace/"
```

`{arch}` is substituted on the batch host from `uname -m` — the launcher can't know the target arch. A pinned path (`-arm64.sqsh`) is passed through untouched.

Then launch with it: `sml advanced --environment src/.../envs/my_image.toml ...`. See [Adding a new model recipe](development.md#adding-a-new-model-recipe).

To try an image from an open images-repository PR, point the env toml at its `pr-<N>` path, and revert before merging — those artifacts are deleted when that PR closes.
