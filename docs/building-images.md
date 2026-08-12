# Building Container Images

Models run inside containers built from `images/<name>/Dockerfile`. CI builds them **on the cluster** (a SLURM job running `podman build`), then publishes two artifacts:

| Artifact | Where | Used for |
| --- | --- | --- |
| OCI image | `ghcr.io/swiss-ai/<name>:<channel>` | Inspection, secret scanning, provenance |
| squashfs | `/capstor/store/cscs/swissai/infra01/container-images/ci/<name>-<arch>.sqsh` | What pyxis mounts at launch |

See [CI/CD](ci-cd.md) for the pipeline itself.

## Adding an image

**1. Create `images/<name>/Dockerfile`.** The directory name becomes the GHCR repo and the sqsh basename; CI discovers it by listing `images/`. Copy an existing one — `vllm_apertus_1.5_release` (source build off a pinned fork), `sglang_cuda13` (conda install), `sglang_0.5.10_rocm` (upstream re-tag).

> Only **top-level files** are uploaded as build context — subdirectories are skipped. A `COPY patches/foo.diff .` builds locally and fails in CI.

**2. Follow the conventions** below.

**3. Lint:** `make dockerlint`. `hadolint` gates the whole pipeline; suppress with `# hadolint ignore=DL3008` only where unavoidable.

**4. Add an env toml** under `src/swiss_ai_model_launch/assets/envs/`:

```toml
image = "/capstor/store/cscs/swissai/infra01/container-images/ci/my_image-{arch}.sqsh"
mounts = ["/capstor", "/iopsstor"]
workdir = "/workspace/"
```

`{arch}` is substituted on the batch host from `uname -m` — the launcher can't know the target arch. A pinned path (`-arm64.sqsh`) is passed through untouched.

**5. Use it:** `sml advanced --environment src/.../envs/my_image.toml ...`. See [Adding a new model recipe](development.md#adding-a-new-model-recipe).

## Dockerfile conventions

- **Pin everything.** Git deps get a branch *and* a commit; packages get `==`. Content is the cache key, so an unpinned `main` makes rebuilds silently differ.
- **Support both arches.** Every image builds on arm64 (Grace) and amd64 from the same Dockerfile. Branch on `TARGETARCH` or `uname -m`; never hardcode a platform-specific wheel. A failing arch blocks the multi-arch tag.
- **Never bake credentials.** Layers are [scanned](ci-cd.md#stage-4-secret-scan), and a pushed layer stays pullable forever even if a later layer deletes the file.
- **`SHELL` works** — CI builds with `--format docker`. OCI format ignores `SHELL`, breaking `RUN` steps that need `pipefail`.
- **Stay under 4 hours** (`--time=04:00:00`, 64 CPUs; the CI step gives up at 300 min). Tune `MAX_JOBS` or use a precompiled wheel.

## Testing before merge

PR builds publish to an isolated `pr-<N>` channel that can never overwrite main's:

| Channel | GHCR tag | capstor path |
| --- | --- | --- |
| `latest` (main) | `<name>:latest` | `.../ci/<name>-<arch>.sqsh` |
| `pr-<N>` | `<name>:pr-<N>` | `.../ci/pr-<N>/<name>-<arch>.sqsh` |

Point an env toml at the `pr-<N>` path to try it, then revert before merging — those artifacts are deleted when the PR closes.

To rebuild an image on main without changing its files, dispatch CI manually with the `image` input. Dispatch is refused from any ref other than `main`.

## Iterating locally

Faster than a CI round-trip — the same commands CI runs:

```bash
srun --partition=normal --nodes=1 --cpus-per-task=64 --time=04:00:00 --pty bash
cd images/my_image
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/podman-runtime-$$" && mkdir -p "$XDG_RUNTIME_DIR"
podman build --format docker -t my_image:dev .
enroot import -o "$SCRATCH/my_image.sqsh" "podman://my_image:dev"
```

This needs `~/.config/containers/storage.conf` pointing podman's storage at tmpfs — without it, rootless podman stores layers under `$HOME/.local/share/containers` on NFS and the first pulled layer fails with `lsetxattr ...: operation not supported`:

```toml
[storage]
driver = "overlay"
graphroot = "/dev/shm/<your-username>/root"
runroot = "/dev/shm/<your-username>/runroot"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs-1.13"
```

`/dev/shm` is the node's RAM, so clear it out (`podman system reset`) when a build tree is no longer needed. CI does the equivalent per job — see [CI/CD](ci-cd.md#stage-3-image-builds).

Then point an env toml at `$SCRATCH/my_image.sqsh`. A laptop `docker build` catches syntax and dependency errors but won't reproduce the CUDA/NCCL/libfabric environment or the arm64 path.

## Updating an image

Editing any file under `images/<name>/` invalidates the cache and rebuilds both arches. There is no version bump step.

For breaking changes, add a new directory instead of editing in place (`vllm_cuda13` → `vllm_cuda13_v2`). Env tomls and running launches reference the old sqsh path by name; replacing its contents changes what they get, with no rollback.
