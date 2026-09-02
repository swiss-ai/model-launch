#!/usr/bin/env bash
#
# Post-build sanity check for the images in this repository.
#
# Runs against a *built* image rather than during the build, so what it checks
# is the artifact that actually gets published. The CI build job invokes it
# between `podman build` and `podman push`, so a failure blocks the release.
#
# Nothing here is specific to one image: every check is driven by SML_SANITY_*
# variables baked in by the image's own Dockerfile, so other images/ entries
# adopt it by declaring their own contract. An image that declares nothing
# still gets its interpreter checked.
#
#   SML_SANITY_PYTHON       expected interpreter major.minor, e.g. "3.12"
#   SML_SANITY_COMMANDS     executables that must be on PATH
#   SML_SANITY_PACKAGES     installed distributions, "name==version" or "name"
#   SML_SANITY_IMPORTS      modules that must import cleanly
#   SML_SANITY_GPU_IMPORTS  modules that only import where a GPU is visible
#   SML_SANITY_FILES        paths that must exist; symlinks must resolve
#
# SML_IMAGE_BUILD, if the image carries it, is reported but never asserted --
# it is provenance, not a property of the stack being checked.
#
# Distributions are checked through importlib.metadata rather than by being
# imported: a version pin is a packaging fact, and metadata answers it without
# paying for the import. SML_SANITY_IMPORTS is for the modules where the import
# itself is the check.
#
# A GPU is not assumed. Importing vLLM initialises platform detection, which
# needs a live CUDA runtime, so it lives in SML_SANITY_GPU_IMPORTS and is
# checked only where a GPU is actually visible: the arm64 build nodes register
# a CDI spec and pass their GH200s in, the amd64 ones do not.
#
# To run it by hand against a saved squashfs:
#   enroot start --mount ./sanity_check.sh:/sanity_check.sh <image>.sqsh \
#       bash /sanity_check.sh

set -uo pipefail

failures=0

section() { printf '\n== %s\n' "$1"; }
pass() { printf '  [ ok ] %s\n' "$1"; }
skip() { printf '  [skip] %s\n' "$1"; }
info() { printf '  [info] %s\n' "$1"; }
fail() {
    printf '  [FAIL] %s\n' "$1" >&2
    failures=$((failures + 1))
}

PYTHON="$(command -v python || command -v python3 || true)"

# Probed via /dev rather than through torch, so images with no GPU stack at all
# take the same path.
gpu_visible=0
if [ -e /dev/nvidiactl ] || compgen -G "/dev/nvidia[0-9]*" > /dev/null 2>&1; then
    gpu_visible=1
fi

# Printed first so a CI log says which build it is checking before it says
# anything about whether that build is good.
section "Build"
if [ -n "${SML_IMAGE_BUILD:-}" ]; then
    info "${SML_IMAGE_BUILD}"
else
    skip "SML_IMAGE_BUILD not set"
fi

section "Interpreter"
if [ -z "${PYTHON}" ]; then
    fail "no python on PATH"
else
    version="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
    if [ -z "${version}" ]; then
        fail "${PYTHON} does not run"
    elif [ -n "${SML_SANITY_PYTHON:-}" ] && [ "${version}" != "${SML_SANITY_PYTHON}" ]; then
        fail "python ${version} on PATH, expected ${SML_SANITY_PYTHON} (${PYTHON})"
    else
        pass "python ${version} (${PYTHON})"
    fi
fi

section "Commands on PATH"
read -ra commands <<< "${SML_SANITY_COMMANDS:-}"
if [ ${#commands[@]} -eq 0 ]; then
    skip "SML_SANITY_COMMANDS not set"
fi
for cmd in "${commands[@]}"; do
    if resolved="$(command -v "${cmd}")"; then
        pass "${cmd} -> ${resolved}"
    else
        fail "${cmd} not on PATH"
    fi
done

section "Installed distributions"
read -ra packages <<< "${SML_SANITY_PACKAGES:-}"
if [ ${#packages[@]} -eq 0 ]; then
    skip "SML_SANITY_PACKAGES not set"
elif [ -z "${PYTHON}" ]; then
    fail "cannot check packages without a python interpreter"
else
    for spec in "${packages[@]}"; do
        name="${spec%%==*}"
        expected="${spec#*==}"
        [ "${expected}" = "${spec}" ] && expected=""

        if ! actual="$("${PYTHON}" -c \
            'import sys; from importlib.metadata import version; print(version(sys.argv[1]))' \
            "${name}" 2>/dev/null)"; then
            fail "${name} is not installed"
            continue
        fi

        # PEP 440: when the pin carries no local version label, the candidate's
        # is ignored for matching -- torch==2.13.0 is satisfied by the cu130
        # index's 2.13.0+cu130. Compare local labels only when the pin has one.
        compare="${actual}"
        case "${expected}" in
            *+*) ;;
            *) compare="${actual%%+*}" ;;
        esac

        if [ -n "${expected}" ] && [ "${compare}" != "${expected}" ]; then
            fail "${name} ${actual} installed, expected ${expected}"
        else
            pass "${name} ${actual}"
        fi
    done
fi

section "Imports"
read -ra imports <<< "${SML_SANITY_IMPORTS:-}"
if [ ${#imports[@]} -eq 0 ]; then
    skip "SML_SANITY_IMPORTS not set"
fi
for module in "${imports[@]}"; do
    if output="$("${PYTHON}" -c 'import sys; __import__(sys.argv[1])' "${module}" 2>&1)"; then
        pass "import ${module}"
    else
        fail "import ${module} failed: $(printf '%s' "${output}" | tail -n 1)"
    fi
done

section "GPU-only imports"
read -ra gpu_imports <<< "${SML_SANITY_GPU_IMPORTS:-}"
if [ ${#gpu_imports[@]} -eq 0 ]; then
    skip "SML_SANITY_GPU_IMPORTS not set"
elif [ "${gpu_visible}" -eq 0 ]; then
    skip "no GPU visible; not importing: ${gpu_imports[*]}"
fi
if [ "${gpu_visible}" -eq 1 ]; then
    for module in "${gpu_imports[@]}"; do
        if output="$("${PYTHON}" -c 'import sys; __import__(sys.argv[1])' "${module}" 2>&1)"; then
            pass "import ${module}"
        else
            fail "import ${module} failed: $(printf '%s' "${output}" | tail -n 1)"
        fi
    done
fi

section "Files"
read -ra files <<< "${SML_SANITY_FILES:-}"
if [ ${#files[@]} -eq 0 ]; then
    skip "SML_SANITY_FILES not set"
fi
for path in "${files[@]}"; do
    if [ -e "${path}" ]; then
        pass "${path}"
    elif [ -L "${path}" ]; then
        fail "${path} is a dangling symlink -> $(readlink "${path}")"
    else
        fail "${path} is missing"
    fi
done

# Informational: the CUDA runtime is already asserted by the imports above --
# vLLM does not import without it. This just puts the driver's view of the
# devices in the build log next to the versions it was built against.
section "Torch runtime report"
if [ -n "${PYTHON}" ] && "${PYTHON}" -c 'import torch' > /dev/null 2>&1; then
    "${PYTHON}" - <<'PY'
import torch

print(f"  [info] torch {torch.__version__}, built against CUDA {torch.version.cuda}")
available = torch.cuda.is_available()
print(f"  [info] torch.cuda.is_available() = {available}")
if available:
    for i in range(torch.cuda.device_count()):
        print(f"  [info] device {i}: {torch.cuda.get_device_name(i)}")
PY
else
    skip "torch not importable; nothing to report"
fi

section "Result"
if [ "${failures}" -eq 0 ]; then
    printf '  all checks passed (gpu_visible=%s)\n' "${gpu_visible}"
    exit 0
fi
printf '  %s check(s) failed\n' "${failures}" >&2
exit 1
