#!/usr/bin/env bash
#
# Compile a TFLite (int8-quantized) model into an Axon command-buffer header and
# install it for the Rust build.
#
# Usage:
#   tools/compile-model.sh <model.tflite> <model_name> [interlayer_bytes] [psum_bytes]
#
# The compile workspace (yaml, logs, outputs/) is created NEXT TO the input
# .tflite, so model projects keep their own artifacts. The generated
# nrf_axon_model_<model_name>_.h is installed into INSTALL_DIR (default: this
# repo's vendor/include/generated/, where build.rs auto-detects, compiles and
# links it, enabling the `has_model` cfg). E.g. for the KWS firmware:
#   INSTALL_DIR=../KWS/firmware/generated tools/compile-model.sh ...
#
# The Axon Compiler runs in a python3.11 + tensorflow container. Engine defaults
# to podman; override with CONTAINER_ENGINE=docker.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler_dir="$here/tools/axon-compiler"
gen_dir="${INSTALL_DIR:-$here/vendor/include/generated}"

# Container layout expected by the Dockerfile / executor.
compiler_root_dir="bin"
executor_root_dir="/usr/local/executor_app/"
executor_work_dir="executor_input_workspace"
container_image_name="nrf-axon-compiler"

if [[ $# -lt 2 ]]; then
	echo "usage: $0 <model.tflite> <model_name> [interlayer_bytes] [psum_bytes]" >&2
	exit 2
fi

tflite_path="$1"
model_name="$2"
interlayer="${3:-}"
psum="${4:-}"

engine="${CONTAINER_ENGINE:-podman}"
if ! command -v "$engine" >/dev/null 2>&1; then
	if [[ "$engine" == "podman" ]] && command -v docker >/dev/null 2>&1; then
		engine="docker"
	else
		echo "error: container engine '$engine' not found." >&2
		exit 1
	fi
fi
if [[ ! -f "$tflite_path" ]]; then
	echo "error: tflite model not found: $tflite_path" >&2
	exit 1
fi
# Workspace = the model's own directory (mounted into the container).
work_dir="$(cd "$(dirname "$tflite_path")" && pwd)"

# A snap-confined shell (e.g. VS Code's integrated terminal) redirects
# XDG_DATA_HOME into the snap sandbox, which makes podman miss its real storage
# DB ("database configuration mismatch"). Point it back at the real home.
if [[ "$engine" == "podman" && "${XDG_DATA_HOME:-}" == *"/snap/"* ]]; then
	export XDG_DATA_HOME="$HOME/.local/share"
	echo "note: overriding snap-redirected XDG_DATA_HOME -> $XDG_DATA_HOME"
fi

mkdir -p "$work_dir" "$gen_dir"
# The container mounts the yaml's directory as the workspace, so the model must
# live alongside the yaml and be referenced by basename.
tflite_base="$(basename "$tflite_path")"
if [[ "$(readlink -f "$tflite_path")" != "$(readlink -f "$work_dir/$tflite_base")" ]]; then
	cp "$tflite_path" "$work_dir/"
fi
yaml_path="$work_dir/${model_name}.yaml"

{
	echo "${model_name}:"
	echo "  model_name: ${model_name}"
	echo "  tflite_model: ${tflite_base}"
	[[ -n "$interlayer" ]] && echo "  interlayer_buffer_size: ${interlayer}"
	[[ -n "$psum" ]] && echo "  psum_buffer_size: ${psum}"
} >"$yaml_path"

# Build the image only when missing (or REBUILD=1). The Dockerfile echoes its
# build args in RUN layers, so passing per-model values (as this script once
# did with yaml_file) busts the layer cache and re-runs the whole pip install
# (~9 min); the yaml is supplied at run time anyway, so use a constant.
if [[ "${REBUILD:-0}" == "1" ]] || ! "$engine" image exists "$container_image_name" 2>/dev/null; then
	echo "==> Building Axon Compiler image with ${engine}"
	( cd "$compiler_dir" && "$engine" build -t "$container_image_name" ./ \
		--build-arg compiler_root="$compiler_root_dir" \
		--build-arg yaml_file="input.yaml" \
		--build-arg root_dir="$executor_root_dir" \
		--build-arg work_dir="$executor_work_dir" )
else
	echo "==> Reusing Axon Compiler image (REBUILD=1 to force)"
fi

# The container reads the yaml and writes the generated header into the mounted
# workspace. For rootless podman, keep the host uid and relabel the volume so the
# output files are owned by us and writable.
mount_dst="${executor_root_dir}${executor_work_dir}"
engine_run_opts=()
if [[ "$engine" == "podman" ]]; then
	engine_run_opts+=(--userns=keep-id)
	vol="${work_dir}:${mount_dst}:z"
else
	vol="${work_dir}:${mount_dst}"
fi

echo "==> Running Axon Compiler (${engine}) on ${tflite_base}"
"$engine" run --rm "${engine_run_opts[@]}" -v "$vol" "$container_image_name" \
	"./${executor_work_dir}/${model_name}.yaml"

# The compiler writes generated sources under an outputs/ subdir of the workspace.
header="$work_dir/outputs/nrf_axon_model_${model_name}_.h"
if [[ ! -f "$header" ]]; then
	echo "error: expected output header not found: $header" >&2
	echo "       (check the compiler log above; output names follow the model_name)" >&2
	exit 1
fi

cp "$header" "$gen_dir/"
echo "==> Installed $(basename "$header") -> $gen_dir/"
echo "    Now: set INTERLAYER/PSUM_BUFFER_SIZE in build.rs + src/main.rs to the"
echo "    model's *_buffer_needed if larger than current, then: cargo build"
