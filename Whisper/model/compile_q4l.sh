#!/usr/bin/env bash
#
# Compile the level-coded per-token decoder submodels (out/submodels-q4l/,
# written by quant4_gate.py) through the Axon compiler and link them as
# slot blobs at the decode base, into out/blobs-q4l/.
#
#   ./compile_q4l.sh [firmware.elf]
#
# The compiler recomputes the per-channel bias words from the requantized
# filters (it folds -zp_in * sum(w) into them), which is why the 4-bit
# blobs must be compiled from the requantized tflites rather than patched.
# Headers land in out/axon-headers-q4l/. Already-compiled models are
# skipped unless FORCE=1.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elf="${1:-$here/../firmware/target/thumbv8m.main-none-eabihf/release/whisper-firmware}"
src="$here/out/submodels-q4l"
hdr="$here/out/axon-headers-q4l"
blobs="$here/out/blobs-q4l"
# firmware app.rs DEC_BASE: top 152 KB of the 208 KB slot
base=0x20059000
mkdir -p "$hdr" "$blobs"

n=0
for t in "$src"/*.tflite; do
	name="$(basename "$t" .tflite)"
	h="$hdr/nrf_axon_model_${name}_.h"
	if [[ "${FORCE:-0}" == "1" || ! -f "$h" ]]; then
		INSTALL_DIR="$hdr" "$here/../../npu/tools/compile-model.sh" "$t" "$name" 131072 32768 \
			>"$src/$name.compile.log" 2>&1 || { echo "compile failed: $name (see $src/$name.compile.log)"; exit 1; }
	fi
	"$here/../firmware/tools/make-blob.sh" "$h" "$elf" "$blobs/$name.bin" "$base"
	n=$((n + 1))
done
echo "$n blobs -> $blobs"
