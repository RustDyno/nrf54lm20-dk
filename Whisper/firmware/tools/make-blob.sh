#!/usr/bin/env bash
#
# Turn a generated Axon model header into a runtime-loadable slot blob.
#
#   tools/make-blob.sh <nrf_axon_model_NAME_.h> <firmware.elf> <out.bin>
#
# The header is compiled into a TU whose first bytes are a slot header
# (magic + pointer to the model descriptor), linked at the fixed SLOT address
# against the firmware ELF's symbol table (interlayer buffer, driver-internal
# tables), and objcopy'd to a flat binary. The host streams the binary to
# SLOT_BASE and issues CMD_RUN_NPU.
#
# Blobs bind to one exact firmware ELF: regenerate them after every firmware
# change.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="$here/../../npu/vendor"

# Must match build.rs / src/main.rs.
interlayer=65536
psum=4096
slot_magic=0x4C415952

if [[ $# -ne 3 ]]; then
	echo "usage: $0 <nrf_axon_model_NAME_.h> <firmware.elf> <out.bin>" >&2
	exit 2
fi
header="$1"
elf="$2"
out="$3"

name="$(basename "$header")"
name="${name#nrf_axon_model_}"
name="${name%_.h}"
if [[ -z "$name" || "$name" == "$(basename "$header")" ]]; then
	echo "error: cannot derive model name from $(basename "$header")" >&2
	exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat >"$work/blob.c" <<EOF
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "$(basename "$header")"

struct slot_header {
	uint32_t magic;
	const nrf_axon_nn_compiled_model_s *model;
};
__attribute__((section(".slot.header"), used))
const struct slot_header SLOT_HEADER = { ${slot_magic}u, &model_${name} };
EOF

arm-none-eabi-gcc -c -mcpu=cortex-m33 -mthumb -mfloat-abi=hard \
	-mfpu=fpv5-sp-d16 -ffreestanding -std=c11 -Os \
	-Wno-old-style-declaration \
	-I "$vendor/include" -I "$vendor/include/drivers" \
	-I "$(cd "$(dirname "$header")" && pwd)" \
	-DNRF_AXON_INTERLAYER_BUFFER_SIZE=$interlayer \
	-DNRF_AXON_PSUM_BUFFER_SIZE=$psum \
	-o "$work/blob.o" "$work/blob.c"

# --just-symbols resolves interlayer/psum/driver symbols to the firmware's
# addresses; the driver archive fills in any driver-internal data the
# firmware image happened to garbage-collect.
arm-none-eabi-ld -T "$here/tools/slot.ld" --just-symbols="$elf" \
	-o "$work/blob.elf" "$work/blob.o" \
	"$vendor/lib/libnrf-axon-driver-internal-fpu.a"

undef="$(arm-none-eabi-nm -u "$work/blob.elf" || true)"
if [[ -n "$undef" ]]; then
	echo "error: unresolved symbols in blob:" >&2
	echo "$undef" >&2
	exit 1
fi

arm-none-eabi-objcopy -O binary "$work/blob.elf" "$out"

size=$(stat -c%s "$out")
if ((size > 208 * 1024)); then
	echo "error: blob ${size} B exceeds the 208K slot" >&2
	exit 1
fi
desc=$(arm-none-eabi-nm "$work/blob.elf" | awk "/ model_${name}\$/ {print \$1}")
echo "$(basename "$out"): ${size} B, descriptor model_${name} @ 0x${desc}"
