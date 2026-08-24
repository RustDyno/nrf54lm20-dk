#!/usr/bin/env bash
#
# Verify a written SD card against out/sd.img (read-only; nothing is
# written to the card).
#
#   ./verify-sd.sh /dev/sdX      (whole device, not a partition)
#
# Checks: the image magic in block 0, then a full byte compare of the
# image span. Passing also proves the card itself reads fine in a USB
# reader -- useful when the DK cannot talk to it.
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "usage: $0 /dev/sdX   (identify with lsblk first)" >&2
	exit 2
fi
dev="$1"
img="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/out/sd.img"

if [[ ! -b "$dev" ]]; then
	echo "error: $dev is not a block device" >&2
	exit 1
fi
if lsblk -no MOUNTPOINT "$dev" 2>/dev/null | grep -q .; then
	echo "error: $dev (or a partition on it) is mounted; unmount first" >&2
	exit 1
fi
if [[ ! -f "$img" ]]; then
	echo "error: $img not found (run make_sd_image.py first)" >&2
	exit 1
fi

size=$(stat -c%s "$img")
echo "image: $img ($size bytes)"

magic=$(sudo dd if="$dev" bs=512 count=1 status=none | head -c 8)
echo "card block 0 magic: '$magic' (expect 'WSPRIMG1')"
if [[ "$magic" != "WSPRIMG1" ]]; then
	echo "FAIL: wrong or missing image header -- was the image written to $dev?" >&2
	exit 1
fi

echo "comparing all $size bytes..."
if sudo cmp -n "$size" "$dev" "$img"; then
	echo "OK: card contents match sd.img exactly"
else
	echo "FAIL: card differs from sd.img (bad write or wrong device)" >&2
	exit 1
fi
