#!/usr/bin/env bash
# Fetch real measured channel S-parameters from the IEEE P802.3ck public
# channel repository for validation of sigil against real-world data.
#
#   https://www.ieee802.org/3/ck/public/tools/index.html
#
# These are measured backplane (KR) and copper cable (CR) channels
# contributed to the 802.3ck task force by member companies and posted
# publicly. They are NOT PCIe channels, but they are real measured 4-port
# differential interconnects in Touchstone format — exactly what the
# Touchstone/mixed-mode/eye pipeline needs for validation against reality.
#
# The data lands in examples/channels_802p3ck/ which is gitignored:
# we do not redistribute IEEE-contributed measurement data in this repo.
#
# Usage: ./fetch_802p3ck_channels.sh

set -euo pipefail

BASE="https://www.ieee802.org/3/ck/public/tools"
DEST="$(cd "$(dirname "$0")" && pwd)/channels_802p3ck"
mkdir -p "$DEST"

fetch() {
    local rel="$1"
    local name
    name="$(basename "$rel")"
    if [ -f "$DEST/$name" ]; then
        echo "already present: $name"
    else
        echo "fetching $name ..."
        curl -fsSL -o "$DEST/$name" "$BASE/$rel"
    fi
    (cd "$DEST" && unzip -q -o -d "${name%.zip}" "$name")
}

# Measured traditional backplane channels (Kareti, Nov 2018).
fetch "backplane/kareti_3ck_01_1118_backplane.zip"

# Measured OSFP 1.5 m copper cable assemblies (Tracy, Mar 2019).
fetch "cucable/tracy_3ck_02_0319_OSFP1p5m.zip"

echo
echo "Touchstone files:"
find "$DEST" -name "*.s4p" | head -20
