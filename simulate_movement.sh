#!/bin/bash
# ============================================================
# simulate_movement.sh
# Cycles the TX power of the monitor-mode interface up and down,
# so the simulated drone's RSSI (and therefore the distance shown
# in the GUI) visibly changes over time -- simulating a drone
# approaching and moving away, for a live demo.
#
# Run this in a separate terminal WHILE start_simulation.sh and
# drone_detector.py's sniffing are both already running.
#
# Usage:
#   sudo ./simulate_movement.sh [monitor_interface]
#
# Example:
#   sudo ./simulate_movement.sh wlan0mon
# ============================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[!] Please run as root: sudo ./simulate_movement.sh wlan0mon"
    exit 1
fi

MON_IFACE=${1:-wlan0mon}

echo "[*] Simulating drone movement on $MON_IFACE. Press Ctrl+C to stop."
echo "[*] Note: not every Wi-Fi driver honors TX power changes. If the"
echo "[*] distance reading in the GUI doesn't change, see the README's"
echo "[*] troubleshooting section for this script."
echo ""

# Power levels in dBm: ramps down (drone approaching -> stronger signal
# is NOT the same as lower tx power; here we simulate by varying the
# drone's own transmit power, which is the variable we can actually
# control on this hardware) then back up.
POWERS=(20 17 14 11 8 5 8 11 14 17 20)

cleanup() {
    echo ""
    echo "[*] Stopped movement simulation."
    exit 0
}
trap cleanup INT TERM

while true; do
    for p in "${POWERS[@]}"; do
        # Try the legacy iwconfig tool first...
        if ! iwconfig "$MON_IFACE" txpower "${p}dBm" 2>/dev/null; then
            # ...fall back to the modern nl80211 'iw' tool (power in mBm = dBm * 100)
            iw dev "$MON_IFACE" set txpower fixed $((p * 100)) 2>/dev/null || true
        fi
        echo "[*] TX power set to ${p} dBm"
        sleep 3
    done
done
