#!/bin/bash
# ============================================================
# start_simulation.sh
# Simulates a drone Wi-Fi access point so drone_detector.py
# has something real to sniff during development/testing.
#
# Usage:
#   sudo ./start_simulation.sh [interface] [ssid] [channel]
#
# Example:
#   sudo ./start_simulation.sh wlan0 DJI-Tello-Sim 6
# ============================================================

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[!] Please run as root: sudo ./start_simulation.sh"
    exit 1
fi

IFACE=${1:-wlan0}
MON_IFACE="${IFACE}mon"
SSID=${2:-"DJI-Tello-Sim"}
CHANNEL=${3:-6}

echo "[*] Checking current interface state..."
if iw dev | grep -A 1 "Interface ${IFACE}$" | grep -q "type monitor"; then
    echo "[*] $IFACE is already in monitor mode, skipping airmon-ng start."
    MON_IFACE="$IFACE"
elif iw dev | grep -q "Interface ${IFACE}mon"; then
    echo "[*] ${IFACE}mon already exists and is in monitor mode, using it."
    MON_IFACE="${IFACE}mon"
else
    echo "[*] Killing processes that may interfere with monitor mode..."
    airmon-ng check kill

    echo "[*] Enabling monitor mode on $IFACE ..."
    airmon-ng start "$IFACE" || true

    DETECTED_MON=$(iw dev | awk '/type monitor/{found=1} /^\s*Interface/{iface=$2} found && /type monitor/{print iface; exit}')
    if [ -n "$DETECTED_MON" ]; then
        MON_IFACE="$DETECTED_MON"
    fi
fi

echo "[*] Monitor interface: $MON_IFACE"
echo "[*] Starting simulated drone AP: '$SSID' on channel $CHANNEL ..."

airbase-ng -e "$SSID" -c "$CHANNEL" "$MON_IFACE" &
AIRBASE_PID=$!

sleep 3

echo "[*] Bringing up at0 with an IP ..."
ip addr add 192.168.2.1/24 dev at0 2>/dev/null || true
ip link set at0 up

echo "[*] Starting DHCP server on at0 ..."
dnsmasq -i at0 --dhcp-range=192.168.2.2,192.168.2.100,12h --no-daemon &
DNSMASQ_PID=$!

echo ""
echo "[+] Simulated drone '$SSID' is now broadcasting on channel $CHANNEL."
echo "[+] Leave this window open. Press Ctrl+C to stop the simulation."
echo ""

cleanup() {
    echo ""
    echo "[*] Stopping simulation..."
    kill "$AIRBASE_PID" 2>/dev/null || true
    kill "$DNSMASQ_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

wait
