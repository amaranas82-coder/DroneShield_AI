#!/usr/bin/env python3
"""
============================================================
 DroneShield - RF Drone Detection & Approximate Localization
============================================================
Detects nearby drones by matching Wi-Fi BSSID OUIs (and known
SSID name patterns) against a known-drone-vendor list, then
estimates distance from RSSI (signal strength) using the
log-distance path loss model.

Run on Kali Linux, as root, with a monitor-mode-capable Wi-Fi
adapter (e.g. LB-LINK):

    sudo python3 drone_detector.py

If the GUI fails to open under sudo with a "cannot connect to
display" error, run this once first (as your normal user, in
the same terminal session) before sudo-ing in:

    xhost +local:root
============================================================
"""

import os
import re
import sys
import csv
import glob
import time
import math
import random
import queue
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Known drone-vendor MAC OUIs (first 3 octets of the BSSID).
# NOTE FOR THE THESIS: verify/extend this list against the IEEE
# OUI registry (https://standards-oui.ieee.org/) and against the
# specific drone models you test with. This is a starting set,
# not a guaranteed-complete database.
DRONE_OUI = {
    "48:02:2A": "DJI / Ryze (Tello-class)",
    "A0:14:3D": "DJI",
    "04:D4:C4": "DJI Mavic",
    "60:60:1F": "DJI",
    "34:D2:62": "DJI",
    "00:12:1C": "Parrot",
    "90:03:B7": "Parrot",
    "A0:CC:2B": "Parrot",
    "88:F0:31": "Yuneec",
}

# Fallback: if the OUI isn't recognized, flag it anyway if the
# broadcast SSID contains one of these keywords (case-insensitive).
DRONE_SSID_KEYWORDS = ["tello", "dji", "mavic", "parrot", "yuneec", "drone"]

# RSSI -> distance model: distance = 10 ^ ((TX_POWER - RSSI) / (10 * N))
TX_POWER = -20      # calibrated RSSI at 1 meter (dBm) -- recalibrate for your adapter
PATH_LOSS_N = 2.0   # path loss exponent (2.0 = free space, higher = more obstructions

POLL_INTERVAL_SEC = 3      # how often we re-read the airodump CSV
STALE_AFTER_SEC = 12       # entries not seen for this long are considered "gone"
CSV_DIR = "/tmp"


def calculate_distance(rssi):
    """RSSI (dBm) -> approximate distance in meters."""
    try:
        return 10 ** ((TX_POWER - rssi) / (10 * PATH_LOSS_N))
    except (TypeError, ZeroDivisionError):
        return float("inf")


def classify_threat(distance_m):
    if distance_m <= 5:
        return ("خطر داهم / CRITICAL", "#e74c3c")
    elif distance_m <= 20:
        return ("قريب / NEAR", "#e67e22")
    elif distance_m <= 50:
        return ("متوسط / MODERATE", "#f1c40f")
    else:
        return ("بعيد / FAR", "#2ecc71")


def lookup_vendor(bssid, essid):
    prefix = bssid[0:8].upper()
    if prefix in DRONE_OUI:
        return DRONE_OUI[prefix]
    essid_l = (essid or "").lower()
    for kw in DRONE_SSID_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", essid_l):
            return f"Unknown OUI, matched SSID keyword '{kw}'"
    return None


# ------------------------------------------------------------
# SNIFFER: wraps airodump-ng and parses its CSV output
# ------------------------------------------------------------

class Sniffer:
    def __init__(self, iface, result_queue, debug_mode_getter=lambda: False):
        self.iface = iface
        self.result_queue = result_queue
        self.debug_mode_getter = debug_mode_getter
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.session_prefix = None
        self.log_path = None

    def start(self):
        if self.proc is not None:
            return  # already running

        # Verify the interface actually exists before wasting time.
        check = subprocess.run(["ip", "link", "show", self.iface],
                                capture_output=True, text=True)
        if check.returncode != 0:
            self.result_queue.put((
                "error",
                f"Interface '{self.iface}' does not exist. Run 'iw dev' in a terminal "
                f"to see real interface names, and set 'Monitor iface' to match."
            ))
            return

        self.session_prefix = os.path.join(CSV_DIR, f"droneshield_{int(time.time())}")
        self.log_path = self.session_prefix + ".log"
        cmd = [
            "airodump-ng",
            "--write", self.session_prefix,
            "--output-format", "csv",
            self.iface,
        ]
        try:
            log_file = open(self.log_path, "w")
            self.proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            self.result_queue.put(("error", "airodump-ng not found. Install aircrack-ng."))
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.proc = None
        # clean up temp csv/log files from this session
        if self.session_prefix:
            for f in glob.glob(self.session_prefix + "*"):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _poll_loop(self):
        csv_path = self.session_prefix + "-01.csv"
        checks_without_csv = 0
        while not self.stop_event.is_set():
            time.sleep(POLL_INTERVAL_SEC)

            # Did airodump-ng die on us? Surface why.
            if self.proc is not None and self.proc.poll() is not None:
                log_tail = self._read_log_tail()
                self.result_queue.put((
                    "error",
                    f"airodump-ng exited unexpectedly (exit code {self.proc.returncode}). "
                    f"Last log lines: {log_tail}"
                ))
                return

            if not os.path.exists(csv_path):
                checks_without_csv += 1
                if checks_without_csv >= 3:
                    log_tail = self._read_log_tail()
                    self.result_queue.put((
                        "error",
                        f"No CSV output from airodump-ng yet after "
                        f"{checks_without_csv * POLL_INTERVAL_SEC}s. Log so far: {log_tail}"
                    ))
                continue

            try:
                detections, total_ap_count = self._parse_csv(csv_path)
                self.result_queue.put(("update", (detections, total_ap_count)))
            except Exception as e:
                self.result_queue.put(("error", f"Parse error: {e}"))

    def _read_log_tail(self, max_chars=400):
        try:
            with open(self.log_path, "r", errors="ignore") as f:
                content = f.read()
            return content[-max_chars:].replace("\n", " | ") or "(log is empty)"
        except OSError:
            return "(could not read log)"

    def _parse_csv(self, csv_path):
        detections = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)

        now = datetime.now()
        debug_mode = self.debug_mode_getter()
        total_ap_count = 0

        for row in rows:
            if len(row) < 14:
                continue
            bssid = row[0].strip()
            if not _looks_like_mac(bssid):
                continue
            try:
                last_seen_str = row[2].strip()
                last_seen = datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                last_seen = now

            age = (now - last_seen).total_seconds()
            if age > STALE_AFTER_SEC:
                continue  # not currently visible

            total_ap_count += 1  # counts every currently-visible AP, drone or not

            try:
                power = int(row[8].strip())
            except (ValueError, IndexError):
                continue

            essid = row[13].strip() if len(row) > 13 else ""
            vendor = lookup_vendor(bssid, essid)
            is_drone_match = vendor is not None

            if not is_drone_match and not debug_mode:
                continue  # not a recognized drone, and debug mode is off

            distance = calculate_distance(power)
            detections.append({
                "bssid": bssid,
                "essid": essid or "(hidden)",
                "vendor": vendor or "(unmatched - not a drone)",
                "power": power,
                "distance": distance,
                "last_seen": last_seen_str,
                "is_drone_match": is_drone_match,
            })
        return detections, total_ap_count


def _extract_monitor_iface(airmon_output, base_iface):
    """
    Determine the actual monitor-mode interface name by asking the OS
    directly via `iw dev`, rather than parsing airmon-ng's prose output
    (which is inconsistent across driver/tool versions and easy to
    mis-parse -- e.g. matching the word "monitor" itself).
    """
    try:
        iw = subprocess.run(["iw", "dev"], capture_output=True, text=True)
    except FileNotFoundError:
        return None

    current_iface = None
    for line in iw.stdout.splitlines():
        line = line.strip()
        if line.startswith("Interface"):
            current_iface = line.split()[-1]
        elif line.startswith("type") and current_iface:
            iface_type = line.split()[-1]
            if iface_type == "monitor":
                # Prefer one that's clearly related to base_iface, but
                # accept any monitor-type interface as a fallback.
                if base_iface in current_iface or current_iface.startswith(base_iface):
                    return current_iface
    # second pass: no name match found, just return the first monitor iface
    current_iface = None
    for line in iw.stdout.splitlines():
        line = line.strip()
        if line.startswith("Interface"):
            current_iface = line.split()[-1]
        elif line.startswith("type") and current_iface:
            if line.split()[-1] == "monitor":
                return current_iface
    return None


def _looks_like_mac(s):
    parts = s.split(":")
    if len(parts) != 6:
        return False
    return all(len(p) == 2 for p in parts)


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------

class DroneShieldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroneShield - RF Drone Detection & Localization")
        self.geometry("1000x620")
        self.configure(bg="#1e1e1e")

        self.result_queue = queue.Queue()
        self.sniffer = None
        self.sim_proc = None
        self.detections_by_bssid = {}
        self.radar_positions = {}  # bssid -> fixed angle for stable radar dots

        self.active_drone = None
        self.active_drone_lock = threading.Lock()
        self.tts_config = self._init_tts()

        self._build_ui()
        self.after(500, self._poll_queue)

        threading.Thread(target=self._voice_loop, daemon=True).start()

    @staticmethod
    def _which(binary):
        found = subprocess.run(["which", binary], capture_output=True, text=True)
        return found.returncode == 0 and bool(found.stdout.strip())

    def _init_tts(self):
        """Pick a TTS backend: prefer offline espeak-ng/espeak; fall back to
        online gTTS + whatever audio player is available."""
        for candidate in ("espeak-ng", "espeak"):
            if self._which(candidate):
                return {"engine": "espeak", "binary": candidate}

        try:
            import gtts  # noqa: F401
            gtts_available = True
        except ImportError:
            gtts_available = False

        player = None
        for candidate in ("mpg123", "ffplay", "cvlc", "mpv"):
            if self._which(candidate):
                player = candidate
                break

        if gtts_available and player:
            return {"engine": "gtts", "player": player}

        return {"engine": None, "gtts_available": gtts_available, "player_available": player is not None}

    def _speak_gtts(self, message):
        try:
            from gtts import gTTS
            tmp_path = "/tmp/droneshield_tts.mp3"
            gTTS(text=message, lang="en").save(tmp_path)
            player = self.tts_config["player"]
            player_cmds = {
                "mpg123": ["mpg123", "-q", tmp_path],
                "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                "cvlc": ["cvlc", "--play-and-exit", "--quiet", tmp_path],
                "mpv": ["mpv", "--really-quiet", tmp_path],
            }
            result = subprocess.run(player_cmds[player], capture_output=True, text=True)
            return result.returncode == 0, (result.stderr.strip() or result.stdout.strip())
        except Exception as e:
            return False, str(e)

    def _voice_loop(self):
        warned_missing_tts = False
        warned_tts_failure = False
        while True:
            time.sleep(2)
            if not self.voice_var.get():
                continue

            engine = self.tts_config.get("engine")
            if engine is None:
                if not warned_missing_tts:
                    gtts_ok = self.tts_config.get("gtts_available")
                    player_ok = self.tts_config.get("player_available")
                    if not gtts_ok:
                        hint = "Install espeak-ng (offline, preferred): sudo apt install espeak-ng"
                    elif not player_ok:
                        hint = "gTTS is installed but no audio player was found. Install one: sudo apt install mpg123"
                    else:
                        hint = "No TTS engine available."
                    self.result_queue.put(("error", f"Voice alerts are ON but can't speak. {hint}"))
                    warned_missing_tts = True
                continue

            with self.active_drone_lock:
                drone = dict(self.active_drone) if self.active_drone else None
            if drone is None:
                continue

            distance_m = int(round(drone["distance"]))
            message = f"Alert. Alert. Drone detected. Distance approximately {distance_m} meters."

            if engine == "espeak":
                try:
                    result = subprocess.run([
                        self.tts_config["binary"],
                        "-v", "en-us+m3",  # deep male voice variant
                        "-s", "150",       # words per minute
                        "-a", "200",       # amplitude/volume (0-200)
                        "-p", "25",        # pitch (0-99, lower = deeper)
                        message,
                    ], capture_output=True, text=True)
                    if result.returncode != 0 and not warned_tts_failure:
                        detail = (result.stderr.strip() or result.stdout.strip() or
                                   "no error text returned")
                        self.result_queue.put((
                            "error",
                            f"TTS command ran but failed (exit {result.returncode}): {detail}. "
                            f"This usually means no audio device is available (common in VMs "
                            f"without a virtual sound card). Try: "
                            f"aplay /usr/share/sounds/alsa/Front_Center.wav to test audio."
                        ))
                        warned_tts_failure = True
                except FileNotFoundError:
                    self.tts_config["engine"] = None
            elif engine == "gtts":
                ok, detail = self._speak_gtts(message)
                if not ok and not warned_tts_failure:
                    self.result_queue.put((
                        "error",
                        f"gTTS playback failed: {detail}. Check internet connectivity "
                        f"and that your audio player/device work."
                    ))
                    warned_tts_failure = True

    # ---------------- UI construction ----------------

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", fieldbackground="#2b2b2b",
                         foreground="white", rowheight=26)
        style.configure("Treeview.Heading", background="#3c3c3c", foreground="white")
        style.map("Treeview", background=[("selected", "#4a6fa5")])

        top = tk.Frame(self, bg="#1e1e1e")
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(top, text="Wi-Fi interface:", bg="#1e1e1e", fg="white").pack(side="left")
        self.iface_var = tk.StringVar(value="wlan0")
        tk.Entry(top, textvariable=self.iface_var, width=10).pack(side="left", padx=5)

        tk.Label(top, text="Monitor iface:", bg="#1e1e1e", fg="white").pack(side="left", padx=(15, 0))
        self.mon_iface_var = tk.StringVar(value="wlan0mon")
        tk.Entry(top, textvariable=self.mon_iface_var, width=10).pack(side="left", padx=5)

        tk.Button(top, text="Enable Monitor Mode", command=self.enable_monitor_mode,
                  bg="#3c3c3c", fg="white").pack(side="left", padx=8)

        self.sniff_btn = tk.Button(top, text="▶ Start Sniffing", command=self.toggle_sniffing,
                                    bg="#2ecc71", fg="black", width=16)
        self.sniff_btn.pack(side="left", padx=8)

        self.sim_btn = tk.Button(top, text="Launch Drone Simulation", command=self.toggle_simulation,
                                  bg="#3c3c3c", fg="white")
        self.sim_btn.pack(side="left", padx=8)

        self.debug_var = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="Show all networks (debug)", variable=self.debug_var,
                        bg="#1e1e1e", fg="white", selectcolor="#1e1e1e",
                        activebackground="#1e1e1e", activeforeground="white").pack(side="left", padx=8)

        self.voice_var = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="🔊 Voice alerts (English)", variable=self.voice_var,
                        bg="#1e1e1e", fg="white", selectcolor="#1e1e1e",
                        activebackground="#1e1e1e", activeforeground="white").pack(side="left", padx=8)

        self.status_label = tk.Label(self, text="Status: idle", bg="#1e1e1e", fg="#aaaaaa", anchor="w")
        self.status_label.pack(fill="x", padx=12)

        self.error_label = tk.Label(self, text="", bg="#1e1e1e", fg="#e74c3c", anchor="w",
                                     wraplength=960, justify="left")
        self.error_label.pack(fill="x", padx=12)

        body = tk.Frame(self, bg="#1e1e1e")
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Radar canvas (left)
        radar_frame = tk.Frame(body, bg="#1e1e1e")
        radar_frame.pack(side="left", fill="both", expand=False)
        self.canvas = tk.Canvas(radar_frame, width=380, height=380, bg="#0d1a0d", highlightthickness=0)
        self.canvas.pack()
        self._draw_radar_grid()

        # Table (right)
        table_frame = tk.Frame(body, bg="#1e1e1e")
        table_frame.pack(side="left", fill="both", expand=True, padx=(15, 0))

        columns = ("vendor", "bssid", "essid", "power", "distance", "threat", "last_seen")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "vendor": "Vendor", "bssid": "BSSID", "essid": "SSID",
            "power": "RSSI (dBm)", "distance": "Distance (m)",
            "threat": "Threat", "last_seen": "Last Seen",
        }
        widths = {"vendor": 170, "bssid": 130, "essid": 120, "power": 90,
                  "distance": 100, "threat": 130, "last_seen": 130}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="center")
        self.tree.pack(fill="both", expand=True)

    def _draw_radar_grid(self):
        self.canvas.delete("grid")
        cx, cy = 190, 190
        for r in (60, 120, 180):
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     outline="#1f6b1f", tags="grid")
        self.canvas.create_line(cx, 10, cx, 370, fill="#1f6b1f", tags="grid")
        self.canvas.create_line(10, cy, 370, cy, fill="#1f6b1f", tags="grid")
        self.canvas.create_text(cx + 65, cy - 3, text="20m", fill="#3c9c3c", tags="grid")
        self.canvas.create_text(cx + 125, cy - 3, text="40m", fill="#3c9c3c", tags="grid")
        self.canvas.create_text(cx + 185, cy - 3, text="60m+", fill="#3c9c3c", tags="grid")

    # ---------------- actions ----------------

    def enable_monitor_mode(self):
        iface = self.iface_var.get().strip()
        if not iface:
            return
        self.status_label.config(text=f"Status: enabling monitor mode on {iface} ...")
        self.update_idletasks()

        def run():
            try:
                kill_out = subprocess.run(["airmon-ng", "check", "kill"],
                                           capture_output=True, text=True)
                print("---- airmon-ng check kill ----")
                print(kill_out.stdout + kill_out.stderr)

                start_out = subprocess.run(["airmon-ng", "start", iface],
                                            capture_output=True, text=True)
                combined = start_out.stdout + start_out.stderr
                print("---- airmon-ng start", iface, "----")
                print(combined)
                sys.stdout.flush()

                detected = _extract_monitor_iface(combined, iface)
                self.result_queue.put(("monitor_result", (combined, detected)))
            except FileNotFoundError:
                self.result_queue.put(("error", "airmon-ng not found. Install aircrack-ng."))

        threading.Thread(target=run, daemon=True).start()

    def toggle_sniffing(self):
        if self.sniffer is None:
            iface = self.mon_iface_var.get().strip() or self.iface_var.get().strip()
            self.sniffer = Sniffer(iface, self.result_queue, debug_mode_getter=self.debug_var.get)
            self.sniffer.start()
            self.sniff_btn.config(text="■ Stop Sniffing", bg="#e74c3c", fg="white")
            self.status_label.config(text=f"Status: sniffing on {iface} ...")
        else:
            self.sniffer.stop()
            self.sniffer = None
            self.sniff_btn.config(text="▶ Start Sniffing", bg="#2ecc71", fg="black")
            self.status_label.config(text="Status: sniffing stopped")

    def toggle_simulation(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_simulation.sh")
        if self.sim_proc is None:
            iface = self.iface_var.get().strip()
            try:
                self.sim_proc = subprocess.Popen(
                    ["bash", script, iface], preexec_fn=os.setsid,
                )
                self.sim_btn.config(text="Stop Drone Simulation", bg="#e67e22")
                self.status_label.config(text="Status: drone simulation running")
            except FileNotFoundError:
                messagebox.showerror("Error", f"Could not find {script}")
        else:
            try:
                os.killpg(os.getpgid(self.sim_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.sim_proc = None
            self.sim_btn.config(text="Launch Drone Simulation", bg="#3c3c3c")
            self.status_label.config(text="Status: drone simulation stopped")

    # ---------------- queue polling / rendering ----------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "update":
                    detections, total_ap_count = payload
                    self._render_detections(detections, total_ap_count)
                elif kind == "error":
                    self.error_label.config(text=f"⚠ {payload}")
                elif kind == "monitor_result":
                    combined, detected = payload
                    if detected:
                        self.mon_iface_var.set(detected)
                        self.status_label.config(
                            text=f"Status: monitor mode enabled — detected interface '{detected}' "
                                 f"(auto-filled above). Full output printed in terminal.")
                    else:
                        self.status_label.config(
                            text="Status: monitor mode command ran but no monitor interface was "
                                 "detected automatically — check the terminal output and set "
                                 "'Monitor iface' manually.")
        except queue.Empty:
            pass
        self.after(500, self._poll_queue)

    def _render_detections(self, detections, total_ap_count):
        self.tree.delete(*self.tree.get_children())
        self.canvas.delete("drone")
        self._draw_radar_grid()

        if not detections:
            with self.active_drone_lock:
                self.active_drone = None
            if total_ap_count == 0:
                self.status_label.config(
                    text="Status: sniffing... but airodump-ng sees 0 networks at all. "
                         "This points to a capture problem (wrong interface, monitor mode "
                         "not actually active, or adapter issue) rather than a filtering issue.")
            else:
                self.status_label.config(
                    text=f"Status: sniffing... airodump-ng sees {total_ap_count} network(s), "
                         f"none match known drone OUIs/SSIDs. Try 'Show all networks (debug)' "
                         f"to inspect them, or confirm your simulated/real drone is actually broadcasting.")
            return

        real_matches = [d for d in detections if d["is_drone_match"]]

        with self.active_drone_lock:
            if real_matches:
                self.active_drone = min(real_matches, key=lambda d: d["distance"])
            else:
                self.active_drone = None

        self.status_label.config(
            text=f"Status: sniffing... {len(real_matches)} drone match(es), "
                 f"{len(detections) - len(real_matches)} other network(s) shown for debug "
                 f"(out of {total_ap_count} total seen)")

        for d in detections:
            if d["is_drone_match"]:
                threat_text, color = classify_threat(d["distance"])
            else:
                threat_text, color = "not a drone", "#888888"
            self.tree.insert("", "end", values=(
                d["vendor"], d["bssid"], d["essid"], d["power"],
                f'{d["distance"]:.1f}', threat_text, d["last_seen"],
            ))
            if d["is_drone_match"]:
                self._plot_radar_dot(d, color)

    def _plot_radar_dot(self, d, color):
        bssid = d["bssid"]
        if bssid not in self.radar_positions:
            self.radar_positions[bssid] = random.uniform(0, 2 * math.pi)
        angle = self.radar_positions[bssid]

        cx, cy = 190, 190
        max_radius = 175
        # scale: 60m maps to max_radius
        r = min(d["distance"] / 60.0, 1.0) * max_radius
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)

        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill=color, outline="white", tags="drone")
        self.canvas.create_text(x, y - 14, text=d["vendor"].split(" ")[0], fill="white",
                                 font=("Arial", 8), tags="drone")

    def on_close(self):
        if self.sniffer is not None:
            self.sniffer.stop()
        if self.sim_proc is not None:
            try:
                os.killpg(os.getpgid(self.sim_proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.destroy()


def main():
    if os.geteuid() != 0:
        print("[!] This must be run as root (sudo python3 drone_detector.py) "
              "so it can control the Wi-Fi adapter.")
        sys.exit(1)

    app = DroneShieldApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
