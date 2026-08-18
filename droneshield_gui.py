"""
droneshield_gui.py
==================
واجهة رسومية موحّدة لمنظومة DroneShield AI.
"""

import csv
import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from drone_detector import (
    COLORS, LEVEL_COLORS,
    detect_monitor_iface,
    kill_process_group,
)

# ======================================================================
# PATHS
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR     = PROJECT_ROOT / "results" / "logs"
EVIDENCE_DIR = LOGS_DIR / "evidence"
USERS_FILE   = PROJECT_ROOT / "auth_users.json"
MAIN_SCRIPT  = PROJECT_ROOT / "main_droneshield.py"

# ======================================================================
# LOG LINE REGEX
# ======================================================================
LOG_LINE_RE = re.compile(
    r"\[(?P<time>\d{2}:\d{2}:\d{2})\]\s*\S+\s*"
    r"(?P<level>CRITICAL|WARNING|INFO|CLEAR)\s*"
    r"\|\s*audio=\s*(?P<audio>[\d.]+)\s*rf=\s*(?P<rf>[\d.]+)"
    r"(?:\s*\(scanned\s*(?P<scanned>\d+)\s*networks\))?\s*"
    r"\|\s*fused=\s*(?P<fused>[\d.]+)"
    r"(?:\s*\|\s*rf_conf=\s*(?P<rf_conf>\d+)\s*rf_susp=\s*(?P<rf_susp>\d+)\s*"
    r"audio_ok=\s*(?P<audio_ok>True|False))?"
)

LOG_PATH_RE = re.compile(r"Log file:\s*(?P<path>.+\.csv)")
AUDIO_CFG_RE = re.compile(
    r"threshold=(?P<threshold>[\w.]+),\s*window=(?P<window>\d+),\s*"
    r"required_hits=(?P<hits>\d+)"
)

# ======================================================================
# AUTH LAYER
# ======================================================================

def _hash_password(password: str, salt_hex: str = None):
    salt_hex = salt_hex or secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 100_000
    ).hex()
    return salt_hex, pwd_hash


def _verify_password(password: str, salt_hex: str, expected: str) -> bool:
    _, h = _hash_password(password, salt_hex)
    return hmac.compare_digest(h, expected)


def load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def ensure_admin_exists():
    users = load_users()
    if not users:
        return
    if any(r.get("role") == "admin" for r in users.values()):
        return
    earliest = min(users, key=lambda u: users[u].get("created", ""))
    users[earliest]["role"]   = "admin"
    users[earliest]["status"] = "approved"
    save_users(users)


# ======================================================================
# SHARED TTK STYLE
# ======================================================================

def apply_theme(style: ttk.Style):
    style.theme_use("clam")
    style.configure("TNotebook", background=COLORS["bg_dark"], borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=COLORS["bg_card"],
                    foreground=COLORS["text_secondary"],
                    padding=(20, 10),
                    font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", "white")])

    style.configure("Custom.Treeview",
                    background=COLORS["bg_input"],
                    foreground=COLORS["text_primary"],
                    fieldbackground=COLORS["bg_input"],
                    rowheight=26,
                    font=("Segoe UI", 9))
    style.map("Custom.Treeview",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", "white")])
    style.configure("Custom.Treeview.Heading",
                    background=COLORS["bg_card"],
                    foreground=COLORS["text_primary"],
                    font=("Segoe UI", 9, "bold"))

    style.configure("Auth.TNotebook", background=COLORS["bg_card"], borderwidth=0)
    style.configure("Auth.TNotebook.Tab",
                    background=COLORS["bg_input"],
                    foreground=COLORS["text_secondary"],
                    padding=(20, 8),
                    font=("Segoe UI", 10, "bold"))
    style.map("Auth.TNotebook.Tab",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", "white")])


# ======================================================================
# AUTH FRAME
# ======================================================================

class AuthFrame(tk.Frame):
    def __init__(self, master, on_success):
        super().__init__(master, bg=COLORS["bg_dark"])
        self.on_success = on_success
        self._build()

    def _build(self):
        card = tk.Frame(self, bg=COLORS["bg_card"], padx=40, pady=30)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="🛡  DroneShield AI",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 20, "bold")).pack(pady=(0, 5))
        tk.Label(card, text="Control Center Login",
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                 font=("Segoe UI", 10)).pack(pady=(0, 20))

        nb = ttk.Notebook(card, style="Auth.TNotebook")
        nb.pack(fill="both", expand=True)
        self.notebook = nb

        self._build_signin(nb)
        self._build_signup(nb)

    def _build_signin(self, nb):
        tab = tk.Frame(nb, bg=COLORS["bg_card"], padx=10, pady=15)
        nb.add(tab, text="Sign In")

        self.si_user = self._entry(tab, "Username")
        self.si_pass = self._entry(tab, "Password", show="*")
        self.si_err  = self._error_label(tab)
        self._button(tab, "Login", self._do_signin, COLORS["accent"]).pack(
            fill="x", pady=(10, 0)
        )
        self.si_pass.bind("<Return>", lambda e: self._do_signin())

    def _do_signin(self):
        username = self.si_user.get().strip()
        password = self.si_pass.get()
        users    = load_users()

        if not username or not password:
            self.si_err.config(text="Please fill in all fields.")
            return
        record = users.get(username)
        if not record or not _verify_password(password, record["salt"], record["hash"]):
            self.si_err.config(text="Invalid username or password.")
            return
        if record.get("status", "approved") == "pending":
            self.si_err.config(text="Account pending admin approval.")
            return

        self.si_err.config(text="")
        self.on_success(username, record.get("role", "user"))

    def _build_signup(self, nb):
        tab = tk.Frame(nb, bg=COLORS["bg_card"], padx=10, pady=15)
        nb.add(tab, text="Sign Up")

        self.su_user  = self._entry(tab, "Choose Username")
        self.su_pass  = self._entry(tab, "Choose Password", show="*")
        self.su_pass2 = self._entry(tab, "Confirm Password", show="*")
        self.su_err   = self._error_label(tab)
        self._button(tab, "Register", self._do_signup, COLORS["success"]).pack(
            fill="x", pady=(10, 0)
        )

    def _do_signup(self):
        username  = self.su_user.get().strip()
        password  = self.su_pass.get()
        password2 = self.su_pass2.get()
        users     = load_users()

        if not username or not password:
            self.su_err.config(text="Please fill in all fields.")
            return
        if password != password2:
            self.su_err.config(text="Passwords do not match.")
            return
        if len(password) < 4:
            self.su_err.config(text="Password must be at least 4 characters.")
            return
        if username in users:
            self.su_err.config(text="Username already exists.")
            return

        first    = len(users) == 0
        salt, h  = _hash_password(password)
        users[username] = {
            "salt":    salt,
            "hash":    h,
            "created": datetime.now().isoformat(timespec="seconds"),
            "role":    "admin" if first else "user",
            "status":  "approved" if first else "pending",
        }
        save_users(users)
        self.su_err.config(text="")
        msg = (
            "Account created as primary Admin. You can login now."
            if first else
            "Account registered successfully. Wait for Admin approval."
        )
        messagebox.showinfo("Sign Up", msg)
        self.notebook.select(0)
        self.si_user.delete(0, "end")
        self.si_user.insert(0, username)

    def _entry(self, parent, label: str, show: str = None):
        tk.Label(parent, text=label, bg=COLORS["bg_card"],
                 fg=COLORS["text_secondary"], font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        e = tk.Entry(parent, bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                     insertbackground=COLORS["text_primary"], relief="flat",
                     font=("Segoe UI", 11), show=show or "")
        e.pack(fill="x", ipady=6)
        return e

    def _error_label(self, parent):
        lbl = tk.Label(parent, text="", bg=COLORS["bg_card"],
                       fg=COLORS["danger"], font=("Segoe UI", 9))
        lbl.pack(pady=(5, 5))
        return lbl

    def _button(self, parent, text: str, command, color: str):
        return tk.Button(parent, text=text, command=command,
                         bg=color, fg="white",
                         font=("Segoe UI", 10, "bold"),
                         relief="flat", pady=8, cursor="hand2",
                         activebackground=color)


# ======================================================================
# CONTROL TAB
# ======================================================================

class ControlTab(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=COLORS["bg_dark"])
        self.app       = app
        self.proc      = None
        self.out_queue = queue.Queue()
        self._build_ui()
        self.after(150, self._poll_output)

    def _build_ui(self):
        opts = tk.Frame(self, bg=COLORS["bg_card"])
        opts.pack(fill="x", padx=15, pady=(15, 10))

        # Row 1: Interfaces, Audio device, Threshold
        row1 = tk.Frame(opts, bg=COLORS["bg_card"])
        row1.pack(fill="x", padx=15, pady=(12, 6))

        self.wifi_iface_var = tk.StringVar(value="wlan0")
        self._field(row1, "Wi-Fi interface:", self.wifi_iface_var, width=8)

        self.mon_iface_var  = tk.StringVar(value="wlan0mon")
        self._field(row1, "Monitor interface:", self.mon_iface_var, width=10)

        self.audio_device_var = tk.StringVar(value="2")
        self._field(row1, "Audio device:", self.audio_device_var, width=6)

        self.threshold_var = tk.StringVar(value="")
        self._field(row1, "Threshold:", self.threshold_var, width=6)

        # Row 2: Window size, hits, and toggles (تم إلغاء زر Debug)
        row2 = tk.Frame(opts, bg=COLORS["bg_card"])
        row2.pack(fill="x", padx=15, pady=(0, 6))

        self.window_var = tk.StringVar(value="5")
        self._field(row2, "Window:", self.window_var, width=4)

        self.hits_var = tk.StringVar(value="3")
        self._field(row2, "Required hits:", self.hits_var, width=4)

        self.no_audio_var = tk.BooleanVar(value=False)
        self.no_rf_var    = tk.BooleanVar(value=False)
        for text, var in (
            ("Disable audio", self.no_audio_var),
            ("Disable RF", self.no_rf_var),
        ):
            tk.Checkbutton(
                row2, text=text, variable=var,
                bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_card"],
                activebackground=COLORS["bg_card"], font=("Segoe UI", 9),
            ).pack(side="left", padx=10)

        # Row 3: Alert levels
        row3 = tk.Frame(opts, bg=COLORS["bg_card"])
        row3.pack(fill="x", padx=15, pady=(0, 6))
        tk.Label(row3, text="Alert on (siren + Telegram):",
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        
        self.alert_warning_var  = tk.BooleanVar(value=True)
        self.alert_critical_var = tk.BooleanVar(value=True)
        for text, var in (
            ("WARNING", self.alert_warning_var),
            ("CRITICAL", self.alert_critical_var),
        ):
            tk.Checkbutton(
                row3, text=text, variable=var,
                bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_card"],
                activebackground=COLORS["bg_card"], font=("Segoe UI", 9),
            ).pack(side="left", padx=10)

        # Command Preview
        preview_card = tk.Frame(opts, bg=COLORS["bg_dark"])
        preview_card.pack(fill="x", padx=15, pady=(0, 10))
        tk.Label(preview_card, text="Command preview (will run on START):",
                 bg=COLORS["bg_dark"], fg=COLORS["text_secondary"],
                 font=("Segoe UI", 8)).pack(anchor="w", padx=8, pady=(6, 0))
        self.cmd_preview_var = tk.StringVar(value="")
        tk.Label(preview_card, textvariable=self.cmd_preview_var,
                 bg=COLORS["bg_dark"], fg=COLORS["accent"],
                 font=("Consolas", 9, "bold"), anchor="w",
                 justify="left", wraplength=1100).pack(
            anchor="w", padx=8, pady=(2, 8), fill="x"
        )

        controls = tk.Frame(opts, bg=COLORS["bg_card"])
        controls.pack(fill="x", padx=15, pady=(6, 12))

        self.start_btn = tk.Button(
            controls, text="▶ START", command=self.toggle_run,
            bg=COLORS["success"], fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=20, pady=6, cursor="hand2",
        )
        self.start_btn.pack(side="left")

        self.status_dot = tk.Label(
            controls, text="● NOT RUNNING",
            bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
            font=("Segoe UI", 10, "bold"),
        )
        self.status_dot.pack(side="left", padx=15)

        if os.geteuid() != 0:
            tk.Label(
                controls,
                text="⚠️ Running without root privileges — consider using sudo.",
                bg=COLORS["bg_card"], fg=COLORS["warning"],
                font=("Segoe UI", 8),
            ).pack(side="right")

        # Live Status Panel
        status_card = tk.Frame(self, bg=COLORS["bg_card"])
        status_card.pack(fill="x", padx=15, pady=(0, 10))

        top_row = tk.Frame(status_card, bg=COLORS["bg_card"])
        top_row.pack(fill="x")

        self.level_label = tk.Label(
            top_row, text="—",
            bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
            font=("Segoe UI", 22, "bold"),
        )
        self.level_label.pack(side="left", padx=20, pady=15)

        metrics = tk.Frame(top_row, bg=COLORS["bg_card"])
        metrics.pack(side="left", padx=20)
        self.metric_vars = {}
        for key, label in (
            ("audio",         "Audio score"),
            ("rf",            "RF score"),
            ("fused",         "Fused score"),
            ("rf_conf",       "RF confirmed APs"),
            ("rf_susp",       "RF suspected APs"),
            ("audio_ok",      "Audio confirmed"),
        ):
            f = tk.Frame(metrics, bg=COLORS["bg_card"])
            f.pack(side="left", padx=12)
            tk.Label(f, text=label, bg=COLORS["bg_card"],
                     fg=COLORS["text_secondary"],
                     font=("Segoe UI", 8)).pack()
            var = tk.StringVar(value="—")
            tk.Label(f, textvariable=var, bg=COLORS["bg_card"],
                     fg=COLORS["text_primary"],
                     font=("Segoe UI", 14, "bold")).pack()
            self.metric_vars[key] = var

        bottom_row = tk.Frame(status_card, bg=COLORS["bg_card"])
        bottom_row.pack(fill="x", padx=20, pady=(0, 12))

        self.session_stats = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "CLEAR": 0}
        self.stats_vars = {}
        stats_frame = tk.Frame(bottom_row, bg=COLORS["bg_card"])
        stats_frame.pack(side="left")
        for level in ("CRITICAL", "WARNING", "INFO", "CLEAR"):
            f = tk.Frame(stats_frame, bg=COLORS["bg_card"])
            f.pack(side="left", padx=10)
            tk.Label(f, text=level, bg=COLORS["bg_card"],
                     fg=LEVEL_COLORS.get(level, COLORS["text_secondary"]),
                     font=("Segoe UI", 8, "bold")).pack()
            var = tk.StringVar(value="0")
            tk.Label(f, textvariable=var, bg=COLORS["bg_card"],
                     fg=COLORS["text_primary"],
                     font=("Segoe UI", 12, "bold")).pack()
            self.stats_vars[level] = var

        self.log_file_var = tk.StringVar(value="Log file: —")
        tk.Label(bottom_row, textvariable=self.log_file_var,
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                 font=("Consolas", 8)).pack(side="right")

        # Console Output
        console_card = tk.Frame(self, bg=COLORS["bg_card"])
        console_card.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        tk.Label(console_card, text="Console output",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        self.console = scrolledtext.ScrolledText(
            console_card,
            bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"], relief="flat",
            font=("Consolas", 9), height=15, wrap="word",
        )
        self.console.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.console.config(state="disabled")

        for var in (
            self.wifi_iface_var, self.mon_iface_var, self.audio_device_var,
            self.threshold_var, self.window_var, self.hits_var,
        ):
            var.trace_add("write", self._update_cmd_preview)
        for var in (
            self.no_audio_var, self.no_rf_var,
            self.alert_warning_var, self.alert_critical_var,
        ):
            var.trace_add("write", self._update_cmd_preview)

        self._update_cmd_preview()

    def _field(self, parent, label: str, var, width: int = 10):
        tk.Label(parent, text=label, bg=COLORS["bg_card"],
                 fg=COLORS["text_secondary"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        tk.Entry(parent, textvariable=var, width=width,
                 bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                 insertbackground=COLORS["text_primary"],
                 relief="flat", font=("Consolas", 9)).pack(
            side="left", padx=(0, 15)
        )

    def toggle_run(self):
        if self.proc is None:
            self._prepare_and_start()
        else:
            self._stop()

    def _prepare_and_start(self):
        if not MAIN_SCRIPT.exists():
            messagebox.showerror(
                "Error", f"main_droneshield.py not found at {MAIN_SCRIPT}"
            )
            return

        self.start_btn.config(state="disabled")

        if not self.no_rf_var.get():
            base = self.wifi_iface_var.get().strip() or "wlan0"
            self.status_dot.config(
                text="⏳ RUNNING airmon-ng start…", fg=COLORS["warning"]
            )
            self._append_console(f"\n$ airmon-ng start {base}\n")
            threading.Thread(
                target=self._airmon_thread, args=(base,), daemon=True
            ).start()
        else:
            iface = self.mon_iface_var.get().strip() or "wlan0mon"
            self._launch(iface)

    def _airmon_thread(self, base_iface: str):
        combined = ""
        detected = None

        try:
            r = subprocess.run(
                ["airmon-ng", "start", base_iface],
                capture_output=True, text=True, timeout=15,
            )
            combined = (r.stdout or "") + (r.stderr or "")
            time.sleep(1)
            detected = detect_monitor_iface(base_iface)

        except FileNotFoundError:
            self.after(0, lambda: self._monitor_failed(
                "airmon-ng not found. Please install aircrack-ng."
            ))
            return
        except subprocess.TimeoutExpired:
            self.after(0, lambda: self._monitor_failed(
                "airmon-ng command timed out."
            ))
            return
        except Exception as e:
            import traceback
            err_text = traceback.format_exc()
            self.after(0, lambda: self._monitor_failed(
                f"Unexpected error running airmon-ng:\n{e}\n\n{err_text}"
            ))
            return

        self.after(0, lambda: self._airmon_done(combined, detected))

    def _airmon_done(self, combined: str, detected):
        self._append_console(combined + "\n")
        iface = detected or self.mon_iface_var.get().strip() or "wlan0mon"
        if detected:
            self.mon_iface_var.set(detected)

        self._append_console(f"[GUI] Launching engine on interface: {iface}\n\n")
        self._launch(iface)

    def _monitor_failed(self, msg: str):
        self.start_btn.config(state="normal")
        self.status_dot.config(
            text="● NOT RUNNING", fg=COLORS["text_secondary"]
        )
        messagebox.showerror("Monitor Mode Error", msg)

    def _resolve_python_bin(self):
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    def _build_cmd(self, iface: str):
        python_bin = self._resolve_python_bin()

        if os.geteuid() != 0:
            cmd = ["sudo", "-E", python_bin, "-u", str(MAIN_SCRIPT), "--iface", iface]
        else:
            cmd = [python_bin, "-u", str(MAIN_SCRIPT), "--iface", iface]

        if self.audio_device_var.get().strip():
            cmd += ["--audio-device", self.audio_device_var.get().strip()]
        if self.threshold_var.get().strip():
            cmd += ["--audio-threshold", self.threshold_var.get().strip()]
        if self.window_var.get().strip():
            cmd += ["--window-size", self.window_var.get().strip()]
        if self.hits_var.get().strip():
            cmd += ["--required-hits", self.hits_var.get().strip()]
        if self.no_audio_var.get():
            cmd += ["--no-audio"]
        if self.no_rf_var.get():
            cmd += ["--no-rf"]

        levels = []
        if self.alert_warning_var.get():
            levels.append("warning")
        if self.alert_critical_var.get():
            levels.append("critical")
        cmd += ["--alert-levels", ",".join(levels)]

        return cmd

    def _update_cmd_preview(self, *_args):
        iface = self.mon_iface_var.get().strip() or "wlan0mon"
        cmd = self._build_cmd(iface)
        
        if not self.no_rf_var.get():
            base = self.wifi_iface_var.get().strip() or "wlan0"
            preview_text = f"$ sudo airmon-ng start {base}\n$ " + " ".join(cmd)
            self.cmd_preview_var.set(preview_text)
        else:
            self.cmd_preview_var.set("$ " + " ".join(cmd))

    def _launch(self, iface: str):
        cmd = self._build_cmd(iface)
        self._append_console(f"$ {' '.join(cmd)}\n")

        self.session_stats = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "CLEAR": 0}
        for level, var in self.stats_vars.items():
            var.set("0")
        self.log_file_var.set("Log file: —")
        for var in self.metric_vars.values():
            var.set("—")
        self.level_label.config(text="—", fg=COLORS["text_secondary"])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                preexec_fn=os.setsid, env=env,
            )
        except Exception as e:
            messagebox.showerror(
                "Launch Error", f"Failed to start process:\n{e}"
            )
            self.proc = None
            self.start_btn.config(state="normal")
            self.status_dot.config(
                text="● NOT RUNNING", fg=COLORS["text_secondary"]
            )
            return

        self.start_btn.config(
            text="⏹ STOP", bg=COLORS["danger"], state="normal"
        )
        self.status_dot.config(text="● RUNNING", fg=COLORS["success"])

        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        try:
            for line in self.proc.stdout:
                self.out_queue.put(line)
        except Exception:
            pass
        self.out_queue.put(None)

    def _stop(self):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        self._append_console(
            "\n[GUI] Stop requested — waiting for graceful shutdown...\n"
        )

    def _poll_output(self):
        try:
            while True:
                line = self.out_queue.get_nowait()
                if line is None:
                    self._on_process_ended()
                    break
                self._append_console(line)
                self._parse_status(line)
        except queue.Empty:
            pass
        self.after(150, self._poll_output)

    def _on_process_ended(self):
        self.proc = None
        self.start_btn.config(text="▶ START", bg=COLORS["success"])
        self.status_dot.config(
            text="● NOT RUNNING", fg=COLORS["text_secondary"]
        )
        self.app.logs_tab.refresh_file_list()

    def _append_console(self, text: str):
        self.console.config(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.config(state="disabled")

    def _parse_status(self, line: str):
        m_path = LOG_PATH_RE.search(line)
        if m_path:
            self.log_file_var.set(f"Log file: {m_path.group('path')}")
            return

        m_cfg = AUDIO_CFG_RE.search(line)
        if m_cfg:
            self._append_console(
                f"[GUI] Audio config -> threshold={m_cfg.group('threshold')}, "
                f"window={m_cfg.group('window')}, required_hits={m_cfg.group('hits')}\n"
            )
            return

        m = LOG_LINE_RE.search(line)
        if not m:
            return

        level = m.group("level")
        self.level_label.config(
            text=level,
            fg=LEVEL_COLORS.get(level, COLORS["text_primary"]),
        )
        self.metric_vars["audio"].set(m.group("audio"))
        self.metric_vars["rf"].set(m.group("rf"))
        self.metric_vars["fused"].set(m.group("fused"))
        self.metric_vars["rf_conf"].set(m.group("rf_conf") or "—")
        self.metric_vars["rf_susp"].set(m.group("rf_susp") or "—")
        self.metric_vars["audio_ok"].set(m.group("audio_ok") or "—")

        if level in self.session_stats:
            self.session_stats[level] += 1
            self.stats_vars[level].set(str(self.session_stats[level]))

    def stop_on_close(self):
        if self.proc is not None:
            kill_process_group(self.proc)


# ======================================================================
# LOGS TAB (محدث لدعم عرض مجلد evidence والمقاطع الصوتية)
# ======================================================================

class LogsTab(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=COLORS["bg_dark"])
        self._build_ui()
        self.refresh_file_list()

    def _build_ui(self):
        paned = tk.Frame(self, bg=COLORS["bg_dark"])
        paned.pack(fill="both", expand=True, padx=15, pady=15)

        left = tk.Frame(paned, bg=COLORS["bg_card"], width=300)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        hdr = tk.Frame(left, bg=COLORS["bg_card"])
        hdr.pack(fill="x", padx=10, pady=(10, 5))
        tk.Label(hdr, text="📁 System Logs & Evidence",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr, text="⟳", command=self.refresh_file_list,
                  bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9)).pack(side="right")

        cols = ("name", "modified", "size")
        self.file_tree = ttk.Treeview(
            left, columns=cols, show="headings",
            style="Custom.Treeview", height=20,
        )
        for col, w in (("name", 170), ("modified", 80), ("size", 50)):
            self.file_tree.heading(col, text=col.capitalize())
            self.file_tree.column(col, width=w, anchor="w")
        self.file_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.file_tree.bind("<<TreeviewSelect>>", self._on_select)

        right = tk.Frame(paned, bg=COLORS["bg_card"])
        right.pack(side="left", fill="both", expand=True)

        self.content_title = tk.Label(
            right, text="Select a file to preview",
            bg=COLORS["bg_card"], fg=COLORS["text_primary"],
            font=("Segoe UI", 11, "bold"), anchor="w",
        )
        self.content_title.pack(fill="x", padx=15, pady=(12, 5))

        self.content_container = tk.Frame(right, bg=COLORS["bg_card"])
        self.content_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def refresh_file_list(self):
        self.file_tree.delete(*self.file_tree.get_children())
        if not LOGS_DIR.exists():
            return

        entries = sorted(
            LOGS_DIR.iterdir(),
            key=lambda p: (not p.is_dir(), p.stat().st_mtime if p.exists() else 0),
            reverse=True,
        )

        for p in entries:
            st = p.stat()
            mod = datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")
            if p.is_dir():
                folder_node = self.file_tree.insert(
                    "", "end", iid=str(p), values=(f"📁 {p.name}", mod, "<DIR>"), open=True
                )
                try:
                    sub_entries = sorted(
                        p.iterdir(),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )
                    for sub_p in sub_entries:
                        if sub_p.is_file():
                            sub_st = sub_p.stat()
                            sub_mod = datetime.fromtimestamp(sub_st.st_mtime).strftime("%m-%d %H:%M")
                            sub_sz = f"{sub_st.st_size / 1024:.1f} KB"
                            icon = "🎵 " if sub_p.suffix.lower() in (".wav", ".mp3", ".ogg") else "📄 "
                            self.file_tree.insert(
                                folder_node, "end", iid=str(sub_p),
                                values=(f"  {icon}{sub_p.name}", sub_mod, sub_sz)
                            )
                except Exception:
                    pass
            else:
                sz = f"{st.st_size / 1024:.1f} KB"
                self.file_tree.insert(
                    "", "end", iid=str(p), values=(p.name, mod, sz)
                )

    def _on_select(self, _event):
        sel = self.file_tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if not path.exists():
            return

        for w in self.content_container.winfo_children():
            w.destroy()

        if path.is_dir():
            self.content_title.config(text=f"📁 Folder: {path.name}")
            items_count = len(list(path.iterdir())) if path.exists() else 0
            self._render_text(
                f"Directory Path: {path.resolve()}\n"
                f"Total Sub-items: {items_count}\n\n"
                f"Expand the tree on the left panel to view files inside this folder."
            )
            return

        self.content_title.config(text=path.name)
        try:
            ext = path.suffix.lower()
            if ext == ".csv":
                self._render_csv(path)
            elif ext == ".json":
                self._render_text(
                    json.dumps(
                        json.loads(path.read_text(encoding="utf-8")),
                        indent=2, ensure_ascii=False,
                    )
                )
            elif ext in (".wav", ".mp3", ".ogg", ".flac"):
                st = path.stat()
                mod_time = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                self._render_text(
                    f"🎵 Audio Evidence Recording\n"
                    f"========================\n\n"
                    f"File Name: {path.name}\n"
                    f"Full Path: {path.resolve()}\n"
                    f"File Size: {st.st_size / 1024:.2f} KB\n"
                    f"Recorded : {mod_time}\n\n"
                    f"[Recorded audio clip captured during drone detection threat event]"
                )
            else:
                self._render_text(
                    path.read_text(encoding="utf-8", errors="replace")[:200_000]
                )
        except Exception as e:
            self._render_text(f"[Error previewing file: {e}]")

    def _render_csv(self, path: Path):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
        if not rows:
            self._render_text("(Empty file)")
            return
        header    = rows[0]
        data_rows = [r for r in rows[1:] if any(c.strip() for c in r)]
        tree = ttk.Treeview(
            self.content_container, columns=header, show="headings",
            style="Custom.Treeview",
        )
        vsb = ttk.Scrollbar(
            self.content_container, orient="vertical", command=tree.yview
        )
        tree.configure(yscrollcommand=vsb.set)
        col_w = max(80, 700 // max(len(header), 1))
        for col in header:
            tree.heading(col, text=col)
            tree.column(col, width=col_w, anchor="center")
        for row in data_rows[-2000:]:
            row = row + [""] * (len(header) - len(row))
            tree.insert("", "end", values=row)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def _render_text(self, text: str):
        box = scrolledtext.ScrolledText(
            self.content_container,
            bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
            relief="flat", font=("Consolas", 9), wrap="word",
        )
        box.insert("1.0", text)
        box.config(state="disabled")
        box.pack(fill="both", expand=True)


# ======================================================================
# ADMIN TAB
# ======================================================================

class AdminTab(tk.Frame):
    def __init__(self, master, current_username: str):
        super().__init__(master, bg=COLORS["bg_dark"])
        self.current_username = current_username
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        paned = tk.Frame(self, bg=COLORS["bg_dark"])
        paned.pack(fill="both", expand=True, padx=15, pady=15)

        left = tk.Frame(paned, bg=COLORS["bg_card"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        hdr = tk.Frame(left, bg=COLORS["bg_card"])
        hdr.pack(fill="x", padx=10, pady=(10, 5))
        tk.Label(hdr, text="🕓 Pending Approvals",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr, text="⟳", command=self.refresh,
                  bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9)).pack(side="right")

        self.pending_tree = ttk.Treeview(
            left, columns=("username", "created"), show="headings",
            style="Custom.Treeview", height=14,
        )
        for col, w in (("username", 150), ("created", 150)):
            self.pending_tree.heading(col, text=col.capitalize())
            self.pending_tree.column(col, width=w, anchor="w")
        self.pending_tree.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        btns = tk.Frame(left, bg=COLORS["bg_card"])
        btns.pack(fill="x", padx=10, pady=(0, 12))
        tk.Button(btns, text="✓ Approve", command=self._approve,
                  bg=COLORS["success"], fg="white", relief="flat",
                  cursor="hand2", font=("Segoe UI", 9, "bold"),
                  padx=12, pady=4).pack(side="left", padx=(0, 8))
        tk.Button(btns, text="✕ Reject", command=self._reject,
                  bg=COLORS["danger"], fg="white", relief="flat",
                  cursor="hand2", font=("Segoe UI", 9, "bold"),
                  padx=12, pady=4).pack(side="left")

        right = tk.Frame(paned, bg=COLORS["bg_card"])
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        hdr2 = tk.Frame(right, bg=COLORS["bg_card"])
        hdr2.pack(fill="x", padx=10, pady=(10, 5))
        tk.Label(hdr2, text="👥 All Users",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Button(hdr2, text="+ Add User", command=self._add_user_dialog,
                  bg=COLORS["accent"], fg="white", relief="flat",
                  cursor="hand2",
                  font=("Segoe UI", 9, "bold")).pack(side="right")

        self.users_tree = ttk.Treeview(
            right,
            columns=("username", "role", "status", "created"),
            show="headings", style="Custom.Treeview", height=14,
        )
        for col, w in (
            ("username", 120), ("role", 70),
            ("status", 80), ("created", 130),
        ):
            self.users_tree.heading(col, text=col.capitalize())
            self.users_tree.column(col, width=w, anchor="w")
        self.users_tree.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        tk.Button(right, text="🗑 Delete Selected",
                  command=self._remove_user,
                  bg=COLORS["danger"], fg="white", relief="flat",
                  cursor="hand2", font=("Segoe UI", 9, "bold"),
                  padx=12, pady=4).pack(anchor="w", padx=10, pady=(0, 12))

    def refresh(self):
        users = load_users()
        self.pending_tree.delete(*self.pending_tree.get_children())
        self.users_tree.delete(*self.users_tree.get_children())
        for uname, rec in sorted(
            users.items(), key=lambda kv: kv[1].get("created", "")
        ):
            status  = rec.get("status", "approved")
            role    = rec.get("role", "user")
            created = rec.get("created", "")
            if status == "pending":
                self.pending_tree.insert(
                    "", "end", iid=uname, values=(uname, created)
                )
            self.users_tree.insert(
                "", "end", iid=uname,
                values=(uname, role, status, created),
            )

    def _approve(self):
        sel = self.pending_tree.selection()
        if not sel:
            return
        users = load_users()
        users[sel[0]]["status"] = "approved"
        save_users(users)
        self.refresh()

    def _reject(self):
        sel = self.pending_tree.selection()
        if not sel:
            return
        if not messagebox.askyesno("Reject", f"Reject and delete user '{sel[0]}' request?"):
            return
        users = load_users()
        users.pop(sel[0], None)
        save_users(users)
        self.refresh()

    def _remove_user(self):
        sel = self.users_tree.selection()
        if not sel:
            return
        uname = sel[0]
        if uname == self.current_username:
            messagebox.showerror("Error", "You cannot delete your own active account.")
            return
        users = load_users()
        rec   = users.get(uname, {})
        if rec.get("role") == "admin":
            remaining = sum(
                1 for u, r in users.items()
                if r.get("role") == "admin" and u != uname
            )
            if remaining == 0:
                messagebox.showerror("Error", "Cannot delete the last remaining admin.")
                return
        if not messagebox.askyesno("Delete User", f"Are you sure you want to delete user '{uname}'?"):
            return
        users.pop(uname, None)
        save_users(users)
        self.refresh()

    def _add_user_dialog(self):
        dlg = tk.Toplevel(self, bg=COLORS["bg_card"])
        dlg.title("Add New User")
        dlg.geometry("320x300")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        def lbl_entry(label: str, show=None):
            tk.Label(dlg, text=label, bg=COLORS["bg_card"],
                     fg=COLORS["text_secondary"],
                     font=("Segoe UI", 9), anchor="w").pack(
                fill="x", padx=20, pady=(12, 2)
            )
            e = tk.Entry(dlg, bg=COLORS["bg_input"],
                         fg=COLORS["text_primary"],
                         insertbackground=COLORS["text_primary"],
                         relief="flat", font=("Segoe UI", 10),
                         show=show or "")
            e.pack(fill="x", padx=20, ipady=5)
            return e

        user_e = lbl_entry("Username")
        pass_e = lbl_entry("Password", show="*")

        tk.Label(dlg, text="Role", bg=COLORS["bg_card"],
                 fg=COLORS["text_secondary"],
                 font=("Segoe UI", 9), anchor="w").pack(
            fill="x", padx=20, pady=(12, 2)
        )
        role_var = tk.StringVar(value="user")
        rf = tk.Frame(dlg, bg=COLORS["bg_card"])
        rf.pack(fill="x", padx=20)
        for text, val in (("User", "user"), ("Admin", "admin")):
            tk.Radiobutton(
                rf, text=text, variable=role_var, value=val,
                bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                selectcolor=COLORS["bg_input"],
                activebackground=COLORS["bg_card"],
            ).pack(side="left", padx=(0, 15))

        err_lbl = tk.Label(dlg, text="", bg=COLORS["bg_card"],
                           fg=COLORS["danger"], font=("Segoe UI", 8))
        err_lbl.pack(pady=(8, 0))

        def create():
            uname = user_e.get().strip()
            pw    = pass_e.get()
            if not uname or not pw:
                err_lbl.config(text="Please fill in both fields.")
                return
            if len(pw) < 4:
                err_lbl.config(text="Password must be at least 4 chars.")
                return
            users = load_users()
            if uname in users:
                err_lbl.config(text="Username already exists.")
                return
            salt, h = _hash_password(pw)
            users[uname] = {
                "salt":    salt, "hash": h,
                "created": datetime.now().isoformat(timespec="seconds"),
                "role":    role_var.get(),
                "status":  "approved",
            }
            save_users(users)
            dlg.destroy()
            self.refresh()

        tk.Button(dlg, text="Create User", command=create,
                  bg=COLORS["success"], fg="white", relief="flat",
                  font=("Segoe UI", 10, "bold"), cursor="hand2",
                  pady=6).pack(fill="x", padx=20, pady=(15, 15))


# ======================================================================
# MAIN FRAME
# ======================================================================

class MainFrame(tk.Frame):
    def __init__(self, master, app, username: str, role: str = "user"):
        super().__init__(master, bg=COLORS["bg_dark"])
        self.app = app

        topbar = tk.Frame(self, bg=COLORS["bg_card"], height=45)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="🛡 DroneShield AI — Control Center",
                 bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=15)
        tk.Label(topbar, text=f"Logged in as: {username} ({role})",
                 bg=COLORS["bg_card"], fg=COLORS["text_secondary"],
                 font=("Segoe UI", 9)).pack(side="right", padx=15)
        tk.Button(topbar, text="Logout", command=app.logout,
                  bg=COLORS["bg_input"], fg=COLORS["text_primary"],
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9)).pack(side="right", padx=5)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.control_tab = ControlTab(nb, app)
        self.logs_tab    = LogsTab(nb)
        app.logs_tab     = self.logs_tab

        nb.add(self.control_tab, text="  ▶  Control  ")
        nb.add(self.logs_tab,    text="  📋  Logs  ")

        if role == "admin":
            self.admin_tab = AdminTab(nb, username)
            nb.add(self.admin_tab, text="  🛠  Admin  ")


# ======================================================================
# ROOT APPLICATION
# ======================================================================

class DroneShieldGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DroneShield AI")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(bg=COLORS["bg_dark"])

        apply_theme(ttk.Style(self))

        self.logs_tab      = None
        self.current_frame = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        ensure_admin_exists()
        self._show_auth()

    def _show_auth(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = AuthFrame(self, on_success=self._show_main)
        self.current_frame.pack(fill="both", expand=True)

    def _show_main(self, username: str, role: str = "user"):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = MainFrame(self, self, username, role)
        self.current_frame.pack(fill="both", expand=True)

    def logout(self):
        if isinstance(self.current_frame, MainFrame):
            self.current_frame.control_tab.stop_on_close()
        self._show_auth()

    def on_close(self):
        if isinstance(self.current_frame, MainFrame):
            self.current_frame.control_tab.stop_on_close()
        self.destroy()


def main():
    app = DroneShieldGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
