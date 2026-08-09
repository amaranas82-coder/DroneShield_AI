"""
main_droneshield.py
====================
المحرك المركزي (Orchestrator) لمنظومة DroneShield AI.

يشغّل بالتوازي:
  1) قسم الصوت (Audio)  -> يعتمد على scripts/04_live_mic_test.py (نموذج YAMNet)
  2) قسم الراديو (RF)   -> يعتمد على drone_detector.py (Sniffer + airodump-ng)
  3) قسم الدمج (Fusion) -> يقرأ مخرجات القسمين، يطبّق الترابط الزمني ومصفوفة
                            القرار، ويسجّل كل شيء في ملف Log عند الإيقاف.

طريقة التشغيل (يتطلب صلاحية root بسبب airodump-ng):
    sudo python3 main_droneshield.py --iface wlan0mon
    

للإيقاف: Ctrl+C  -> سيتم حفظ الـ Log تلقائياً وطباعة ملخص الجلسة.
"""

import argparse
import csv
import importlib.util
import queue
import signal
import subprocess
import sys
import threading
import time
import requests
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# إعداد المسارات
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_module_from_path(module_name: str, file_path: Path):
    """تحميل ملف بايثون كوحدة، مفيد هنا لأن 04_live_mic_test.py يبدأ برقم
    ولا يمكن استيراده مباشرة بـ import عادي."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# تحميل وحدة الصوت (04_live_mic_test.py)
audio_mod = _load_module_from_path(
    "live_mic_test", SCRIPTS_DIR / "04_live_mic_test.py"
)

# تحميل وحدة الـ RF (drone_detector.py) — نبحث عنه في جذر المشروع أو داخل
# scripts/ لأن مكانه قد يختلف حسب تنظيم المستخدم للملفات.
_RF_CANDIDATES = [
    PROJECT_ROOT / "drone_detector.py",
    SCRIPTS_DIR / "drone_detector.py",
]
_rf_path = next((p for p in _RF_CANDIDATES if p.exists()), None)
if _rf_path is None:
    raise FileNotFoundError(
        "لم أجد drone_detector.py لا في جذر المشروع ولا في scripts/. "
        "تأكد من مساره ثم عدّل _RF_CANDIDATES أعلاه."
    )
rf_mod = _load_module_from_path("drone_detector_core", _rf_path)
Sniffer = rf_mod.Sniffer
calculate_distance = rf_mod.calculate_distance

from utils.config import Config  # noqa: E402

# ----------------------------------------------------------------------
# إعدادات الدمج (Fusion)
# ----------------------------------------------------------------------
TEMPORAL_WINDOW_SEC = 5      # نافذة الترابط الزمني بين الصوت والـ RF
WEIGHT_AUDIO = 0.5
WEIGHT_RF = 0.5
FUSION_LOG_EVERY_SEC = 1.0   # الحد الأدنى بين سطرين في اللوغ


# ----------------------------------------------------------------------
# 1) قسم الصوت: يشتغل كخيط منفصل ويرسل نتائجه إلى fusion_queue
# ----------------------------------------------------------------------
class AudioSection:
    def __init__(self, fusion_queue: queue.Queue, threshold=None,
                 window_size=5, required_hits=3, device=None):
        self.fusion_queue = fusion_queue
        self.stop_event = threading.Event()
        self.threshold = threshold if threshold is not None else audio_mod.load_best_threshold()
        self.voter = audio_mod.TemporalVoter(window_size, required_hits)
        self.device = device
        self.model = None

        # مجلد حفظ مقاطع الأدلة الصوتية (Evidence) — نفس منطق 04_live_mic_test.py
        self.evidence_dir = Config.LOGS_DIR / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.device is not None:
            import sounddevice as sd
            sd.default.device = self.device
        self.model = audio_mod.load_model()
        threading.Thread(target=self._loop, daemon=True, name="AudioSection").start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                waveform = audio_mod.record_chunk(Config.DURATION, Config.SAMPLE_RATE)
                is_drone_frame, proba, _ = audio_mod.classify_chunk(
                    self.model, waveform, self.threshold
                )
                confirmed, hits = self.voter.update(is_drone_frame)

                evidence_file = None
                if confirmed:
                    filename = f"drone_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    evidence_path = self.evidence_dir / filename
                    audio_mod.sf.write(str(evidence_path), waveform, Config.SAMPLE_RATE)
                    evidence_file = str(evidence_path)

                self.fusion_queue.put(("audio", {
                    "timestamp": datetime.now(),
                    "proba": proba,
                    "confirmed": confirmed,
                    "hits": hits,
                    "evidence_file": evidence_file,
                }))
            except Exception as e:
                self.fusion_queue.put(("audio_error", str(e)))
                time.sleep(1)

    def stop(self):
        self.stop_event.set()


# ----------------------------------------------------------------------
# 2) قسم الـ RF: يستخدم Sniffer الموجود في drone_detector.py كما هو
# ----------------------------------------------------------------------
class RFSection:
    def __init__(self, iface: str, fusion_queue: queue.Queue, debug=False):
        self.iface = iface
        self.fusion_queue = fusion_queue
        # Sniffer يضع مباشرة ("update", (detections, total_ap_count))
        # أو ("error", msg) في نفس الطابور -> نلتقطها في حلقة الدمج
        self.sniffer = Sniffer(iface, fusion_queue, debug_mode_getter=lambda: debug)

    def start(self):
        self.sniffer.start()

    def stop(self):
        self.sniffer.stop()


# ----------------------------------------------------------------------
# نظام التنبيه الصوتي الذكي: صفارة إنذار (Siren) + تحويل نص لكلام (TTS)
# ----------------------------------------------------------------------
class AlertManager:
    """
    يشغّل صفارة إنذار عند وصول الحالة إلى CRITICAL، بالإضافة إلى تنبيه لفظي
    (Text-to-Speech) يوضح تفاصيل الكشف. يعمل في خيوط منفصلة كي لا يوقف
    حلقة الدمج الرئيسية، ومع فترة تبريد (cooldown) لتفادي تكرار الإنذار
    كل ثانية طالما استمرت الحالة الحرجة.
    """

    def __init__(self, cooldown_sec: float = 8.0):
        self.cooldown_sec = cooldown_sec
        self._last_alert_time = 0.0
        self._lock = threading.Lock()

    def _play_siren_tone(self):
        """يولّد نغمة إنذار صناعياً (صعود/هبوط) ويشغّلها عبر aplay (ALSA مباشرة) —
        وليس عبر sounddevice، لتفادي تعارض جهاز الصوت مع خيط تسجيل المايكروفون
        الذي يعمل بالتوازي في AudioSection (كان يسبب تجمّد الحلقة عند كل WARNING)."""
        tmp_path = Path("/tmp") / f"droneshield_siren_{int(time.time() * 1000)}.wav"
        try:
            sr = 44100
            duration = 1.4
            t = np.linspace(0, duration, int(sr * duration), False)
            freq = 850 + 450 * np.sin(2 * np.pi * 2.2 * t)  # نمط صفارة صاعد-هابط
            tone = 0.5 * np.sin(2 * np.pi * freq * t)
            audio_mod.sf.write(str(tmp_path), tone.astype(np.float32), sr)

            result = subprocess.run(
                ["aplay", "-q", str(tmp_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError("aplay فشل أو غير موجود")
        except Exception as e:
            # نظام احتياطي: لو aplay غير متوفر (مثلاً على نظام غير Linux)
            try:
                audio_mod.sd.play(tone.astype(np.float32), sr)
                audio_mod.sd.wait()
            except Exception as e2:
                print(f"[AlertManager] فشل تشغيل صفارة الإنذار (aplay: {e} | sounddevice: {e2})")
        finally:
            tmp_path.unlink(missing_ok=True)

    def _speak(self, text: str):
        """تحويل نص لكلام محلي بالكامل (pyttsx3 -> espeak) — لا يحتاج إنترنت إطلاقاً،
        مناسب للسيناريو الميداني."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[AlertManager] فشل التنبيه اللفظي (pyttsx3): {e}. "
                  f"تأكد من تثبيت: sudo apt install espeak -y && pip install pyttsx3")

    def trigger_critical(self, message: str):
        """يُستدعى من FusionSection عند كل تقييم؛ يطبّق cooldown داخلياً."""
        now = time.time()
        with self._lock:
            if (now - self._last_alert_time) < self.cooldown_sec:
                return
            self._last_alert_time = now

        def _sequence():
            self._play_siren_tone()   # الصفارة أولاً
            self._speak(message)      # ثم الكلام — بالتسلسل لتفادي تعارض جهاز الإخراج

        threading.Thread(target=_sequence, daemon=True).start()


# ----------------------------------------------------------------------
# نظام إشعارات تيليجرام
# ----------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.last_sent_time = 0.0
        self.cooldown = 5.0  

    def send_alert(self, level: str, message: str, fused_score: float, rf_score: float, audio_score: float):
        now = time.time()
        if (now - self.last_sent_time) < self.cooldown:
            return
        self.last_sent_time = now

        icon = "🔴" if level == "CRITICAL" else "🟠"
        text = (
            f"{icon} *DroneShield Alert: {level}*\n\n"
            f"📝 *Details:* {message}\n"
            f"🎯 *Fused Score:* `{fused_score:.2f}%`\n"
            f"📡 *RF Score:* `{rf_score:.2f}%`\n"
            f"🎤 *Audio Score:* `{audio_score:.2f}%`\n"
            f"🕒 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        threading.Thread(target=self._send_request, args=(payload,), daemon=True).start()

    def _send_request(self, payload):
        try:
            response = requests.post(self.api_url, json=payload, timeout=5)
            if response.status_code != 200:
                print(f"[Telegram] خطأ في الإرسال: {response.text}")
        except Exception as e:
            print(f"[Telegram] فشل الاتصال بالشبكة: {e}")


# ----------------------------------------------------------------------
# 3) قسم الدمج: يقرأ من نفس الطابور، يحسب القرار النهائي، ويسجّل Log
# ----------------------------------------------------------------------
class FusionSection:
    def __init__(self, fusion_queue: queue.Queue, log_dir: Path, on_update=None,
                 alert_levels=("CRITICAL", "WARNING")):
        self.fusion_queue = fusion_queue
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.on_update = on_update  # callable(dict) اختياري، تستخدمه واجهة الويب للبث اللحظي
        self.alert_levels = {lvl.upper() for lvl in alert_levels}  # أي المستويات تُطلق الإنذار الصوتي

        self.last_audio = None          # آخر قراءة صوتية (dict)
        self.last_rf_detections = []    # آخر قائمة اكتشافات RF
        self.last_rf_time = None

        self.stats = {"CRITICAL": 0, "WARNING": 0, "INFO": 0, "CLEAR": 0}
        self._rows_buffer = []
        self.alert_manager = AlertManager(cooldown_sec=8.0)

        TELEGRAM_BOT_TOKEN = "8963368614:AAFVhGvAl-_zPwuquQGCtI8cgZCIwDa0odw"
        TELEGRAM_CHAT_ID = "-5454190789"
        self.telegram = TelegramNotifier(bot_token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID)

        session_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"fusion_session_{session_str}.csv"
        self._log_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._log_file)
        self._writer.writerow([
            "timestamp", "audio_proba", "audio_confirmed", "audio_evidence_file",
            "rf_best_score", "rf_confirmed_count", "rf_suspected_count",
            "fused_score", "threat_level"
        ])
        self._last_log_time = 0.0

    # -------------------- منطق الدمج --------------------
    def _audio_state(self):
        """يرجع (audio_hit: bool, audio_score: float) مع مراعاة النافذة الزمنية."""
        if self.last_audio is None:
            return False, 0.0
        age = (datetime.now() - self.last_audio["timestamp"]).total_seconds()
        if age > TEMPORAL_WINDOW_SEC:
            return False, 0.0
        return self.last_audio["confirmed"], self.last_audio["proba"] * 100

    def _rf_state(self):
        """يرجع (rf_confirmed, rf_suspected, rf_best_score) مع مراعاة النافذة الزمنية."""
        if self.last_rf_time is None:
            return False, False, 0.0
        age = (datetime.now() - self.last_rf_time).total_seconds()
        if age > TEMPORAL_WINDOW_SEC:
            return False, False, 0.0

        confirmed = any(d["is_drone_match"] for d in self.last_rf_detections)
        suspected = any(d["is_suspected"] for d in self.last_rf_detections)
        best_score = max((d["score"] for d in self.last_rf_detections), default=0.0)
        return confirmed, suspected, best_score

    def _decide(self):
        audio_hit, audio_score = self._audio_state()
        rf_confirmed, rf_suspected, rf_score = self._rf_state()

        fused_score = WEIGHT_AUDIO * audio_score + WEIGHT_RF * rf_score

        # مصفوفة اتخاذ القرار (كما في خطة الدمج)
        if audio_hit and rf_confirmed:
            level = "CRITICAL"
        elif audio_hit and not rf_confirmed:
            level = "WARNING"
        elif not audio_hit and rf_confirmed:
            level = "WARNING"
        elif not audio_hit and rf_suspected:
            level = "INFO"
        else:
            level = "CLEAR"

        return level, fused_score, audio_score, rf_score, rf_confirmed, rf_suspected

    # -------------------- المعالجة الرئيسية --------------------
    def handle_message(self, kind, payload):
        if kind == "audio":
            self.last_audio = payload
        elif kind == "audio_error":
            print(f"[AUDIO ERROR] {payload}")
            return
        elif kind == "update":
            detections, _total = payload
            self.last_rf_detections = detections
            self.last_rf_time = datetime.now()
        elif kind == "error":
            print(f"[RF ERROR] {payload}")
            return
        elif kind == "monitor_result":
            return
        else:
            return

        self._evaluate_and_log()

    def _evaluate_and_log(self):
        now = time.time()
        if now - self._last_log_time < FUSION_LOG_EVERY_SEC:
            return
        self._last_log_time = now

        level, fused_score, audio_score, rf_score, rf_conf, rf_susp = self._decide()
        self.stats[level] += 1

        icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🟡", "CLEAR": "🟢"}[level]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {icon} {level:<8} | audio={audio_score:5.1f} "
              f"rf={rf_score:5.1f} | fused={fused_score:5.1f}")

        if level == "CRITICAL" and level in self.alert_levels:
            alert_text = "Warning. Drone detection confirmed. Both audio and radio signatures matched."
            self.alert_manager.trigger_critical(alert_text)
            self.telegram.send_alert(level, alert_text, fused_score, rf_score, audio_score)
        elif level == "WARNING" and level in self.alert_levels:
            alert_text = "Warning. Drone detection. Drone detection."
            self.alert_manager.trigger_critical(alert_text)
            self.telegram.send_alert(level, alert_text, fused_score, rf_score, audio_score)

        confirmed_count = sum(1 for d in self.last_rf_detections if d["is_drone_match"])
        suspected_count = sum(1 for d in self.last_rf_detections if d["is_suspected"])

        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{audio_score:.2f}",
            self.last_audio["confirmed"] if self.last_audio else False,
            (self.last_audio.get("evidence_file") or "") if self.last_audio else "",
            f"{rf_score:.2f}",
            confirmed_count,
            suspected_count,
            f"{fused_score:.2f}",
            level,
        ])
        self._log_file.flush()

        if self.on_update:
            try:
                self.on_update({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "level": level,
                    "audio_score": round(audio_score, 2),
                    "rf_score": round(rf_score, 2),
                    "fused_score": round(fused_score, 2),
                    "rf_confirmed_count": confirmed_count,
                    "rf_suspected_count": suspected_count,
                    "audio_confirmed": bool(self.last_audio["confirmed"]) if self.last_audio else False,
                    "evidence_file": (self.last_audio.get("evidence_file") if self.last_audio else None),
                    "stats": dict(self.stats),
                    "log_file": str(self.log_path),
                })
            except Exception as e:
                print(f"[on_update callback error] {e}")

    def close(self):
        """يُستدعى عند إيقاف قسم الدمج -> يحفظ الملخص النهائي في نفس ملف الـ Log."""
        self._writer.writerow([])
        self._writer.writerow(["--- SESSION SUMMARY ---"])
        for level, count in self.stats.items():
            self._writer.writerow([level, count])
        self._log_file.close()
        print("\n" + "=" * 60)
        print(f"تم حفظ سجل الجلسة في: {self.log_path}")
        for level, count in self.stats.items():
            print(f"  {level}: {count}")
        print("=" * 60)


# ----------------------------------------------------------------------
# التشغيل الرئيسي
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DroneShield AI - Central Fusion Engine")
    parser.add_argument("--iface", type=str, default="wlan0mon",
                         help="واجهة الواي فاي في وضع Monitor Mode")
    parser.add_argument("--audio-device", type=int, default=None,
                         help="رقم جهاز الميكروفون (اختياري)")
    parser.add_argument("--audio-threshold", type=float, default=None,
                         help="عتبة قرار الصوت (اختياري)")
    parser.add_argument("--window-size", type=int, default=5,
                         help="حجم نافذة التصويت الزمني لقسم الصوت (عدد آخر اللقطات المحتفظ بها)")
    parser.add_argument("--required-hits", type=int, default=3,
                         help="كم لقطة من نافذة التصويت يجب أن تكون إيجابية حتى يُعلن Confirmed")
    parser.add_argument("--alert-levels", type=str, default="critical",
                         help="أي مستويات تهديد تُطلق الإنذار الصوتي (صفارة+كلام)، مفصولة بفاصلة. "
                              "مثال: --alert-levels warning  (لتعطيل الإنذار عند critical مؤقتاً)")
    parser.add_argument("--debug", action="store_true",
                         help="عرض جميع شبكات RF حتى غير المشتبه بها")
    parser.add_argument("--no-audio", action="store_true", help="تعطيل قسم الصوت")
    parser.add_argument("--no-rf", action="store_true", help="تعطيل قسم RF")
    args = parser.parse_args()

    fusion_queue = queue.Queue()
    alert_levels = [lvl.strip() for lvl in args.alert_levels.split(",") if lvl.strip()]
    fusion = FusionSection(fusion_queue, log_dir=Config.LOGS_DIR, alert_levels=alert_levels)

    audio_section = None
    rf_section = None

    if not args.no_audio:
        audio_section = AudioSection(
            fusion_queue,
            threshold=args.audio_threshold,
            device=args.audio_device,
            window_size=args.window_size,
            required_hits=args.required_hits,
        )
        print(f"[*] تشغيل قسم الصوت... (threshold={audio_section.threshold}, "
              f"window={args.window_size}, required_hits={args.required_hits})")
        audio_section.start()

    if not args.no_rf:
        rf_section = RFSection(args.iface, fusion_queue, debug=args.debug)
        print(f"[*] تشغيل قسم RF على الواجهة: {args.iface}")
        rf_section.start()

    print("[*] المحرك المركزي يعمل الآن — اضغط Ctrl+C لإيقاف الجلسة وحفظ الـ Log.\n")

    try:
        while True:
            try:
                kind, payload = fusion_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            fusion.handle_message(kind, payload)
    except KeyboardInterrupt:
        print("\n[*] إيقاف المنظومة...")
    finally:
        if audio_section:
            audio_section.stop()
        if rf_section:
            rf_section.stop()
        fusion.close()


if __name__ == "__main__":
    main()
