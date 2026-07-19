
import sys
import argparse
import json
import csv
import time
from datetime import datetime
import librosa
import soundfile as sf
from pathlib import Path
from collections import deque

# إعداد المسارات الجذرية للمشروع لضمان استدعاء الحزم بشكل سليم
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import sounddevice as sd
from tensorflow import keras
from utils.config import Config

try:
    from models.model_trainer_unfreeze import YAMNetEmbedding
except ImportError:
    YAMNetEmbedding = None

def print_banner(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def load_model():
    """يحمّل النموذج المحسن مع الحفاظ على الأوزان المعمارية."""
    model_path = Config.MODELS_DIR / 'best_yamnet_finetuned_unfreeze.keras'
    
    if not model_path.exists():
        model_path = Config.BEST_MODEL_PATH
        
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run training first.")
        sys.exit(1)
        
    print(f"\nLoading model from: {model_path}")
    
    custom_objects = {}
    if YAMNetEmbedding is not None:
        custom_objects = {'YAMNetEmbedding': YAMNetEmbedding}
        
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print("Model loaded successfully.")
    return model

def load_best_threshold(default=0.3):
  
    threshold_file = Config.LOGS_DIR / 'threshold_optimization_external_1000.json'
    if threshold_file.exists():
        try:
            data = json.loads(threshold_file.read_text())
            return float(data['best_threshold']['threshold'])
        except Exception:
            pass
    return default

def record_chunk(duration, target_sample_rate):
    """يسجل مقطعاً صوتياً بالتردد الأصلي للجهاز ديناميكياً ثم يعيد أخذ العينات."""
    try:
        device_info = sd.query_devices(sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device, 'input')
        hardware_sample_rate = int(device_info['default_samplerate'])
    except Exception:
        hardware_sample_rate = 48000
    
    recording = sd.rec(
        int(duration * hardware_sample_rate),
        samplerate=hardware_sample_rate,
        channels=1,
        dtype='float32',
    )
    sd.wait()
    
    audio_flat = recording.flatten()
    
    resampled_audio = librosa.resample(
        audio_flat, 
        orig_sr=hardware_sample_rate, 
        target_sr=target_sample_rate
    )
    
    expected_samples = int(duration * target_sample_rate)
    if len(resampled_audio) < expected_samples:
        resampled_audio = np.pad(resampled_audio, (0, expected_samples - len(resampled_audio)))
    else:
        resampled_audio = resampled_audio[:expected_samples]
    
    return resampled_audio

def classify_chunk(model, waveform: np.ndarray, threshold: float):
    """يصنف المقطع الصوتي الخام مباشرة."""
    x = waveform.astype(np.float32).reshape(1, -1) 
    proba = float(model.predict(x, verbose=0)[0][0])
    
    is_drone = proba >= threshold
    label = "DRONE DETECTED" if is_drone else "no drone"
    
    return is_drone, proba, label

class TemporalVoter:
    """نافذة انزلاقية للتصويت الزمني (Temporal Smoothing)."""
    def __init__(self, window_size: int = 5, required_hits: int = 3):
        self.window_size = window_size
        self.required_hits = required_hits
        self.buffer = deque(maxlen=window_size)

    def update(self, is_drone_frame: bool) -> tuple[bool, int]:
        self.buffer.append(1 if is_drone_frame else 0)
        hits = sum(self.buffer)
        confirmed = hits >= self.required_hits
        return confirmed, hits


def main():
    parser = argparse.ArgumentParser(description="اختبار حي للتصنيف الصوتي عبر الميكروفون")
    parser.add_argument('--threshold', type=float, default=None, help="عتبة القرار (افتراضياً: أفضل عتبة محفوظة، أو 0.3)")
    parser.add_argument('--device', type=int, default=None, help="رقم جهاز الميكروفون (اختياري)")
    parser.add_argument('--list-devices', action='store_true', help="اعرض قائمة أجهزة الصوت المتاحة ثم اخرج")
    parser.add_argument('--window-size', type=int, default=5, help="عدد المقاطع في نافذة التصويت الزمني (افتراضي: 5)")
    parser.add_argument('--required-hits', type=int, default=3, help="عدد المقاطع الإيجابية اللازمة لتأكيد الإنذار (افتراضي: 3)")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.device is not None:
        sd.default.device = args.device

    threshold = args.threshold if args.threshold is not None else load_best_threshold()
    voter = TemporalVoter(window_size=args.window_size, required_hits=args.required_hits)

    # تجهيز مجلدات السجلات والأدلة
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_dir = Config.LOGS_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    session_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = Config.LOGS_DIR / f"live_session_{session_time_str}.csv"
    
    # إنشاء ملف السجل (CSV) وكتابة العناوين
    with open(log_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Probability', 'Status', 'Confirmed_Hits', 'Evidence_File'])

    print_banner("DroneShield AI — Live Audio Classification Unit")
    print(f"Sample Rate: {Config.SAMPLE_RATE} Hz | Chunk Size: {Config.DURATION} seconds")
    print(f"Decision Threshold: {threshold}")
    print(f"Temporal Voting: {args.required_hits} out of last {args.window_size} chunks")
    print(f"Logging session to: {log_filename}")
    print(f"Audio evidence will be saved to: {evidence_dir}")
    print("Press Ctrl+C to stop.\n")
    
    model = load_model()
    
    start_time = time.time()
    total_detections = 0
    
    try:
        while True:
            waveform = record_chunk(Config.DURATION, Config.SAMPLE_RATE)
            is_drone_frame, proba, label = classify_chunk(model, waveform, threshold)

            confirmed, hits = voter.update(is_drone_frame)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            evidence_file = "None"

            if confirmed:
                marker, status = "🚁", "DRONE DETECTED (CONFIRMED)"
                
                # حفظ المقطع الصوتي كدليل (Evidence)
                total_detections += 1
                filename = f"drone_evidence_{datetime.now().strftime('%H%M%S')}.wav"
                evidence_path = evidence_dir / filename
                sf.write(str(evidence_path), waveform, Config.SAMPLE_RATE)
                evidence_file = filename
                
            elif is_drone_frame:
                marker, status = "⚠️ ", "suspected (not confirmed)"
            else:
                marker, status = "  ", "no drone"

            # طباعة الحالة المباشرة
            print(f"[{timestamp}] {marker} confidence={proba:.3f} | votes={hits}/{args.window_size} -> {status}")

            # تسجيل القراءة في ملف السجل
            with open(log_filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, f"{proba:.4f}", status.strip(), f"{hits}/{args.window_size}", evidence_file])

    except KeyboardInterrupt:
        # إغلاق النظام برمجياً وطباعة ملخص الجلسة
        end_time = time.time()
        duration_mins = (end_time - start_time) / 60
        print_banner("Live Testing Stopped Gracefully")
        print(f"Session Duration: {duration_mins:.2f} minutes")
        print(f"Total Confirmed Detections: {total_detections}")
        print(f"Full log saved to: {log_filename}")
        if total_detections > 0:
            print(f"Audio evidence saved in: {evidence_dir}")

if __name__ == "__main__":
    main()
