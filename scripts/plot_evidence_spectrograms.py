
import argparse
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def compute_mel_log_spectrogram(waveform: np.ndarray, sample_rate: int = 16000,
                                 n_mels: int = 64, n_fft: int = 1024, hop_length: int = 256):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def plot_single_file(wav_path: Path, output_dir: Path, sample_rate: int = 16000):
    waveform, sr = librosa.load(str(wav_path), sr=sample_rate)
    log_mel_spec = compute_mel_log_spectrogram(waveform, sample_rate=sr)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        log_mel_spec, sr=sr, hop_length=256, x_axis='time', y_axis='mel', ax=ax, cmap='magma',
    )
    ax.set_title(f"Mel-Log Spectrogram — {wav_path.name}")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()

    output_path = output_dir / f"{wav_path.stem}_spectrogram.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="رسم Mel-Log Spectrogram لمقاطع الأدلة الصوتية")
    parser.add_argument('--evidence-dir', type=str, required=True)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=16000)

    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.exists():
        raise FileNotFoundError(f"المجلد غير موجود: {evidence_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else evidence_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        wav_files = [evidence_dir / args.file]
    else:
        wav_files = sorted(evidence_dir.glob("*.wav"))

    if not wav_files:
        print(f"لم يتم العثور على أي ملفات .wav في: {evidence_dir}")
        return

    print(f"Found {len(wav_files)} file(s). Generating spectrograms...\n")
    for wav_path in wav_files:
        if not wav_path.exists():
            print(f"Skipped (not found): {wav_path}")
            continue
        plot_single_file(wav_path, output_dir, sample_rate=args.sample_rate)

    print(f"\nDone. Spectrogram images saved in: {output_dir}")


if __name__ == "__main__":
    main()
