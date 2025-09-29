"""
Standalone DOA (Direction of Arrival) Demo
==========================================

A simple audio-only test for ReSpeaker microphone array direction detection.
This is useful for:
- Testing audio hardware without camera
- Debugging audio issues in isolation  
- Educational reference for GCC-PHAT algorithm
- Quick audio-only validation

Note: This duplicates the core algorithm from speaker_detection.py
but provides a simpler, audio-only testing interface.
"""

import numpy as np
import sounddevice as sd
import time
from collections import deque

# ========= CONFIG =========
DEVICE_INDEX = 0       # ReSpeaker 4 Mic Array (UAC1.0) - consistent with main system
FS = 16000
N_CHANNELS = 6         # as macOS reports
MIC_INDEXES = [2, 3, 4, 5]   # the 4 real microphones on ReSpeaker UAC1.0
FRAME_SAMPLES = 1024         # analysis window (64 ms @16k)
HOP_SAMPLES   = 512          # 50% overlap

# Mic geometry (linear array) — adjust spacing if you’ve measured it
SPACING = 0.035  # meters between adjacent mics (~3.5 cm)
C = 343.0        # speed of sound m/s

# Choose mic pairs (outer + diagonals) for robust TDOA averaging
PAIRS = [
    (0, 3, 3*SPACING),  # outermost pair: d = 3*spacing
    (0, 2, 2*SPACING),
    (1, 3, 2*SPACING),
    (1, 2, 1*SPACING),
    (0, 1, 1*SPACING),
    (2, 3, 1*SPACING),
]

# Optional calibration offset (deg). Measure later; keep 0.0 for now.
AZ_OFFSET = 0.0

# ========= GCC-PHAT DOA =========
def gcc_phat(sig, ref, fs, max_tau=None, interp=8):
    """Return (tau, cc) where tau is the delay of sig relative to ref (seconds)."""
    n = sig.shape[0] + ref.shape[0]
    nfft = 1 << (n - 1).bit_length()
    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)
    R = SIG * np.conj(REF)
    denom = np.abs(R)
    denom[denom == 0] = 1e-15
    R /= denom

    cc = np.fft.irfft(R, n=nfft*interp)
    max_shift = int(interp * nfft / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    if max_tau is not None:
        max_shift_lim = min(int(interp * fs * max_tau), max_shift)
        mid = max_shift
        cc = cc[mid - max_shift_lim: mid + max_shift_lim + 1]
        shift = np.argmax(np.abs(cc)) - max_shift_lim
    else:
        shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc

def tdoa_to_angle(tau, d):
    """theta = arcsin(c * tau / d), return degrees in [0,180]."""
    val = np.clip((C * tau) / max(d, 1e-6), -1.0, 1.0)
    theta = np.degrees(np.arcsin(val))
    # Abs because linear array cannot resolve left/right without extra cues
    return abs(theta)

# ========= Stream + processing =========
buf = deque(maxlen=4)  # rolling overlap

def process_block(x4):
    # x4 shape: (samples, 4)
    angles = []
    for (i, j, d) in PAIRS:
        tau, _ = gcc_phat(x4[:, i], x4[:, j], FS, max_tau=d/C, interp=8)
        angles.append(tdoa_to_angle(tau, d))
    # robust average (trim extremes)
    angles = np.array(angles)
    if angles.size >= 4:
        angles = np.sort(angles)[1:-1]  # drop min/max
    az = np.mean(angles) if angles.size else 0.0
    az = (az - AZ_OFFSET) % 180.0
    return az

def callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    # take the 4 real mic channels
    x = indata[:, MIC_INDEXES].astype(np.float32)  # (frames, 4)
    buf.append(x)
    cat = np.concatenate(list(buf), axis=0)
    # process in HOP steps to keep up
    while cat.shape[0] >= FRAME_SAMPLES:
        frame = cat[:FRAME_SAMPLES, :]
        cat = cat[HOP_SAMPLES:, :]
        az = process_block(frame)
        print(f"Estimated azimuth (linear): {az:6.1f}°   ", end="\r")

def main():
    print(sd.query_devices())
    print("Using samplerate:", FS, "channels:", N_CHANNELS, "device index:", DEVICE_INDEX)
    with sd.InputStream(device=DEVICE_INDEX,
                        channels=N_CHANNELS,
                        samplerate=FS,
                        dtype='float32',
                        blocksize=HOP_SAMPLES,
                        callback=callback):
        print("Listening… Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
