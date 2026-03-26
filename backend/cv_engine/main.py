"""
FocusAI — Ana Döngü
CVPipeline'ı başlatır, her frame'i çizer ve özet istatistik basar.
Orijinal monolitik script'in draw() + ana döngü mantığını
cv_engine paketi mimarisiyle çalıştırır.

Çalıştır:
    python main.py
Çıkmak için Q tuşuna bas.
"""

import time
from collections import Counter
import numpy as np
import cv2

from cv_engine import CVPipeline, CameraSignal


# --------------------------------------------------------------------------
# Renk / etiket sabitler
# --------------------------------------------------------------------------

STATE_COLOR = {
    "FOCUSED":    (0, 200, 0),
    "DISTRACTED": (0, 165, 255),
    "SLEEPY":     (0, 0, 220),
    "UNKNOWN":    (128, 128, 128),
}
STATE_LABEL = {
    "FOCUSED":    "ODAKLI",
    "DISTRACTED": "DAGINK",
    "SLEEPY":     "UYKULU",
    "UNKNOWN":    "YUZ YOK",
}

SCRATCH_WINDOW = 120   # GestureExtractor ile senkron olmalı


# --------------------------------------------------------------------------
# Çizim
# --------------------------------------------------------------------------

def draw(frame: np.ndarray, signal: CameraSignal) -> np.ndarray:
    f = frame.copy()
    h, w = f.shape[:2]

    state = signal.state
    score = signal.attention_score
    gaze  = signal.gaze
    gesture = signal.gesture

    color = STATE_COLOR.get(state, (128, 128, 128))
    label = STATE_LABEL.get(state, state)

    # Yarı saydam başlık şeridi
    ov = f.copy()
    cv2.rectangle(ov, (0, 0), (w, 125), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, f, 0.45, 0, f)

    # Skor + durum etiketi
    score_text = f"Dikkat: {score:.2f}" if score is not None else "Yuz yok"
    cv2.putText(f, score_text, (10, 38),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(f, label,      (w - 180, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if gaze:
        cv2.putText(
            f,
            f"EAR:{gaze['ear_avg']:.3f}  Bakis:{gaze['gaze_direction']}  MAR:{gaze['mar']:.3f}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
        if gaze.get("is_drowsy"):
            cv2.putText(f, "UYUKLAMA!", (10, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if gaze.get("is_blinking"):
            cv2.putText(f, "KIRPIYOR",  (200, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 1)
        if gaze.get("is_yawning"):
            cv2.putText(
                f,
                f"ESNIYOR ({gaze.get('yawn_count', 0)}x)",
                (350, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2,
            )

    if gesture:
        cv2.putText(
            f,
            f"Bas:{gesture['head_pitch']:+.0f}p {gesture['head_yaw']:+.0f}y",
            (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1,
        )
        if gesture.get("is_head_down"):
            cv2.putText(f, "BAS ASAGI", (w - 195, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        if gesture.get("is_head_turned"):
            cv2.putText(f, "BAS YANA",  (w - 185, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        if gesture.get("hand_on_chin"):
            cv2.putText(f, "EL CENEDE", (w - 195, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        if gesture.get("hand_on_mouth") and not (gaze and gaze.get("is_yawning")):
            cv2.putText(f, "EL AGIZDA", (w - 195, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
        if gesture.get("hand_scratching"):
            cv2.putText(f, "KASIYOR",   (10, 143),       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        sc = gesture.get("scratch_count", 0)
        if sc > 0:
            sp  = gesture.get("scratch_penalty", 0.0)
            col = (0, 200, 255) if sp == 0 else (0, 80, 255)
            cv2.putText(
                f,
                f"KASIMA:{sc}x/{SCRATCH_WINDOW}s  ceza:{sp:.2f}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
            )

    # Alt progress bar
    if score is not None:
        cv2.rectangle(f, (0, h - 14), (w, h), (40, 40, 40), -1)
        cv2.rectangle(f, (0, h - 14), (int(w * score), h), color, -1)

    return f


# --------------------------------------------------------------------------
# Ana döngü
# --------------------------------------------------------------------------

def main():
    pipeline = CVPipeline(camera_index=0, target_fps=15, use_emotion=False)
    ok = pipeline.start()
    if not ok:
        print("Pipeline başlatılamadı. Yine de pencere açılıyor (UNKNOWN modu).")

    print("📷 Başlıyor... Çıkmak için Q tuşuna bas.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    scores_log = []
    state_log  = []
    yawn_log   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera okunamadı!")
            break

        signal    = pipeline.latest()
        annotated = draw(frame, signal)
        cv2.imshow("FocusAI", annotated)

        # Log
        scores_log.append(signal.attention_score if signal.attention_score is not None else 0.0)
        state_log.append(signal.state)
        yawn_log.append(1 if (signal.gaze and signal.gaze.get("is_yawning")) else 0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()

    # ── Özet ─────────────────────────────────────────────────────────────
    if scores_log:
        s = np.array(scores_log)
        yawn_total = sum(yawn_log)
        print(f"\n✅ Bitti! {len(scores_log)} frame işlendi.")
        print(f"Ortalama dikkat : {s.mean():.3f}")
        print(f"Dikkatli (≥0.7) : %{(s >= 0.7).mean() * 100:.1f}")
        print(f"Dağınık  (<0.4) : %{(s < 0.4).mean() * 100:.1f}")
        print(f"Toplam esneme   : {yawn_total} kez")
        sc = Counter(state_log)
        print("Durum dağılımı  : " + " | ".join(f"{k}:{v}" for k, v in sc.items()))


if __name__ == "__main__":
    main()
