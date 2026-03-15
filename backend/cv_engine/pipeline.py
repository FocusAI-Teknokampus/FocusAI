# backend/cv_engine/pipeline.py
import cv2
from datetime import datetime
from backend.core.schemas import FrameData
from backend.cv_engine.buffer import SlidingWindowBuffer
from backend.cv_engine.extractors import gaze, pose, emotion, gesture

class CVPipeline:
    def __init__(self, camera_id: int = 0, fps: int = 5, window_size: int = 5):
        self.camera_id = camera_id
        self.buffer = SlidingWindowBuffer(window_size_sec=window_size, fps=fps)
        self.delay_between_frames = int(1000 / fps) # 5 FPS için 200ms bekleme
        
    def start(self):
        """Kamerayı açar ve sonsuz frame yakalama döngüsünü başlatır."""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("Hata: Kamera açılamadı!")
            return

        print("Dikkat Analiz Kamerası Başlatıldı. Çıkmak için 'q' tuşuna basın.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Kameradan gelen görüntüyü analiz modüllerine gönderiyoruz
            # (İleride bu fonksiyonların içini MediaPipe ve DeepFace ile dolduracağız)
            is_looking = gaze.check_gaze_on_screen(frame)
            ear_value = gaze.calculate_ear(frame)
            is_bored = emotion.check_boredom(frame)
            is_hand_on_chin = gesture.check_hand_on_chin(frame)

            # 1. Adım: Çıkarılan bu özellikleri Pydantic şemamıza (Sözleşmeye) uyarlıyoruz
            current_data = FrameData(
                timestamp=datetime.now(),
                gaze_on_screen=is_looking,
                hand_on_chin=is_hand_on_chin,
                emotion_bored=is_bored,
                ear_score=ear_value
            )

            # 2. Adım: Veriyi Sliding Window (Kayan Pencere) tamponuna ekliyoruz
            self.buffer.add_frame(current_data)

            # 3. Adım: Eğer 5 saniyelik tampon dolduysa, skorlama motoruna gönder
            if self.buffer.is_ready():
                window_data = self.buffer.get_window_data()
                
                # Sadece test amaçlı: Buffer'daki son karedeki verileri ekrana yazdırıyoruz
                son_kare = window_data[-1] 
                print("\n--- 5 SANİYELİK BLOK DOLDU ---")
                print(f"Göz Açıklığı (EAR): {son_kare.ear_score:.3f}")
                print(f"El Çenede mi?: {son_kare.hand_on_chin}")
                print(f"Sıkılmış mı?: {son_kare.emotion_bored}")
                print("------------------------------")

            # Geliştirici için kamerayı ekranda göster (Debug amaçlı)
            cv2.imshow('EduFocus AI - Kamera', frame)
            
            # FPS ayarı ve 'q' ile çıkış
            if cv2.waitKey(self.delay_between_frames) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Sadece bu dosyayı test etmek istersek:
if __name__ == "__main__":
    pipeline = CVPipeline(fps=5)
    pipeline.start()