# backend/cv_engine/pipeline.py
import cv2
from datetime import datetime

from backend.core.schemas import FrameData
from backend.cv_engine.buffer import SlidingWindowBuffer
from backend.cv_engine.extractors import gaze, emotion, gesture
from backend.agents.chain import AgentCoordinator

# Karar Motorumuzu (Scorer) içeri aktarıyoruz
from backend.scorer.scorer import AttentionScorer

class CVPipeline:
    def __init__(self, camera_id: int = 0, fps: int = 5, window_size: int = 5):
        self.camera_id = camera_id
        self.fps = fps
        self.buffer = SlidingWindowBuffer(window_size_sec=window_size, fps=fps)
        self.scorer = AttentionScorer(fps=fps) # Skorlayıcıyı başlattık
        self.delay_between_frames = int(1000 / fps)
        self.coordinator = AgentCoordinator()

        self.frame_count = 0 # Ne zaman rapor vereceğimizi bilmek için sayaç
        
    def start(self):
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("Hata: Kamera açılamadı!")
            return

        print("Dikkat Analiz Kamerası Başlatıldı. Çıkmak için 'q' tuşuna basın.")
        print("İlk analiz için 5 saniye bekleniyor...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Kameradan verileri çek (Extractors)
            is_looking = gaze.check_gaze_on_screen(frame)
            ear_value = gaze.calculate_ear(frame)
            is_bored = emotion.check_boredom(frame)
            
            # Not: Eğer gesture.py dosyasını tamamen doldurmadıysanız 
            # burası hata verebilir. Eğer hata verirse aşağıdaki satırı:
            # is_hand_on_chin = False 
            # şeklinde değiştirebilirsiniz.
            try:
                is_hand_on_chin = gesture.check_hand_on_chin(frame)
            except Exception:
                is_hand_on_chin = False

            # 2. Veriyi Sözleşmeye (Schema) uygun hale getir
            current_data = FrameData(
                timestamp=datetime.now(),
                gaze_on_screen=is_looking,
                hand_on_chin=is_hand_on_chin,
                emotion_bored=is_bored,
                ear_score=ear_value
            )

            # 3. Hafızaya (Buffer) ekle
            self.buffer.add_frame(current_data)
            self.frame_count += 1

            # 4. SKORLAMA ZAMANI: Buffer doluysa VE tam 5 saniyenin (25 karenin) sonundaysak
            if self.buffer.is_ready() and self.frame_count % self.buffer.max_frames == 0:
                window_data = self.buffer.get_window_data()
                
                # Beyin (Scorer) devreye giriyor!
                result = self.scorer.compute_score(window_data)
                
                # Çıktıyı firmaların seveceği profesyonel bir terminal arayüzüyle yazdırıyoruz
                print("\n" + "="*40)
                print(f"📊 DİKKAT ANALİZ RAPORU (Son 5 Saniye)")
                print("="*40)
                print(f"🎯 DİKKAT SKORU: {result['score']:.0f} / 100")
                
                if result['score'] < 80.0:
                    print("⚠️  Sorunlar:")
                    for r in result['reasons']:
                        print(f"   - {r}")
                        
                    print("\n🤖 LANGCHAIN AJANLARI DEVREYE GİRİYOR...")
                    
                    # Veriyi AgentCoordinator (chain.py) dosyasına gönderiyoruz
                    agent_result = self.coordinator.process_low_attention(
                        score=result['score'], 
                        reasons=result['reasons']
                    )
                    
                    print(f"💡 Pedagojik Öneri: {agent_result['action_advice']}")
                else:

                    print("✅  Durum: Öğrenci pür dikkat derste!")

            # Geliştirici ekranı
            cv2.imshow('EduFocus AI - Kamera', frame)
            
            if cv2.waitKey(self.delay_between_frames) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = CVPipeline(fps=5)
    pipeline.start()