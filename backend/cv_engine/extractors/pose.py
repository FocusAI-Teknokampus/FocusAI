"""
CV Engine / Extractors — Baş Pozu (Bağımsız modül)
GestureExtractor'dan bağımsız olarak sadece baş açısı lazımsa kullanılır.
Genellikle gesture.py yeterlidir; bu dosya ayrı ihtiyaç için ayrıldı.
"""

# Baş pozu hesaplaması gesture.py içinde _head_pose() fonksiyonunda.
# Bu modül gelecekte DeepFace veya başka bir pose kütüphanesi
# entegre edilmek istenirse buraya yazılacak.

# Şimdilik gesture.py'ı yeniden export et
from .gesture import GestureExtractor as PoseExtractor  # noqa: F401
