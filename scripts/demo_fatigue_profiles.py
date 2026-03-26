from __future__ import annotations

from datetime import datetime

from backend.state.feature_extractor import FeatureExtractor
from backend.state.state_model import StateModel


QUESTION = "Limitte neden paydada eslenik ile sadelestirme yapiyoruz?"

DEMO_PROFILES = [
    {
        "profile": "normal_odakta",
        "message": (
            "Bu soruda once paydada eslenik ile sadelestirme denedim. "
            "Sonra limitin 2 oldugunu buldum ama son adimi kontrol etmek istiyorum."
        ),
    },
    {
        "profile": "yorgun_ogrenci",
        "message": (
            "Ayni soruda yoruldum, kafam almiyor. "
            "Paydada neden eslenik kullandigimizi su an anlayamiyorum."
        ),
    },
    {
        "profile": "frustre_ogrenci",
        "message": (
            "Of ya, ayni soruda yine takildim. "
            "Bu cok sinir bozucu, neden hala paydada eslenik kullandigimizi cozemiyorum."
        ),
    },
    {
        "profile": "kendinden_emin_ogrenci",
        "message": (
            "Bence cozdum. Paydada eslenik kullaninca limitin 2 ciktigindan eminim, "
            "istersen sonucu challenge ederek kontrol edelim."
        ),
    },
    {
        "profile": "bunalmis_ve_acele_ogrenci",
        "message": (
            "Bunaldim, konu ust uste geliyor ve sinavim var. "
            "Hizlica, kisa ve net bir ozetle paydada neden eslenik kullandigimizi anlat."
        ),
    },
]


def main() -> None:
    extractor = FeatureExtractor()
    model = StateModel()
    now = datetime(2026, 3, 27, 10, 0, 0)

    print("Demo question:")
    print(QUESTION)
    print()

    for item in DEMO_PROFILES:
        feature = extractor.extract(
            session_id=f"demo-{item['profile']}",
            message_content=item["message"],
            message_timestamp=now,
        )
        estimate = model.predict(feature)

        print(f"[{item['profile']}]")
        print(item["message"])
        print(
            "state=", estimate.state.value,
            "| policy=", estimate.response_policy.value,
            "| fatigue=", feature.fatigue_text_score,
            "| frustration=", feature.frustration_text_score,
            "| confidence=", feature.confidence_text_score,
            "| overwhelm=", feature.overwhelm_text_score,
            "| urgency=", feature.urgency_text_score,
        )
        print("dominant_signals=", estimate.dominant_signals)
        print("reasons=", estimate.reasons)
        print("probabilities=", estimate.state_probabilities)
        print()


if __name__ == "__main__":
    main()
