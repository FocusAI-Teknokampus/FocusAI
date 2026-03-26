"""
Mem0 local hafizasini gelistirici tarafinda incelemek icin kucuk yardimci script.

Kullanim:
    python scripts/inspect_mem0.py --user-id user_001
    python scripts/inspect_mem0.py --user-id user_001 --limit 20

Bu script UI'a bagli degildir.
Sadece terminalde hangi memory kayitlarinin tutuldugunu gormek icin kullanilir.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from backend.memory.long_term import LongTermMemory


def _normalize_results(payload: Any) -> list[dict]:
    """
    Mem0 farkli surumlerde farkli sekillerde veri donebilir.
    Bu yardimci metod listeyi tek formata toplar.
    """
    if isinstance(payload, dict):
        results = payload.get("results", [])
        return results if isinstance(results, list) else []
    if isinstance(payload, list):
        return payload
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mem0 local hafizasindaki kayitlari terminalde gosterir."
    )
    parser.add_argument("--user-id", required=True, help="Incelenecek kullanici kimligi")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="En fazla kac memory kaydi gosterilsin",
    )
    args = parser.parse_args()

    memory = LongTermMemory()
    if not memory._available or memory.client is None:
        print("Mem0 kullanilabilir degil. .env ve bagimliliklari kontrol et.")
        return

    # Mem0'dan ham kayitlari cekiyoruz.
    # Burada profil ozetine degil, dogrudan saklanan memory metinlerine bakmak istiyoruz.
    payload = memory.client.get_all(user_id=args.user_id)
    results = _normalize_results(payload)

    print(f"Kullanici: {args.user_id}")
    print(f"Toplam memory kaydi: {len(results)}")

    for index, item in enumerate(results[: args.limit], start=1):
        print(f"\n--- Memory {index} ---")
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
