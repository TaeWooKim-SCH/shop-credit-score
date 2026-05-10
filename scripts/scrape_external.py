"""외부 데이터 스크래핑 데모 (mock).

실행:
  python -m scripts.scrape_external --limit 50

산출:
  output/external_naver.csv  — 매장별 네이버 노출/리뷰 통계
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import Paths
from src.external import NaverPlaceScraper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="처리할 매장 수")
    parser.add_argument("--input", default="output/master_dataset.csv")
    parser.add_argument("--output", default="output/external_naver.csv")
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()

    master = pd.read_csv(args.input, low_memory=False)
    shops = (
        master[["platform_shop_id", "shop_name", "shop_address"]]
        .drop_duplicates("platform_shop_id")
        .head(args.limit)
        .copy()
    )
    print(f"대상 매장: {len(shops)}개 ({'mock' if args.mock else 'live'} 모드)")

    scraper = NaverPlaceScraper(mock=args.mock)
    enriched = scraper.fetch_batch(shops, limit=args.limit)
    enriched.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n저장: {args.output}")
    print(enriched.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
