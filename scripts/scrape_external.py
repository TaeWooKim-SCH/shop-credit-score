"""외부 데이터 스크래핑 — 네이버 + 상권정보 + SGIS 통합 수집.

실행:
  # 전체 소스, 처음 50개
  python -m scripts.scrape_external --limit 50 --source all

  # 네이버만 전수
  python -m scripts.scrape_external --source naver

산출:
  output/external_naver.csv   — 매장별 검색 노출/평판
  output/external_sbiz.csv    — 매장별 경쟁 밀도
  output/external_sgis.csv    — 매장별 입지 인구 통계

이후 DataEngineer가 이 파일들을 master에 병합.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import Paths
from src.external import (
    NaverPlaceScraper,
    SGISStatScraper,
    SbizCommercialAreaScraper,
)

SCRAPER_MAP = {
    "naver": (NaverPlaceScraper, "external_naver.csv"),
    "sbiz": (SbizCommercialAreaScraper, "external_sbiz.csv"),
    "sgis": (SGISStatScraper, "external_sgis.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="all",
        choices=["all", "naver", "sbiz", "sgis"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--input", default="output/master_dataset.csv")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="강제 mock 모드 (API 키 무시)")
    args = parser.parse_args()

    master = pd.read_csv(args.input, low_memory=False)
    shops = (
        master[["platform_shop_id", "shop_name", "shop_address"]]
        .drop_duplicates("platform_shop_id")
        .head(args.limit)
        .copy()
    )
    print(f"대상 매장: {len(shops)}개")

    sources = ["naver", "sbiz", "sgis"] if args.source == "all" else [args.source]

    for src in sources:
        cls, out_file = SCRAPER_MAP[src]
        print(f"\n=== {src.upper()} 수집 시작 ===")
        scraper = cls(mock=args.mock if args.mock else None)
        print(f"  mock={scraper.mock}")
        enriched = scraper.fetch_batch(shops, limit=args.limit)

        # raw 컬럼은 dict라 CSV 저장 시 json 문자열로 변환
        if "raw" in enriched.columns:
            enriched["raw"] = enriched["raw"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x
            )
        # raw에서 추가 컬럼 풀어내기 (sbiz, sgis 같은 dict-only scraper용)
        enriched = _expand_raw_columns(enriched, src)

        out_path = Paths().output_dir / out_file
        enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {out_path}  shape={enriched.shape}")
        print(enriched.head(3).to_string(index=False, max_colwidth=60))


def _expand_raw_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """raw dict에서 source별 핵심 변수를 flat 컬럼으로."""
    if "raw" not in df.columns:
        return df
    parsed = df["raw"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("{") else (x or {})
    )

    if source == "sbiz":
        df["competitor_count_1km"] = parsed.apply(lambda d: d.get("competitor_count_1km"))
        df["total_shops_1km"] = parsed.apply(lambda d: d.get("total_shops_1km"))
        df["competition_density"] = parsed.apply(lambda d: d.get("competition_density"))
    elif source == "sgis":
        df["tot_ppltn"] = parsed.apply(lambda d: d.get("tot_ppltn"))
        df["avg_age"] = parsed.apply(lambda d: d.get("avg_age"))
        df["tot_household"] = parsed.apply(lambda d: d.get("tot_household"))
        df["adm_cd"] = parsed.apply(lambda d: d.get("adm_cd"))

    return df


if __name__ == "__main__":
    main()
