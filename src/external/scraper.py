from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ScrapeRecord:
    platform_shop_id: str
    source: str
    visibility_score: float | None  # 노출 순위 점수 (0~1)
    review_count: int | None
    avg_rating: float | None
    blog_mention_30d: int | None
    sns_mention_30d: int | None
    raw: dict | None = None


class ExternalDataScraper(ABC):
    """외부 데이터 수집기 추상 인터페이스.

    설계 원칙:
      - 실시간 호출 책임은 구현체가 가짐 (네이버/구글/SNS 등)
      - 본 클래스는 캐시 + rate limit + 표준 산출만 담당
      - robots.txt 및 ToS 준수는 구현체 책임
    """

    DEFAULT_CACHE_TTL_HOURS = 24
    DEFAULT_RATE_LIMIT_SECONDS = 1.5  # 매장당 최소 간격

    def __init__(self, cache_dir: Path | str = "output/external_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call: float = 0.0

    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def fetch_one(self, shop_name: str, address: str | None = None) -> ScrapeRecord: ...

    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)[:120]
        return self.cache_dir / f"{self.source_name()}__{safe}.json"

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        age_hours = (time.time() - p.stat().st_mtime) / 3600
        if age_hours > self.DEFAULT_CACHE_TTL_HOURS:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, key: str, payload: dict) -> None:
        self._cache_path(key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.DEFAULT_RATE_LIMIT_SECONDS:
            time.sleep(self.DEFAULT_RATE_LIMIT_SECONDS - elapsed)
        self._last_call = time.time()

    def fetch_batch(
        self,
        shops: pd.DataFrame,
        id_col: str = "platform_shop_id",
        name_col: str = "shop_name",
        addr_col: str = "shop_address",
        limit: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        records: list[ScrapeRecord] = []
        rows = shops.head(limit) if limit else shops
        for i, row in enumerate(rows.itertuples(index=False), 1):
            sid = getattr(row, id_col)
            name = getattr(row, name_col, "")
            addr = getattr(row, addr_col, None) if hasattr(row, addr_col) else None
            cache_key = f"{sid}__{name}"
            cached = self._read_cache(cache_key)
            if cached is not None:
                rec = ScrapeRecord(**cached)
            else:
                self._throttle()
                rec = self.fetch_one(name, addr)
                rec.platform_shop_id = sid
                self._write_cache(cache_key, asdict(rec))
            records.append(rec)
            if verbose and i % 50 == 0:
                print(f"  [{self.source_name()}] {i}/{len(rows)} 처리 완료")
        return pd.DataFrame([asdict(r) for r in records])
