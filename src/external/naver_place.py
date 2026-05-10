from __future__ import annotations

import hashlib
import random

from .scraper import ExternalDataScraper, ScrapeRecord


class NaverPlaceScraper(ExternalDataScraper):
    """네이버 플레이스 노출/평판 수집기.

    ⚠️ 실제 운영 시 주의:
      - 네이버 검색 API 또는 공식 Place API 사용 권장
      - 무단 스크래핑은 robots.txt + ToS 위반 가능
      - 본 클래스는 인터페이스 데모용. mock 모드로 결정론적 가짜 데이터 반환

    실제 구현 시 fetch_one() 안에서:
      1) requests.get("https://openapi.naver.com/v1/search/local.json", ...)
      2) headers={"X-Naver-Client-Id": ..., "X-Naver-Client-Secret": ...}
      3) JSON 파싱 → 매장명/주소 fuzzy match → 첫 결과의 ranking/review 추출
    """

    SOURCE_NAME = "naver_place"

    def __init__(
        self,
        cache_dir="output/external_cache",
        client_id: str | None = None,
        client_secret: str | None = None,
        mock: bool = True,
    ):
        super().__init__(cache_dir=cache_dir)
        self.client_id = client_id
        self.client_secret = client_secret
        self.mock = mock or (client_id is None or client_secret is None)

    def source_name(self) -> str:
        return self.SOURCE_NAME

    def fetch_one(self, shop_name: str, address: str | None = None) -> ScrapeRecord:
        if self.mock:
            return self._mock(shop_name, address)
        return self._real(shop_name, address)

    def _mock(self, shop_name: str, address: str | None) -> ScrapeRecord:
        """결정론적 가짜 데이터 — shop_name 해시 기반 시드로 안정 재현."""
        seed = int(hashlib.md5((shop_name or "").encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        return ScrapeRecord(
            platform_shop_id="",  # batch 단계에서 채움
            source=self.SOURCE_NAME,
            visibility_score=round(rng.random(), 4),
            review_count=int(rng.expovariate(1 / 80)),
            avg_rating=round(rng.uniform(3.5, 5.0), 2),
            blog_mention_30d=int(rng.expovariate(1 / 10)),
            sns_mention_30d=int(rng.expovariate(1 / 5)),
            raw={"mock": True, "shop_name": shop_name, "address": address},
        )

    def _real(self, shop_name: str, address: str | None) -> ScrapeRecord:
        # 실제 구현 자리. 네이버 검색 API v1 local 엔드포인트.
        # 캡스톤 발표 시 실제 호출 시연을 위해 구현 필요.
        raise NotImplementedError(
            "실제 호출은 client_id/secret 발급 후 구현하세요. "
            "현재는 mock=True로 사용해주세요."
        )
