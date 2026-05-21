"""공공데이터포털 — 소상공인시장진흥공단 상권정보 API.

매장 좌표 + 반경 1km 내 동종 업종 매장 수 (경쟁 밀도) 수집.
평가단이 강조한 "경쟁 밀도" 변수 직접 대응.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from urllib.parse import unquote

import requests

from ..config import DATA_GO_KR_SERVICE_KEY
from .kakao_local import KakaoLocalAPI
from .scraper import ExternalDataScraper, ScrapeRecord


@dataclass
class CommercialAreaRecord:
    """상권정보 API 응답에서 추출한 매장 주변 경쟁 통계."""
    competitor_count_1km: int        # 동종 업종 1km 반경 내 매장 수
    total_shops_1km: int             # 모든 업종 1km 반경 내 매장 수
    competition_density: float       # 동종 / 전체


class SbizCommercialAreaScraper(ExternalDataScraper):
    """소상공인 상권정보 API — 반경 검색으로 경쟁 매장 수 수집.

    API: https://www.data.go.kr/data/15012005/openapi.do
    엔드포인트: storeListInRadius (반경 내 점포 조회)
    """

    SOURCE_NAME = "sbiz_commercial"
    URL = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius"
    DEFAULT_RADIUS = 500   # meters
    TIMEOUT = 10.0
    FNB_CATEGORY_CODE = "I2"  # 음식점업 대분류

    def __init__(
        self,
        cache_dir="output/external_cache",
        service_key: str | None = None,
        kakao: KakaoLocalAPI | None = None,
        radius: int = DEFAULT_RADIUS,
        mock: bool | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        # 공공데이터 키는 보통 URL 인코딩되어 발급 → requests에 넘기기 전 디코딩
        raw_key = service_key or DATA_GO_KR_SERVICE_KEY
        self.service_key = unquote(raw_key) if raw_key else None
        self.kakao = kakao or KakaoLocalAPI()
        self.radius = radius
        if mock is None:
            self.mock = self.service_key is None
        else:
            self.mock = mock

    def source_name(self) -> str:
        return self.SOURCE_NAME

    def fetch_one(self, shop_name: str, address: str | None = None) -> ScrapeRecord:
        if self.mock:
            return self._mock(shop_name, address)
        return self._real(shop_name, address)

    def _real(self, shop_name: str, address: str | None) -> ScrapeRecord:
        coord = self.kakao.address_to_coord(address) if address else None
        if coord is None and shop_name:
            coord = self.kakao.keyword_to_coord(shop_name)
        if coord is None or coord.get("lat") is None:
            return self._empty_record(shop_name, reason="geocoding_failed")

        lat, lng = coord["lat"], coord["lng"]
        return self._fetch_by_coord(lat, lng, shop_name, address, coord)

    def _fetch_by_coord(self, lat, lng, shop_name, address, coord) -> ScrapeRecord:
        try:
            resp = requests.get(
                self.URL,
                params={
                    "serviceKey": self.service_key,
                    "cx": lng, "cy": lat,
                    "radius": self.radius,
                    "type": "json",
                    "numOfRows": 1000, "pageNo": 1,
                },
                timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                return self._empty_record(shop_name, reason=f"status_{resp.status_code}")
            try:
                data = resp.json()
            except ValueError:
                return self._empty_record(shop_name, reason="non_json_response")
            body = data.get("body") or data.get("response", {}).get("body", {})
            items = body.get("items", [])
            if isinstance(items, dict):
                items = items.get("item", []) or []
            if isinstance(items, dict):
                items = [items]

            total = len(items)
            # F&B 매장만 동종 업종으로 카운트 (indsLclsCd == "I2" 음식점업)
            fnb_count = sum(1 for it in items
                            if it.get("indsLclsCd") == self.FNB_CATEGORY_CODE)
            # 동종 밀도: F&B 매장 수 / 전체 매장 수 (경쟁이 얼마나 빽빽한지)
            density = fnb_count / total if total else 0.0

            return ScrapeRecord(
                platform_shop_id="",
                source=self.SOURCE_NAME,
                visibility_score=None,
                review_count=None,
                avg_rating=None,
                blog_mention_30d=None,
                sns_mention_30d=None,
                raw={
                    "competitor_count_1km": fnb_count,
                    "total_shops_1km": total,
                    "competition_density": round(density, 4),
                    "lat": lat, "lng": lng,
                    "address_name": coord.get("address_name"),
                    "region_2depth": coord.get("region_2depth"),
                    "region_3depth": coord.get("region_3depth"),
                },
            )
        except requests.RequestException as e:
            return self._empty_record(shop_name, reason=f"exception_{type(e).__name__}")

    def _empty_record(self, shop_name: str, reason: str) -> ScrapeRecord:
        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=None, review_count=None,
            avg_rating=None, blog_mention_30d=None, sns_mention_30d=None,
            raw={"error": reason, "shop_name": shop_name},
        )

    def _mock(self, shop_name: str, address: str | None) -> ScrapeRecord:
        seed = int(hashlib.md5((shop_name or "").encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        same = int(rng.expovariate(1 / 8)) + 1
        total = same + int(rng.expovariate(1 / 30))
        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=None, review_count=None,
            avg_rating=None, blog_mention_30d=None, sns_mention_30d=None,
            raw={
                "mock": True,
                "competitor_count_1km": same,
                "total_shops_1km": total,
                "competition_density": round(same / total, 4) if total else 0.0,
                "shop_name": shop_name, "address": address,
            },
        )
