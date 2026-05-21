from __future__ import annotations

import hashlib
import random
import re

import requests

from ..config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
from .scraper import ExternalDataScraper, ScrapeRecord


class NaverPlaceScraper(ExternalDataScraper):
    """네이버 검색 API (지역 / 블로그)로 매장 평판 신호 수집.

    수집 항목:
      - visibility_score: 검색 결과에 매장이 등장하는지 (0/1) — 추후 ranking으로 확장 가능
      - review_count    : 지역 검색 결과의 매장 정보 (있으면 1, 없으면 0)  — Local API v1은 review_count 직접 제공 안 함
      - avg_rating      : 동일 (API 한계)
      - blog_mention_30d: 매장명으로 블로그 검색 → total 결과 수
      - sns_mention_30d: 카페 검색 → total 결과 수

    Local API: 일 25K, 검색 API (Blog/Cafe): 일 25K.
    매장 1,200개 × (1 local + 1 blog + 1 cafe) = 3,600 호출 → 한도 내.
    """

    SOURCE_NAME = "naver_place"
    LOCAL_URL = "https://openapi.naver.com/v1/search/local.json"
    BLOG_URL = "https://openapi.naver.com/v1/search/blog.json"
    CAFE_URL = "https://openapi.naver.com/v1/search/cafearticle.json"
    TIMEOUT = 5.0
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    def __init__(
        self,
        cache_dir="output/external_cache",
        client_id: str | None = None,
        client_secret: str | None = None,
        mock: bool | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.client_id = client_id or NAVER_CLIENT_ID
        self.client_secret = client_secret or NAVER_CLIENT_SECRET
        if mock is None:
            self.mock = self.client_id is None or self.client_secret is None
        else:
            self.mock = mock

    def source_name(self) -> str:
        return self.SOURCE_NAME

    def fetch_one(self, shop_name: str, address: str | None = None) -> ScrapeRecord:
        if self.mock:
            return self._mock(shop_name, address)
        return self._real(shop_name, address)

    # ── 실제 호출 ─────────────────────────────────────────────
    def _headers(self) -> dict:
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    def _safe_get(self, url: str, query: str, display: int = 5) -> dict:
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                params={"query": query, "display": display},
                timeout=self.TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"_error": resp.status_code, "_text": resp.text[:200]}
        except requests.RequestException as e:
            return {"_error": "exception", "_text": str(e)[:200]}

    @staticmethod
    def _fuzzy_match(shop_name: str, title: str) -> bool:
        """공백 제거 + 핵심 점포명(마지막 2-4글자) 비교로 매칭."""
        if not shop_name or not title:
            return False
        s = shop_name.replace(" ", "")
        t = title.replace(" ", "")
        if s in t or t in s:
            return True
        # 점포명 추출: 매장명 마지막 "XX점" 패턴
        m = re.search(r"([가-힣A-Za-z0-9]{2,8}점)$", s)
        if m and m.group(1) in t:
            return True
        # 토큰 겹침 50% 이상
        s_tokens = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", shop_name))
        t_tokens = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", title))
        if s_tokens and t_tokens:
            overlap = len(s_tokens & t_tokens) / min(len(s_tokens), len(t_tokens))
            return overlap >= 0.5
        return False

    def _real(self, shop_name: str, address: str | None) -> ScrapeRecord:
        query = shop_name or ""
        # 지역 검색
        local = self._safe_get(self.LOCAL_URL, query, display=5)
        local_items = local.get("items", []) or []
        clean_title = lambda s: self._HTML_TAG_RE.sub("", s or "")
        found = False
        local_rank = None
        for i, item in enumerate(local_items, 1):
            title = clean_title(item.get("title", ""))
            if self._fuzzy_match(shop_name, title):
                found = True
                local_rank = i
                break
        # 노출 점수 = 1 / 순위 (못 찾으면 0)
        visibility = (1.0 / local_rank) if local_rank else 0.0

        # 블로그 검색
        blog = self._safe_get(self.BLOG_URL, query, display=1)
        blog_total = int(blog.get("total", 0) or 0)

        # 카페 검색
        cafe = self._safe_get(self.CAFE_URL, query, display=1)
        cafe_total = int(cafe.get("total", 0) or 0)

        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=round(visibility, 4),
            review_count=len(local_items),
            avg_rating=None,  # 지역 검색 API는 별점 미제공
            blog_mention_30d=blog_total,
            sns_mention_30d=cafe_total,
            raw={
                "found_in_local": found,
                "local_rank": local_rank,
                "first_title": clean_title(local_items[0]["title"]) if local_items else None,
                "address_input": address,
            },
        )

    # ── Mock ─────────────────────────────────────────────────
    def _mock(self, shop_name: str, address: str | None) -> ScrapeRecord:
        seed = int(hashlib.md5((shop_name or "").encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=round(rng.random(), 4),
            review_count=int(rng.expovariate(1 / 80)),
            avg_rating=round(rng.uniform(3.5, 5.0), 2),
            blog_mention_30d=int(rng.expovariate(1 / 10)),
            sns_mention_30d=int(rng.expovariate(1 / 5)),
            raw={"mock": True, "shop_name": shop_name, "address": address},
        )
