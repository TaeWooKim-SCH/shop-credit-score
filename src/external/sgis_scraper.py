"""SGIS Open API — 매장 주소 → 시도/시군구 인구통계 매핑.

공식 문서: https://sgis.mods.go.kr/developer/html/newOpenApi/api/dataApi/basics.html

SGIS는 자체 행정코드 체계를 사용 (11=서울, 38=경남 ...). 카카오 행정코드와 다름.
→ 시도/시군구 매핑 캐시를 한 번 구축한 뒤 매장 region 이름으로 lookup.
"""
from __future__ import annotations

import hashlib
import random
import re

import requests

from ..config import SGIS_CONSUMER_KEY, SGIS_CONSUMER_SECRET
from .kakao_local import KakaoLocalAPI
from .scraper import ExternalDataScraper, ScrapeRecord


# 카카오 region_1depth ↔ SGIS 시도명 정규화 (둘 다 비교용)
_PROVINCE_ALIAS = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시",
    "경기": "경기도", "강원": "강원특별자치도",
    "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도",
    "제주": "제주특별자치도",
}


def _norm_province(name: str) -> str:
    """카카오/SGIS 시도명을 정규화 (둘 다 같은 형태로 비교)."""
    if not name:
        return ""
    name = name.strip()
    # "경남" → "경상남도"
    for k, v in _PROVINCE_ALIAS.items():
        if name.startswith(k):
            return v
    return name


def _norm_sigungu(name: str) -> str:
    """시군구명 정규화 — 공백·이름 통일."""
    if not name:
        return ""
    return name.strip().replace(" ", "")


class SGISStatScraper(ExternalDataScraper):
    """SGIS 시도+시군구 단위 인구통계 수집기."""

    SOURCE_NAME = "sgis_population"
    AUTH_URL = "https://sgisapi.mods.go.kr/OpenAPI3/auth/authentication.json"
    POPULATION_URL = "https://sgisapi.mods.go.kr/OpenAPI3/stats/population.json"
    TIMEOUT = 8.0
    DEFAULT_YEAR = 2023

    def __init__(
        self,
        cache_dir="output/external_cache",
        consumer_key: str | None = None,
        consumer_secret: str | None = None,
        kakao: KakaoLocalAPI | None = None,
        mock: bool | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.consumer_key = consumer_key or SGIS_CONSUMER_KEY
        self.consumer_secret = consumer_secret or SGIS_CONSUMER_SECRET
        self.kakao = kakao or KakaoLocalAPI()
        self._token: str | None = None
        # 매핑 캐시: { 시도_정규화명: { 시군구_정규화명: stats_dict } }
        self._lookup: dict[str, dict[str, dict]] | None = None
        self._lookup_top: dict[str, dict] = {}  # 시도 단위 fallback
        if mock is None:
            self.mock = self.consumer_key is None or self.consumer_secret is None
        else:
            self.mock = mock

    def source_name(self) -> str:
        return self.SOURCE_NAME

    # ── 토큰 ──────────────────────────────────────────────────
    def _get_token(self) -> str | None:
        if self._token:
            return self._token
        try:
            resp = requests.get(
                self.AUTH_URL,
                params={
                    "consumer_key": self.consumer_key,
                    "consumer_secret": self.consumer_secret,
                },
                timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            self._token = (resp.json().get("result") or {}).get("accessToken")
            return self._token
        except requests.RequestException:
            return None

    # ── 매핑 캐시 구축 (한 번만) ──────────────────────────────
    def _ensure_lookup(self) -> None:
        if self._lookup is not None:
            return
        token = self._get_token()
        if token is None:
            self._lookup = {}
            return

        lookup: dict[str, dict[str, dict]] = {}
        # 1) 시도 단위 (low_search=1)
        try:
            r = requests.get(
                self.POPULATION_URL,
                params={"accessToken": token, "year": self.DEFAULT_YEAR, "low_search": 1},
                timeout=self.TIMEOUT,
            ).json()
            for row in r.get("result", []):
                top_name = _norm_province(row.get("adm_nm", ""))
                self._lookup_top[top_name] = row
                lookup[top_name] = {}
                # 2) 시군구 단위
                adm = row.get("adm_cd")
                if not adm:
                    continue
                try:
                    r2 = requests.get(
                        self.POPULATION_URL,
                        params={"accessToken": token, "year": self.DEFAULT_YEAR,
                                "adm_cd": adm, "low_search": 1},
                        timeout=self.TIMEOUT,
                    ).json()
                    for sub in r2.get("result", []):
                        sub_name = _norm_sigungu(sub.get("adm_nm", ""))
                        lookup[top_name][sub_name] = sub
                except requests.RequestException:
                    continue
        except requests.RequestException:
            pass
        self._lookup = lookup

    # ── 매장별 호출 ────────────────────────────────────────────
    def fetch_one(self, shop_name: str, address: str | None = None) -> ScrapeRecord:
        if self.mock:
            return self._mock(shop_name, address)
        return self._real(shop_name, address)

    def _real(self, shop_name: str, address: str | None) -> ScrapeRecord:
        self._ensure_lookup()
        if not self._lookup:
            return self._empty(shop_name, "lookup_init_failed")

        # 1) 주소에서 시도/시군구 이름 추출 (카카오 또는 정규식)
        prov, sig = self._extract_region(address)
        if not prov:
            return self._empty(shop_name, "region_parse_failed")

        # 2) 시도 매칭
        prov_norm = _norm_province(prov)
        prov_data = self._lookup.get(prov_norm)
        if prov_data is None:
            return self._empty(shop_name, f"unknown_province_{prov}")

        # 3) 시군구 매칭 (있으면 시군구, 없으면 시도 단위 폴백)
        stats = None
        used_level = "province"
        if sig:
            sig_norm = _norm_sigungu(sig)
            # 정확 매칭
            stats = prov_data.get(sig_norm)
            # 부분 매칭 ("창원시 의창구" → "창원시"로 시작하는 SGIS 시군구 찾기)
            if stats is None:
                main = re.match(r"([가-힣]+시|[가-힣]+군|[가-힣]+구)", sig)
                main_token = main.group(1) if main else sig.split()[0]
                for sname, sdata in prov_data.items():
                    if main_token and main_token in sname:
                        stats = sdata
                        used_level = "sigungu"
                        break
            else:
                used_level = "sigungu"

        # 4) 시도 단위 폴백
        if stats is None:
            stats = self._lookup_top.get(prov_norm)
            used_level = "province"

        if not stats:
            return self._empty(shop_name, "no_match")

        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=None, review_count=None,
            avg_rating=None, blog_mention_30d=None, sns_mention_30d=None,
            raw={
                "tot_ppltn": int(float(stats.get("tot_ppltn", 0) or 0)),
                "avg_age": float(stats.get("avg_age", 0) or 0),
                "tot_household": int(float(stats.get("tot_household", 0) or 0)),
                "adm_cd": stats.get("adm_cd"),
                "adm_nm": stats.get("adm_nm"),
                "matched_level": used_level,
            },
        )

    def _extract_region(self, address: str | None) -> tuple[str, str]:
        """주소에서 시도(prov)와 시군구(sig)를 추출.

        예: "경상남도 창원시 의창구 사림로 53" → ("경상남도", "창원시 의창구")
            "경남 창원시 의창구 사림로 53"     → ("경남",     "창원시 의창구")
        """
        if not address:
            return ("", "")
        tokens = address.strip().split()
        if not tokens:
            return ("", "")
        prov = tokens[0]
        # 시군구: 두 번째 토큰이 "OO시"이면 다음 토큰이 "OO구"인지 확인
        sig = ""
        if len(tokens) >= 2:
            if len(tokens) >= 3 and (tokens[2].endswith("구") or tokens[2].endswith("군")):
                sig = f"{tokens[1]} {tokens[2]}"
            else:
                sig = tokens[1]
        return (prov, sig)

    def _empty(self, shop_name: str, reason: str) -> ScrapeRecord:
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
        return ScrapeRecord(
            platform_shop_id="",
            source=self.SOURCE_NAME,
            visibility_score=None, review_count=None,
            avg_rating=None, blog_mention_30d=None, sns_mention_30d=None,
            raw={
                "mock": True,
                "tot_ppltn": int(rng.gauss(50000, 20000)),
                "avg_age": round(rng.uniform(35, 50), 1),
                "tot_household": int(rng.gauss(20000, 8000)),
                "shop_name": shop_name, "address": address,
            },
        )
