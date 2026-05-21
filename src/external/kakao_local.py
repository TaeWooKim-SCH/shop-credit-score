"""카카오 로컬 API — 매장 주소 → 좌표 변환 (geocoding).

다른 scraper(SGIS, 상권정보)가 좌표 또는 행정동 코드를 요구하므로
주소 → (위도, 경도, 행정동) 변환 게이트웨이 역할.
"""
from __future__ import annotations

import requests

from ..config import KAKAO_REST_API_KEY


class KakaoLocalAPI:
    URL_ADDRESS = "https://dapi.kakao.com/v2/local/search/address.json"
    URL_KEYWORD = "https://dapi.kakao.com/v2/local/search/keyword.json"
    URL_REGION = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    TIMEOUT = 5.0

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or KAKAO_REST_API_KEY
        self.available = bool(self.api_key)

    def _headers(self) -> dict:
        return {"Authorization": f"KakaoAK {self.api_key}"}

    def address_to_coord(self, address: str) -> dict | None:
        if not self.available or not address:
            return None
        try:
            resp = requests.get(
                self.URL_ADDRESS, headers=self._headers(),
                params={"query": address}, timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            docs = resp.json().get("documents", [])
            if not docs:
                return None
            d = docs[0]
            return {
                "lat": float(d.get("y", 0)) or None,
                "lng": float(d.get("x", 0)) or None,
                "address_name": d.get("address_name"),
                "region_1depth": d.get("address", {}).get("region_1depth_name"),
                "region_2depth": d.get("address", {}).get("region_2depth_name"),
                "region_3depth": d.get("address", {}).get("region_3depth_name"),
            }
        except requests.RequestException:
            return None

    def keyword_to_coord(self, query: str) -> dict | None:
        """주소로 못 찾을 때 매장명으로 검색해 좌표 보완."""
        if not self.available or not query:
            return None
        try:
            resp = requests.get(
                self.URL_KEYWORD, headers=self._headers(),
                params={"query": query, "size": 1}, timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            docs = resp.json().get("documents", [])
            if not docs:
                return None
            d = docs[0]
            return {
                "lat": float(d.get("y", 0)) or None,
                "lng": float(d.get("x", 0)) or None,
                "category": d.get("category_name"),
                "place_name": d.get("place_name"),
            }
        except requests.RequestException:
            return None

    def coord_to_region(self, lat: float, lng: float) -> dict | None:
        """좌표 → 행정동 코드 (SGIS API용)."""
        if not self.available or lat is None or lng is None:
            return None
        try:
            resp = requests.get(
                self.URL_REGION, headers=self._headers(),
                params={"x": lng, "y": lat}, timeout=self.TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            docs = resp.json().get("documents", [])
            for d in docs:
                if d.get("region_type") == "H":
                    return {
                        "region_code": d.get("code"),
                        "region_name": d.get("address_name"),
                        "region_3depth": d.get("region_3depth_name"),
                    }
            return None
        except requests.RequestException:
            return None
