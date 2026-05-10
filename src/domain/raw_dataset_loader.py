from __future__ import annotations

from ..config import Paths
from ..io import ColumnCleaner, ExcelLoader, JsonLoader
from .raw_dataset import RawDataset


class RawDatasetLoader:
    """경로에서 6개 raw 파일을 읽고 컬럼 정리/리네임까지 담당."""

    SHOP_RENAMES = {
        "shops_platform_shop_id": "platform_shop_id",
        "shop_name_(익명)": "shop_name",
        "shops_category_name": "category_name",
        "brand_shop_id_brands_display_name": "brand_name",
        "홍보문구_추가_여부": "promo_flag",
        "페르소나_말투_지정_여부_(user_persona___shops_user_id_=_user_persona_user_id)": "persona_flag",
        "구분": "group_type",
    }
    TREAT_RENAMES = {"service_term_agree_date_utc": "service_term_agree_date"}
    CONTROL_RENAMES = {
        "가게_ID": "shop_id_raw",
        "사업자번호": "business_number",
        "조회_일시_(KST)": "menu_check_time",
        "메뉴판_ID": "menu_board_id",
        "메뉴_상태": "menu_status",
        "카테고리명": "menu_category_name",
        "메뉴_이름": "menu_name",
        "메뉴_ID": "menu_id",
        "배달_금액": "delivery_price",
        "픽업_금액": "pickup_price",
    }

    def __init__(
        self,
        paths: Paths,
        json_loader: JsonLoader | None = None,
        excel_loader: ExcelLoader | None = None,
        cleaner: ColumnCleaner | None = None,
    ):
        self.paths = paths
        self.json = json_loader or JsonLoader()
        self.excel = excel_loader or ExcelLoader()
        self.cleaner = cleaner or ColumnCleaner()

    def load(self, verbose: bool = True) -> RawDataset:
        if verbose:
            for f in self.paths.all_input_files():
                print(f.name, "->", f.exists())

        ds = RawDataset(
            shop=self.json.load(self.paths.shop_file),
            treat=self.json.load(self.paths.treat_file),
            order=self.json.load(self.paths.order_file),
            review=self.json.load(self.paths.review_file),
            control=self.json.load(self.paths.control_file),
            address=self.excel.load(self.paths.address_file),
        )

        if verbose:
            print("\n[데이터 로드 완료]")
            for name, shape in ds.shapes().items():
                print(f"  {name}_df: {shape}")

        self._clean_and_rename(ds)
        self._normalize_control_shop_id(ds)
        return ds

    def _clean_and_rename(self, ds: RawDataset) -> None:
        ds.shop = self.cleaner.rename_if_exists(self.cleaner.clean(ds.shop), self.SHOP_RENAMES)
        ds.treat = self.cleaner.rename_if_exists(self.cleaner.clean(ds.treat), self.TREAT_RENAMES)
        ds.order = self.cleaner.clean(ds.order)
        ds.review = self.cleaner.clean(ds.review)
        ds.control = self.cleaner.rename_if_exists(self.cleaner.clean(ds.control), self.CONTROL_RENAMES)
        ds.address = self.cleaner.clean(ds.address)

    def _normalize_control_shop_id(self, ds: RawDataset) -> None:
        if "shop_id_raw" in ds.control.columns:
            ds.control["shop_id_raw"] = (
                ds.control["shop_id_raw"].astype(str).str.replace(".0", "", regex=False)
            )
            ds.control["platform_shop_id"] = "ba_" + ds.control["shop_id_raw"]
