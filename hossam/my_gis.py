# -*- coding: utf-8 -*-

# ===================================================================
# 패키지 참조
# ===================================================================
import os
import time
import threading
import warnings
import requests
import concurrent.futures as futures

from pandas import DataFrame, to_numeric
from tqdm.auto import tqdm
from geopandas import GeoDataFrame, read_file, points_from_xy
from pyproj import CRS

from .my_util import pretty_table


# ===================================================================
# 지오코딩 전용 예외
# ===================================================================
class AddressNotFoundError(Exception):
    """주소를 찾을 수 없는 경우(소프트 실패). 해당 행만 건너뛰고 계속 진행합니다."""


class GeocodingError(Exception):
    """지오코딩 중 발생한 HTTP/API 오류.

    Attributes:
        fatal (bool): True면 배치 전체를 중단해야 하는 치명적 오류(예: 인증 키 오류).
    """

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


# ===================================================================
# 스레드 안전 레이트 리미터 (요청 시작 간 최소 간격 보장)
# ===================================================================
class _RateLimiter:
    """여러 스레드에서 호출돼도 요청 시작 사이의 최소 간격을 보장합니다."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return

        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
                self._last = time.monotonic()
            else:
                self._last = now


# ===================================================================
# 단일 주소를 VWorld API로 지오코딩
# ===================================================================
def __geocode_item(
    session: requests.Session,
    index: int,
    addr: str,
    key: str,
    limiter: "_RateLimiter | None" = None,
    max_retries: int = 3,
) -> tuple[float, float]:
    """단일 주소를 VWorld API로 지오코딩합니다.

    Args:
        session (requests.Session): 재사용할 `requests.Session` 인스턴스.
        index (int): 입력 데이터의 인덱스(로그용).
        addr (str): 지오코딩할 도로명 주소 문자열.
        key (str): VWorld API 키.
        limiter (_RateLimiter | None): 요청 속도를 제어할 레이트 리미터(선택).
        max_retries (int): 일시적 네트워크 오류 시 재시도 횟수.

    Returns:
        (latitude, longitude) 튜플.

    Raises:
        ValueError: 주소가 비어있거나 잘못된 경우.
        AddressNotFoundError: 주소를 찾지 못한 경우(소프트 실패).
        GeocodingError: HTTP 오류 코드나 API 내부 오류, 재시도 후에도 실패한 네트워크 오류.
    """
    if not addr or addr == "nan":
        raise ValueError(
            "⚠️[Warning] 주소가 존재하지 않습니다. (%d) -> %s" % (index, addr)
        )

    url: str = "https://api.vworld.kr/req/address"
    params = {
        "service": "address",
        "request": "getCoord",
        "key": key,
        "address": addr,
        "type": "ROAD",
        "format": "json",
    }

    # 일시적 네트워크 오류(ConnectionError/Timeout)는 지수 백오프로 재시도
    response = None
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        if limiter is not None:
            limiter.wait()

        try:
            response = session.get(url, params=params, timeout=(3, 30))
            break
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(0.5 * (2**attempt))
    else:
        raise GeocodingError(
            "⚠️[NetworkError] 네트워크 오류로 요청에 실패했습니다. (%d) -> %s : %s"
            % (index, addr, last_exc)
        )

    if response.status_code != 200:
        raise GeocodingError(
            "⚠️[%d-Error] %s - API 요청에 실패했습니다. (%d) -> %s"
            % (response.status_code, response.reason, index, addr)
        )

    response.encoding = "utf-8"
    result = response.json()
    status = result["response"]["status"]

    if status == "ERROR":
        error_code = result["response"]["error"]["code"]
        error_text = result["response"]["error"]["text"]
        # 인증 키 관련 오류는 배치 전체를 중단해야 하는 치명적 오류로 분류
        fatal = "KEY" in str(error_code).upper()
        raise GeocodingError(
            f"[{error_code}] {error_text} (%d) -> %s" % (index, addr), fatal=fatal
        )
    elif status == "NOT_FOUND":
        raise AddressNotFoundError(
            "⚠️[Warning] 주소를 찾을 수 없습니다. (%d) -> %s" % (index, addr)
        )

    longitude = float(result["response"]["result"]["point"]["x"])
    latitude = float(result["response"]["result"]["point"]["y"])
    return (latitude, longitude)


# ===================================================================
# 주소 컬럼을 일괄 지오코딩하여 위도/경도 컬럼을 추가
# ===================================================================
def geocode(
    df: DataFrame,
    addr: str,
    key: str,
    max_workers: int = 8,
    requests_per_second: float = 10.0,
) -> DataFrame:
    """주소 컬럼을 일괄 지오코딩하여 위도/경도 컬럼을 추가합니다.

    여러 요청을 스레드 풀로 병렬 처리하며, `requests_per_second`로 호출 속도를 제어합니다.
    개별 주소의 실패(주소 없음/찾지 못함/일반 API 오류)는 해당 행만 결측 처리한 뒤
    계속 진행하고, 인증 키 오류 같은 치명적 오류만 배치 전체를 중단합니다.

    Args:
        df (DataFrame): 입력 `DataFrame`.
        addr (str): 주소가 들어있는 컬럼명.
        key (str): VWorld API 키.
        max_workers (int): 동시 요청 스레드 수.
        requests_per_second (float): 초당 최대 요청 수(0 이하면 제한 없음).

    Returns:
        DataFrame: 위도(`latitude`), 경도(`longitude`) 컬럼이 추가된 `DataFrame`.
            입력 인덱스는 0부터 시작하도록 재설정됩니다.

    Raises:
        GeocodingError: 치명적 오류(예: 잘못된 API 키)가 발생한 경우.

    Examples:
        ```python
        from hossam import *
        result = my_gis.geocode(df, addr="address", key="YOUR_VWORLD_KEY")
        set(["latitude","longitude"]).issubset(result.columns)
        # True
        ```
    """
    # 비연속/비정수 인덱스에서도 .loc 접근이 안전하도록 인덱스를 재설정
    data: DataFrame = df.reset_index(drop=True)
    size: int = len(data)
    success = 0
    fail = 0

    data["latitude"] = None
    data["longitude"] = None

    min_interval = 1.0 / requests_per_second if requests_per_second and requests_per_second > 0 else 0.0
    limiter = _RateLimiter(min_interval)

    print("ℹ️요청 데이터 개수: %d" % size)

    with tqdm(total=size, colour="yellow") as pbar:
        with requests.Session() as session:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 작업을 먼저 제출한 뒤 완료되는 순서대로 결과를 수거(실제 병렬 처리)
                future_to_idx = {
                    executor.submit(
                        __geocode_item,
                        session,
                        i,
                        str(data.loc[i, addr]).strip(),
                        key,
                        limiter,
                    ): i
                    for i in range(size)
                }

                for future in futures.as_completed(future_to_idx):
                    i = future_to_idx[future]

                    try:
                        latitude, longitude = future.result()
                        data.loc[i, "latitude"] = latitude
                        data.loc[i, "longitude"] = longitude
                        success += 1
                    except (AddressNotFoundError, ValueError) as e:
                        print(e)
                        fail += 1
                    except GeocodingError as e:
                        fail += 1
                        if e.fatal:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        print(e)
                    finally:
                        pbar.set_postfix({"success": success, "fail": fail})
                        pbar.update(1)

    data["latitude"] = to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = to_numeric(data["longitude"], errors="coerce")

    print(
        f"✅총 {size}개의 데이터 중 성공 {success}개 / 실패 {fail}개 처리되었습니다."
    )

    return data


# ===================================================================
# Shapefile을 읽어 `GeoDataFrame`으로 로드
# ===================================================================
def load_shape(path: str) -> GeoDataFrame:
    """Shapefile을 읽어 `GeoDataFrame`으로 로드합니다.

    Args:
        path (str): 읽을 Shapefile(.shp) 경로.
        info (bool): True면 데이터 프리뷰와 통계를 출력.

    Returns:
        GeoDataFrame: 로드된 `GeoDataFrame`.

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우.

    Examples:
        ```python
        from hossam import *
        gdf = my_gis.load_shape("path/to/file.shp", info=False)
        ```
    """
    if not os.path.exists(path):
        raise FileNotFoundError("⚠️[FileNotFoundException] 주어진 파일을 찾을 수 없습니다.\n - %s" % path)

    data = read_file(path)

    return data


# ===================================================================
# 전처리된 데이터(GeoDataFrame 또는 DataFrame)를 Shapefile 또는 GeoPackage로 저장
# ===================================================================
def save_shape(
    gdf: GeoDataFrame | DataFrame,
    path: str,
    crs: str | None = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> None:
    """전처리된 데이터(GeoDataFrame 또는 DataFrame)를 Shapefile 또는 GeoPackage로 저장합니다.

    - GeoDataFrame 입력:
      - CRS가 있으면 그대로 유지합니다.
      - CRS가 없으면 `crs`(기본 WGS84)를 지정합니다.
    - DataFrame 입력:
      - 오직 이 경우에만 `lat_col`, `lon_col`을 사용해 포인트 지오메트리를 생성합니다.
      - 좌표가 유효하지 않거나 범위를 벗어난 행은 제외되며, 유효한 좌표가
        하나도 없으면 예외를 발생시킵니다.

    파일 형식:
      - .shp: ESRI Shapefile (필드명 10자 제한, ASCII 권장)
      - .gpkg: GeoPackage (필드명 제약 없음, 한글 가능)
      - 확장자 없으면 .shp로 저장

    Args:
        gdf (GeoDataFrame | DataFrame): 저장할 `GeoDataFrame` 또는 `DataFrame`.
        path (str): 저장 경로(.shp 또는 .gpkg, 확장자 없으면 .shp 자동 추가).
        crs (str | None): 좌표계 문자열(e.g., "EPSG:4326"). 미지정 시 WGS84.
        lat_col (str): DataFrame 입력 시 위도 컬럼명.
        lon_col (str): DataFrame 입력 시 경도 컬럼명.
    Returns:
        None: 파일을 저장하고 반환값이 없습니다.

    Raises:
        TypeError: 입력 타입이 잘못된 경우.
        ValueError: 경로가 잘못되었거나 CRS가 유효하지 않은 경우,
            또는 DataFrame에서 유효 좌표가 하나도 없는 경우.
    """
    if gdf is None or not isinstance(gdf, (GeoDataFrame, DataFrame)):
        raise TypeError("⚠️[TypeError] GeoDataFrame 또는 DataFrame 타입의 데이터가 필요합니다.")

    if not path or not isinstance(path, str):
        raise ValueError("⚠️[ValueError] 저장할 경로(path)가 올바르지 않습니다.")

    # 기본 좌표계를 WGS84로 설정
    crs_input = crs if crs and str(crs).strip() else "EPSG:4326"

    try:
        target_crs = CRS.from_user_input(crs_input)
    except Exception as e:
        raise ValueError(f"⚠️[ValueError] 유효하지 않은 좌표계 값입니다: {crs_input}") from e

    # DataFrame인 경우 위경도 컬럼으로 포인트 지오메트리 생성
    if isinstance(gdf, DataFrame) and not isinstance(gdf, GeoDataFrame):
        if lat_col not in gdf.columns or lon_col not in gdf.columns:
            raise ValueError(
                f"⚠️[ValueError] DataFrame에 '{lat_col}', '{lon_col}' 컬럼이 필요합니다."
            )

        df = gdf.copy()
        # 숫자 변환 및 결측 제거
        df[lat_col] = to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])

        # 좌표 범위 검증(위도 -90~90, 경도 -180~180): 컬럼 매핑 실수를 조기에 차단
        df = df[df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)]

        if df.empty:
            raise ValueError(
                "⚠️[ValueError] 유효한 위경도 값이 없어 Shapefile을 생성할 수 없습니다."
            )

        geometry = points_from_xy(x=df[lon_col], y=df[lat_col])
        gdf = GeoDataFrame(df, geometry=geometry, crs=target_crs)
    else:
        # GeoDataFrame의 CRS 처리: 존재하면 유지, 없으면만 설정
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)

    # 디렉터리 생성 보장
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # 확장자에 따라 드라이버 선택
    path_lower = path.lower()
    if path_lower.endswith(".gpkg"):
        driver = "GPKG"
        file_format = "GeoPackage"
    elif path_lower.endswith(".shp"):
        driver = "ESRI Shapefile"
        file_format = "Shapefile"
    else:
        # 확장자 없으면 .shp로 저장
        path = f"{path}.shp"
        driver = "ESRI Shapefile"
        file_format = "Shapefile"

    # 저장 (경고 메시지 억제). encoding 인자는 Shapefile에만 유효
    save_kwargs = {"driver": driver}
    if driver == "ESRI Shapefile":
        save_kwargs["encoding"] = "utf-8"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gdf.to_file(path, **save_kwargs)
    print(f"✅ {file_format} 저장 완료: {path} (CRS: {target_crs.to_string()})")
