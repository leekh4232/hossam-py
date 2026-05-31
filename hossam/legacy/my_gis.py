# -*- coding: utf-8 -*-
# ===================================================================
# 패키지 참조
# ===================================================================
import os
import time
import warnings
import requests
import concurrent.futures as futures

from pandas import DataFrame, to_numeric
from tqdm.auto import tqdm
from geopandas import GeoDataFrame, read_file, points_from_xy
from pyproj import CRS

from .my_util import pretty_table

# ===================================================================
# 단일 주소를 VWorld API로 지오코딩
# ===================================================================
def __geocode_item(session: requests.Session, index: int, addr: str, key: str) -> tuple[float, float]:
    """단일 주소를 VWorld API로 지오코딩합니다.

    Args:
        session (requests.Session): 재사용할 `requests.Session` 인스턴스.
        index (int): 입력 데이터의 인덱스(로그용).
        addr (str): 지오코딩할 도로명 주소 문자열.
        key (str): VWorld API 키.

    Returns:
        (latitude, longitude) 튜플.

    Raises:
        ValueError: 주소가 비어있거나 잘못된 경우.
        requests.exceptions.RequestException: 주소를 찾지 못한 경우 등 요청 관련 오류.
        Exception: HTTP 오류 코드나 API 내부 오류 등 기타 예외.
    """
    if not addr or addr == "nan":
        raise ValueError(
            "⚠️[Warning] 주소가 존재하지 않습니다. (%d) -> %s" % (index, addr)
        )

    url: str = f"https://api.vworld.kr/req/address"
    params = {
        "service": "address",
        "request": "getCoord",
        "key": key,
        "address": addr,
        "type": "ROAD",
        "format": "json",
    }

    response = None

    try:
        response = session.get(url, params=params, timeout=(3, 30))
    except Exception as e:
        raise e

    if response.status_code != 200:
        raise Exception(
            "⚠️[%d-Error] %s - API 요청에 실패했습니다. (%d) -> %s"
            % (response.status_code, response.reason, index, addr)
        )

    response.encoding = "utf-8"
    result = response.json()
    status = result["response"]["status"]

    if status == "ERROR":
        error_code = result["response"]["error"]["code"]
        error_text = result["response"]["error"]["text"]
        raise Exception(f"[{error_code}] {error_text} (%d) -> %s" % (index, addr))
    elif status == "NOT_FOUND":
        raise requests.exceptions.RequestException(
            "⚠️[Warning] 주소를 찾을 수 없습니다. (%d) -> %s" % (index, addr)
        )

    longitude = float(result["response"]["result"]["point"]["x"])
    latitude = float(result["response"]["result"]["point"]["y"])
    result = (latitude, longitude)
    #print("%s --> (%s, %s)" % (addr, latitude, longitude))
    return result


# ===================================================================
# 주소 컬럼을 일괄 지오코딩하여 위도/경도 컬럼을 추가
# ===================================================================
def geocode(df: DataFrame, addr: str, key: str) -> DataFrame:
    """주소 컬럼을 일괄 지오코딩하여 위도/경도 컬럼을 추가합니다.

    Args:
        df (DataFrame): 입력 `DataFrame`.
        addr (str): 주소가 들어있는 컬럼명.
        key (str): VWorld API 키.

    Returns:
        DataFrame: 위도(`latitude`), 경도(`longitude`) 컬럼이 추가된 `DataFrame`.

    Raises:
        Exception: 지오코딩 과정에서 발생한 예외를 전파합니다.

    Examples:
        ```python
        from hossam import *
        result = my_gis.geocode(df, addr="address", key="YOUR_VWORLD_KEY")
        set(["latitude","longitude"]).issubset(result.columns)
        # True
        ```
    """
    data: DataFrame = df.copy()
    size: int = len(data)
    success = 0
    fail = 0

    print("ℹ️요청 데이터 개수: %d" % size)

    with tqdm(total=size, colour="yellow") as pbar:
        with requests.Session() as session:
            with futures.ThreadPoolExecutor(max_workers=30) as executor:
                for i in range(size):
                    time.sleep(0.1)
                    address: str = str(data.loc[i, addr]).strip()

                    p = executor.submit(
                        __geocode_item, session, index=i, addr=address, key=key
                    )

                    try:
                        result = p.result()
                        latitude, longitude = result
                        data.loc[i, "latitude"] = latitude
                        data.loc[i, "longitude"] = longitude
                        success += 1
                    except requests.exceptions.RequestException as re:
                        print(re)
                        data.loc[i, "latitude"] = None
                        data.loc[i, "longitude"] = None
                        fail += 1
                    except ValueError as ve:
                        print(ve)
                        data.loc[i, "latitude"] = None
                        data.loc[i, "longitude"] = None
                        fail += 1
                    except Exception as e:
                        fail += 1
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e
                    finally:
                        pbar.set_postfix({"success": success, "fail": fail})
                        pbar.update(1)

    data["latitude"] = data["latitude"].astype(float)
    data["longitude"] = data["longitude"].astype(float)

    print(f"✅총 {size}개의 데이터 중 {success}개의 데이터가 처리되었습니다.")

    return data

# ===================================================================
# Shapefile을 읽어 `GeoDataFrame`으로 로드
# ===================================================================
def load_shape(path: str, info: bool = True) -> GeoDataFrame:
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

    if info:
        print("\n✅ 테이블 정보")
        pretty_table(data.info(), tablefmt="pretty") # type: ignore

        print("\n✅ 상위 5개 행")
        pretty_table(data.head(), tablefmt="pretty")

        print("\n✅ 하위 5개 행")
        pretty_table(data.tail(), tablefmt="pretty")

        print("\n📊 기술통계")
        desc = data.describe().T
        desc["nan"] = data.isnull().sum()
        pretty_table(desc, tablefmt="pretty")

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
      - 좌표가 유효하지 않은 행은 제외되며, 유효한 좌표가 하나도 없으면 예외를 발생시킵니다.

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

    # 저장 (경고 메시지 억제)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gdf.to_file(path, driver=driver, encoding="utf-8")
    print(f"✅ {file_format} 저장 완료: {path} (CRS: {target_crs.to_string()})")
