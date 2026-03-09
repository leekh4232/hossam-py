# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import requests
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from importlib.metadata import distributions
import pandas as pd
import numpy as np
import glob as gl
from typing import Literal
from PIL import Image, ImageEnhance
# -------------------------------------------------------------
from pandas import DataFrame, DatetimeIndex, read_csv, read_excel
from scipy.stats import normaltest
from tabulate import tabulate
from os.path import join, exists
import os
from io import BytesIO
from pandas import DataFrame, read_csv, read_excel
from typing import Optional, Tuple, Any

BASE_URL = "https://data.hossam.kr"

# -------------------------------------------------------------
def __get_df(path: str, index_col=None) -> DataFrame:
    p = path.rfind(".")
    exec = path[p+1:].lower()
    tmp_dir = None

    # 파일 확장자가 압축파일인 경우 로컬에 파일을 다운로드 후 압축 해제
    if exec == "zip":
        tmp_dir = os.getcwd() + "/.hossam_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        zip_path = join(tmp_dir, "data.zip")

        # 원격 URL인 경우 파일 다운로드
        if path.lower().startswith(('http://', 'https://')):
            path = path.replace("\\", "/")
            with requests.Session() as session:
                r = session.get(path)

                if r.status_code != 200:
                    raise Exception(f"HTTP {r.status_code} Error - {r.reason} > {path}")

                with open(zip_path, "wb") as f:
                    f.write(r.content)
        else:
            zip_path = Path(path)

        # 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # 압축 해제된 파일 중 첫 번째 파일을 데이터로 로드
        extracted_files = list(gl.glob(join(tmp_dir, '*')))
        if not extracted_files:
            raise FileNotFoundError("압축 파일 내에 데이터 파일이 없습니다.")

        path = str(extracted_files[0])
        p = path.rfind(".")
        exec = path[p+1:].lower()

    if exec == 'xlsx':
        # If path is a remote URL, fetch the file once and reuse the bytes
        if path.lower().startswith(('http://', 'https://')):
            path = path.replace("\\", "/")
            with requests.Session() as session:
                r = session.get(path)

                if r.status_code != 200:
                    raise Exception(f"HTTP {r.status_code} Error - {r.reason} > {path}")

                data_bytes = r.content

            # Use separate BytesIO objects for each read to avoid pointer/stream issues
            df = read_excel(BytesIO(data_bytes), index_col=index_col)

            try:
                info = read_excel(BytesIO(data_bytes), sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                pretty_table(info)
                print()
            except Exception:
                #print(f"\033[91m[!] Cannot read metadata\033[0m")
                pass
        else:
            df = read_excel(path, index_col=index_col)

            try:
                info = read_excel(path, sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                pretty_table(info)
                print()
            except:
                #print(f"\033[91m[!] Cannot read metadata\033[0m")
                pass
    else:
        df = read_csv(path, index_col=index_col)

    if tmp_dir:
        shutil.rmtree(tmp_dir)

    return df

# -------------------------------------------------------------
def __get_data_url(key: str, local: str | None = None) -> Tuple[str, Any, Any]:
    global BASE_URL

    path = None

    if not local:
        data_path = join(BASE_URL, "metadata.json").replace("\\", "/")

        with requests.Session() as session:
            r = session.get(data_path)

            if r.status_code != 200:
                raise Exception("[%d Error] %s" % (r.status_code, r.reason))

        my_dict = r.json()
        info = my_dict.get(key.lower())

        if not info:
            raise FileNotFoundError("%s는 존재하지 않는 데이터에 대한 요청입니다." % key)

        path = join(BASE_URL, info['url'])
    else:
        data_path = join(local, "metadata.json")

        if not exists(data_path):
            raise FileNotFoundError("존재하지 않는 데이터에 대한 요청입니다.")

        with open(data_path, "r", encoding="utf-8") as f:
            my_dict = json.loads(f.read())

        info = my_dict.get(key.lower())
        path = join(local, info['url'])

    return path, info.get('desc'), info.get('index')

# -------------------------------------------------------------
def load_info(search: str | None = None, local: str | None = None) -> DataFrame:
    """메타데이터에서 사용 가능한 데이터셋 정보를 로드한다.

    Args:
        search (str, optional): 이름 필터 문자열. 포함하는 항목만 반환.
        local (str, optional): 로컬 메타데이터 경로. None이면 원격(BASE_URL) 사용.

    Returns:
        DataFrame: name, chapter, desc, url 컬럼을 갖는 테이블

    Examples:
        ```python
        from hossam import *
        info = load_info()
        list(info.columns) #['name', 'chapter', 'desc', 'url']
        ```
    """
    global BASE_URL

    path = None

    if not local:
        data_path = join(BASE_URL, "metadata.json").replace("\\", "/")

        with requests.Session() as session:
            r = session.get(data_path)

            if r.status_code != 200:
                raise Exception("[%d Error] %s ::: %s" % (r.status_code, r.reason, data_path))

        my_dict = r.json()
    else:
        data_path = join(local, "metadata.json")

        if not exists(data_path):
            raise FileNotFoundError("존재하지 않는 데이터에 대한 요청입니다.")

        with open(data_path, "r", encoding="utf-8") as f:
            my_dict = json.loads(f.read())

    my_data = []
    for key in my_dict:
        if 'index' in my_dict[key]:
            del my_dict[key]['index']

        my_dict[key]['url'] = "%s/%s" % (BASE_URL, my_dict[key]['url'])
        my_dict[key]['name'] = key

        if 'chapter' in my_dict[key]:
            my_dict[key]['chapter'] = ", ".join(my_dict[key]['chapter'])
        else:
            my_dict[key]['chapter'] = '공통'

        my_data.append(my_dict[key])

    my_df = DataFrame(my_data)
    my_df2 = my_df.reindex(columns=['name', 'chapter', 'desc', 'url'])

    if search:
        my_df2 = my_df2[my_df2['name'].str.contains(search.lower())]

    return my_df2

# -------------------------------------------------------------
def _load_data_remote(key: str, local: str | None = None) -> Optional[DataFrame]:
    """키로 지정된 데이터셋을 로드한다.

    Args:
        key (str): 메타데이터에 정의된 데이터 식별자(파일명 또는 별칭)
        local (str, optional): 로컬 메타데이터 경로. None이면 원격(BASE_URL) 사용.

    Returns:
        DataFrame | None: 성공 시 데이터프레임, 실패 시 None

    Examples:
        ```python
        from hossam import *
        df = load_data('AD_SALES')  # 메타데이터에 해당 키가 있어야 함
        ```
    """
    index = None
    try:
        url, desc, index = __get_data_url(key, local=local)
    except Exception as e:
        try:
            print(f"\033[91m{str(e)}\033[0m")
        except Exception:
            print(e)
        return

    #print("\033[94m[data]\033[0m", url.replace("\\", "/"))
    #print("\033[94m[desc]\033[0m", desc)
    print(f"\033[94m{desc}\033[0m")

    df = None

    try:
        df = __get_df(url, index_col=index)
    except Exception as e:
        try:
            print(f"\033[91m{str(e)}\033[0m")
        except Exception:
            print(e)
        return


    return df

# ===================================================================
# 설치된 파이썬 패키지 목록 반환
# ===================================================================
def my_packages():
    """
    현재 파이썬 인터프리터에 설치된 모든 패키지의 이름과 버전을
    패키지 이름순으로 정렬하여 pandas DataFrame으로 반환합니다.
    Returns:
        pd.DataFrame: columns=['name', 'version']
    """
    pkgs = []
    for dist in distributions():
        name = dist.metadata['Name'] if 'Name' in dist.metadata else dist.name
        version = dist.version
        summary = dist.metadata.get('Summary', '')
        pkgs.append((name, version, summary))
    pkgs = sorted(pkgs, key=lambda x: x[0].lower())
    return pd.DataFrame(pkgs, columns=['name', 'version', 'summary'])

# ===================================================================
# 정규분포 데이터 생성
# ===================================================================
def make_normalize_values(
    mean: float, std: float, size: int = 100, round: int = 2
) -> np.ndarray:
    """정규분포를 따르는 데이터를 생성한다.

    Args:
        mean (float): 평균
        std (float): 표준편차
        size (int, optional): 데이터 크기. Defaults to 100.
        round (int, optional): 소수점 반올림 자리수. Defaults to 2.

    Returns:
        np.ndarray: 정규분포를 따르는 데이터

    Examples:
        ```python
        from hossam import *
        x = hs.util.make_normalize_values(mean=0.0, std=1.0, size=100)
        ```
    """
    p = 0.0
    x: np.ndarray = np.array([])
    attempts = 0
    max_attempts = 100  # 무한 루프 방지
    while p < 0.05 and attempts < max_attempts:
        x = np.random.normal(mean, std, size).round(round)
        _, p = normaltest(x)
        attempts += 1

    return x


# ===================================================================
# 정규분포 데이터프레임 생성
# ===================================================================
def make_normalize_data(
    means: list | None = None,
    stds: list | None = None,
    sizes: list | None = None,
    rounds: int = 2,
) -> DataFrame:
    """정규분포를 따르는 데이터프레임을 생성한다.

    Args:
        means (list, optional): 평균 목록. Defaults to [0, 0, 0].
        stds (list, optional): 표준편차 목록. Defaults to [1, 1, 1].
        sizes (list, optional): 데이터 크기 목록. Defaults to [100, 100, 100].
        rounds (int, optional): 반올림 자리수. Defaults to 2.

    Returns:
        DataFrame: 정규분포를 따르는 데이터프레임
    """
    means = means if means is not None else [0, 0, 0]
    stds = stds if stds is not None else [1, 1, 1]
    sizes = sizes if sizes is not None else [100, 100, 100]

    if not (len(means) == len(stds) == len(sizes)):
        raise ValueError("means, stds, sizes 길이는 동일해야 합니다.")

    data = {}
    for i in range(len(means)):
        data[f"X{i+1}"] = make_normalize_values(
            means[i], stds[i], sizes[i], rounds
        )

    return DataFrame(data)


# ===================================================================
# DataFrame을 이쁘게 출력
# ===================================================================
def pretty_table(data: DataFrame, tablefmt="simple", headers: str = "keys") -> None:
    """`tabulate`를 사용해 DataFrame을 단순 표 형태로 출력한다.

    Args:
        data (DataFrame): 출력할 데이터프레임
        tablefmt (str, optional): `tabulate` 테이블 포맷. Defaults to "simple".
        headers (str | list, optional): 헤더 지정 방식. Defaults to "keys".

    Returns:
        None

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        hs_util.pretty_table(DataFrame({"a":[1,2],"b":[3,4]}))
        ```
    """

    tabulate.WIDE_CHARS_MODE = False # type: ignore
    print(
        tabulate(data, headers=headers, tablefmt=tablefmt, showindex=True, numalign="right") # type: ignore
    )


# ===================================================================
# 데이터 프레임을 통해 필요한 초기 작업을 수행
# ===================================================================
def __data_info(
    origin: DataFrame,
    index_col: str | None = None,
    timeindex: bool = False,
    info: bool = True,
    categories: list | None = None,
) -> DataFrame:
    """데이터 프레임을 통해 필요한 초기 작업을 수행한다.

    Args:
        origin (DataFrame): 원본 데이터 프레임
        index_col (str, optional): 인덱스 필드의 이름. Defaults to None.
        timeindex (bool, optional): True일 경우 인덱스를 시계열로 설정. Defaults to False.
        info (bool, optional): True일 경우 정보 출력. Defaults to True.
        categories (list, optional): 카테고리로 지정할 필드 목록. Defaults to None.

    Returns:
        DataFrame: 데이터프레임 객체
    """

    data = origin.copy()

    if index_col is not None and index_col in data.columns:
        data.set_index(index_col, inplace=True)

    if timeindex:
        data.index = DatetimeIndex(data.index)

    if categories:
        from .hs_prep import set_category  # type: ignore
        data = set_category(data, *categories)

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

        # 전달된 필드 이름 리스트가 있다면 반복
        if categories:
            print("\n🗂️ 카테고리 정보")
            for c in categories:
                d = DataFrame({"count": data[c].value_counts()})
                d.index.name = c
                pretty_table(d, tablefmt="pretty")

    return data


# ===================================================================
# 데이터 로드
# ===================================================================
def load_data(key: str,
                index_col: str | None = None,
                timeindex: bool = False,
                info: bool = True,
                categories: list | None = None,
                local: str | None = None) -> DataFrame:
    """데이터 키를 통해 데이터를 로드한 뒤 기본 전처리/출력을 수행한다.

    Args:
        key (str): 데이터 키 (metadata.json에 정의된 데이터 식별자)
        index_col (str, optional): 인덱스로 설정할 컬럼명. Defaults to None.
        timeindex (bool, optional): True일 경우 인덱스를 시계열(DatetimeIndex)로 설정한다. Defaults to False.
        info (bool, optional): True일 경우 데이터 정보(head, tail, 기술통계, 카테고리 정보)를 출력한다. Defaults to True.
        categories (list, optional): 카테고리 dtype으로 설정할 컬럼명 목록. Defaults to None.
        local (str, optional): 원격 데이터 대신 로컬 메타데이터 경로를 사용한다. Defaults to None.

    Returns:
        DataFrame: 전처리(인덱스 설정, 카테고리 변환)가 완료된 데이터프레임

    Examples:
        ```python
        from hossam import *
        df = hs_util.load_data("AD_SALES", index_col=None, timeindex=False, info=False)
        ```
    """

    k = key.lower()

    origin = None

    if k.endswith(".xlsx"):
        origin = read_excel(key)
    elif k.endswith(".csv"):
        origin = read_csv(key)
    else:
        origin = _load_data_remote(key, local) # type: ignore

    if origin is None:
        raise RuntimeError("Data loading failed: origin is None")

    return __data_info(origin, index_col, timeindex, info, categories)


# ===================================================================
# 2차원 리스트 여부 확인
# ===================================================================
def is_2d(x) -> bool:
    """
    주어진 객체가 2차원 리스트인지 확인합니다.

    Args:
        x: 확인할 객체

    Returns:
        bool: 객체가 2차원 리스트인 경우 True, 그렇지 않은 경우 False
    """
    return (
        isinstance(x, (list, tuple)) and
        len(x) > 0 and
        all(isinstance(i, (list, tuple)) for i in x)
    )

# ===================================================================
# 이미지 튜닝
# ===================================================================
def tune_image(
    img: Image,
    mode: Literal["RGB", "color", "L", "gray"] = "RGB",
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지를 튜닝한다.

    Args:
        img (Image): 이미지 객체
        mode (Literal['RGB', 'color', 'L', 'gray'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 튜닝된 이미지
    """
    if mode:
        if mode == "color":
            mode = "RGB"
        elif mode == "gray":
            mode = "L"

        img = img.convert(mode=mode)

    if size:
        w = size[0] if size[0] > 0 else 0
        h = size[1] if size[1] > 0 else 0
        img = img.resize(size=(w, h))

    if color:
        if color < 0:
            color = 0
        img = ImageEnhance.Color(image=img).enhance(factor=color)

    if contrast:
        img = ImageEnhance.Contrast(image=img).enhance(
            factor=contrast if contrast > 0 else 0
        )

    if brightness:
        img = ImageEnhance.Brightness(image=img).enhance(
            factor=brightness if brightness > 0 else 0
        )

    if sharpness:
        img = ImageEnhance.Sharpness(image=img).enhance(
            factor=sharpness if sharpness > 0 else 0
        )

    img.array = np.array(img)

    return img


# ===================================================================
# 이미지 로드 + 튜닝
# ===================================================================
def load_image(
    path: str,
    mode: Literal["RGB", "L"] = None,
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지 파일을 로드한다. 필요한 경우 로드한 이미지에 대해 튜닝을 수행한다. 최종 로드된 이미지에 대한 배열 데이터를 array 속성에 저장한다.

    Args:
        path (str): 이미지 파일 경로
        mode (Literal['RGB', 'color', 'L', 'gray'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 로드된 이미지
    """
    img = Image.open(fp=path)
    img = tune_image(
        img=img,
        mode=mode,
        size=size,
        color=color,
        contrast=contrast,
        brightness=brightness,
        sharpness=sharpness,
    )

    return img