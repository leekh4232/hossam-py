# -*- coding: utf-8 -*-
# -------------------------------------------------------------
from typing import TYPE_CHECKING
from importlib.metadata import distributions
import pandas as pd
import numpy as np
from pandas import DataFrame, DatetimeIndex, read_csv, read_excel
from scipy.stats import normaltest
from tabulate import tabulate

from .data_loader import load_data as _load_data_remote

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
