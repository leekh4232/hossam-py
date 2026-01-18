# -*- coding: utf-8 -*-
# ===================================================================
# 패키지 참조
# ===================================================================
import joblib
import numpy as np
from itertools import combinations
from typing import Any

import pandas as pd
import jenkspy
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


from .hs_util import pretty_table

# ===================================================================
# 연속형 변수를 표준정규화(Z-score)로 변환한다
# ===================================================================
def standard_scaler(
    data: Any, yname: str | None = None, save_path: str | None = None, load_path: str | None = None
) -> DataFrame:
    """연속형 변수에 대해 Standard Scaling을 수행한다.

    - DataFrame 입력 시: 비수치형/종속변수를 분리한 후 스케일링하고 다시 합칩니다.
    - 배열 입력 시: 그대로 스케일링된 ndarray를 반환합니다.
    - `load_path`가 주어지면 기존 스케일러를 재사용하고, `save_path`가 주어지면 학습된 스케일러를 저장합니다.

    Args:
        data (DataFrame | ndarray): 스케일링할 데이터.
        yname (str | None): 종속변수 컬럼명. 분리하지 않으려면 None.
        save_path (str | None): 학습된 스케일러 저장 경로.
        load_path (str | None): 기존 스케일러 로드 경로.

    Returns:
        DataFrame | ndarray: 스케일링된 데이터(입력 타입과 동일).

    Examples:
        ```python
        from hossam import *
        std_df = hs_prep.standard_scaler(df, yname="y", save_path="std.pkl")
        ```
    """

    is_df = isinstance(data, DataFrame)

    # ndarray 처리 분기
    if not is_df:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        scaler = joblib.load(load_path) if load_path else StandardScaler()
        sdata = scaler.transform(arr) if load_path else scaler.fit_transform(arr)
        if save_path:
            joblib.dump(value=scaler, filename=save_path)
        return sdata # type: ignore

    df = data.copy()

    y = None
    if yname and yname in df.columns:
        y = df[yname]
        df = df.drop(columns=[yname])

    category_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cate = df[category_cols] if category_cols else DataFrame(index=df.index)
    X = df.drop(columns=category_cols)

    if X.shape[1] == 0:
        return data

    scaler = joblib.load(load_path) if load_path else StandardScaler()
    sdata = scaler.transform(X) if load_path else scaler.fit_transform(X)

    if save_path:
        joblib.dump(value=scaler, filename=save_path)

    std_df = DataFrame(data=sdata, index=data.index, columns=X.columns)

    if category_cols:
        std_df[category_cols] = cate
    if yname and y is not None:
        std_df[yname] = y

    return std_df


# ===================================================================
# 연속형 변수를 0부터 1 사이의 값으로 정규화한다
# ===================================================================
def minmax_scaler(
    data: Any, yname: str | None = None, save_path: str | None = None, load_path: str | None = None
) -> DataFrame:
    """연속형 변수에 대해 MinMax Scaling을 수행한다.

    DataFrame은 비수치/종속변수를 분리 후 스케일링하고 재결합하며, 배열 입력은 그대로 ndarray를 반환한다.
    `load_path` 제공 시 기존 스케일러를 사용하고, `save_path` 제공 시 학습 스케일러를 저장한다.

    Args:
        data (DataFrame | ndarray): 스케일링할 데이터.
        yname (str | None): 종속변수 컬럼명. 분리하지 않으려면 None.
        save_path (str | None): 학습된 스케일러 저장 경로.
        load_path (str | None): 기존 스케일러 로드 경로.

    Returns:
        DataFrame | ndarray: 스케일링된 데이터(입력 타입과 동일).

    Examples:
        ```python
        from hossam import *
        mm_df = hs_prep.minmax_scaler(df, yname="y")
        ```
    """

    is_df = isinstance(data, DataFrame)

    if not is_df:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        scaler = joblib.load(load_path) if load_path else MinMaxScaler()
        sdata = scaler.transform(arr) if load_path else scaler.fit_transform(arr)
        if save_path:
            joblib.dump(scaler, save_path)
        return sdata # type: ignore

    df = data.copy()

    y = None
    if yname and yname in df.columns:
        y = df[yname]
        df = df.drop(columns=[yname])

    category_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cate = df[category_cols] if category_cols else DataFrame(index=df.index)
    X = df.drop(columns=category_cols)

    if X.shape[1] == 0:
        return data

    scaler = joblib.load(load_path) if load_path else MinMaxScaler()
    sdata = scaler.transform(X) if load_path else scaler.fit_transform(X)

    if save_path:
        joblib.dump(scaler, save_path)

    std_df = DataFrame(data=sdata, index=data.index, columns=X.columns)

    if category_cols:
        std_df[category_cols] = cate
    if yname and y is not None:
        std_df[yname] = y

    return std_df


# ===================================================================
# 지정된 컬럼들을 범주형 데이터로 설정한다
# ===================================================================
def set_category(data: DataFrame, *args: str, columns: list | None = None) -> DataFrame:
    """카테고리 데이터를 설정한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        *args (str): 컬럼명 목록
        columns (list, optional): 변환할 컬럼명 목록. args와 중복 사용 불가.

    Returns:
        DataFrame: 카테고리 설정된 데이터프레임
    """
    # columns 인자가 있으면 args보다 우선한다.
    if columns is not None:
        if args:
            raise ValueError("args와 columns 인자는 중복 사용할 수 없습니다.")
        args = columns # type: ignore

    df = data.copy()

    for k in args:
        df[k] = df[k].astype("category")

    return df


# ===================================================================
# 명목형 변수의 값 종류에 따른 데이터 분리
# ===================================================================
def unmelt(
    data: DataFrame, id_vars: str = "class", value_vars: str = "values"
) -> DataFrame:
    """두 개의 컬럼으로 구성된 데이터프레임에서 하나는 명목형, 나머지는 연속형일 경우
    명목형 변수의 값에 따라 고유한 변수를 갖는 데이터프레임으로 변환한다.

    각 그룹의 데이터 길이가 다를 경우 짧은 쪽에 NaN을 채워 동일한 길이로 맞춥니다.
    이는 독립표본 t-검정(ttest_ind) 등의 분석을 위한 데이터 준비에 유용합니다.

    Args:
        data (DataFrame): 데이터프레임
        id_vars (str, optional): 명목형 변수의 컬럼명. Defaults to 'class'.
        value_vars (str, optional): 연속형 변수의 컬럼명. Defaults to 'values'.

    Returns:
        DataFrame: 변환된 데이터프레임 (각 그룹이 개별 컬럼으로 구성)

    Examples:
        ```python
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        from hossam import *
        result = hs_prep.unmelt(df, id_vars='group', value_vars='value')
        # 결과: A 컬럼에는 [1, 2, NaN], B 컬럼에는 [3, 4, 5]
        ```
    """
    # 그룹별로 값들을 리스트로 모음
    grouped = data.groupby(id_vars, observed=True)[value_vars].apply(lambda x: x.tolist())
    series_dict = {}
    for idx, values in grouped.items():
        series_dict[str(idx)] = pd.Series(values)

    return DataFrame(series_dict)

# ===================================================================
# 지정된 변수의 이상치 테이블로 반환한다
# ===================================================================
def outlier_table(data: DataFrame, *fields: str, columns: list | None = None) -> DataFrame:
    """수치형 컬럼에 대한 사분위수 및 IQR 기반 이상치 경계를 계산한다.

    전달된 `fields`가 없으면 데이터프레임의 모든 수치형 컬럼을 대상으로 한다.
    결측치는 제외하고 사분위수를 계산한다.

    Args:
        data (DataFrame): 분석할 데이터프레임.
        *fields (str): 대상 컬럼명(들). 생략 시 모든 수치형 컬럼 대상.
        columns (list, optional): 변환할 컬럼명 목록. args와 중복 사용 불가.

    Returns:
        DataFrame: Q1, Q2(중앙값), Q3, IQR, 하한, 상한을 포함한 통계표.

    Examples:
        from hossam import *
        hs_prep.outlier_table(df, "value")
    """
    # columns 인자가 있으면 args보다 우선한다.
    if columns is not None:
        if args: # type: ignore
            raise ValueError("args와 columns 인자는 중복 사용할 수 없습니다.")
        args = columns

    target_fields = list(fields) if fields else list(data.select_dtypes(include=[np.number]).columns)
    result = []
    for f in target_fields:
        if f not in data.columns:
            continue

        series = data[f].dropna()
        if series.empty:
            continue

        q1 = series.quantile(q=0.25)
        q2 = series.quantile(q=0.5)
        q3 = series.quantile(q=0.75)

        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        result.append(
            {
                "FIELD": f,
                "Q1": q1,
                "Q2": q2,
                "Q3": q3,
                "IQR": iqr,
                "UP": up,
                "DOWN": down,
            }
        )

    return DataFrame(result).set_index("FIELD") if result else DataFrame()


# ===================================================================
# 이상치를 대체값(NaN, 0) 또는 중앙값으로 교체한다
# ===================================================================
def replace_outliner(data: DataFrame, method: str = "nan", *fields: str, columns: list | None = None) -> DataFrame:
    """이상치 경계값을 넘어가는 데이터를 경계값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        method (str): 대체 방법
            - nan: 결측치 대체
            - outline: 경계값 대체
            - mean: 평균 대체
            - most: 최빈값 대체
            - median: 중앙값 대체
        *fields (str): 컬럼명 목록
        columns (list, optional): 변환할 컬럼명 목록. args와 중복 사용 불가.

    Returns:
        DataFrame: 이상치가 경계값으로 대체된 데이터 프레임
    """
    # columns 인자가 있으면 args보다 우선한다.
    if columns is not None:
        if args: # type: ignore
            raise ValueError("args와 columns 인자는 중복 사용할 수 없습니다.")
        args = columns

    # 원본 데이터 프레임 복사
    df = data.copy()

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 이상치 경계값을 구한다.
    outliner_table = outlier_table(df, *fields)

    if outliner_table.empty:
        return data.copy()

    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        if method == "outline":
            df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = outliner_table.loc[f, "DOWN"]
            df.loc[df[f] > outliner_table.loc[f, "UP"], f] = outliner_table.loc[f, "UP"]
        else:
            df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = np.nan
            df.loc[df[f] > outliner_table.loc[f, "UP"], f] = np.nan

    # NaN으로 표시된 이상치를 지정한 방법으로 대체
    if method in {"mean", "median", "most"}:
        strategy_map = {"mean": "mean", "median": "median", "most": "most_frequent"}
        imr = SimpleImputer(missing_values=np.nan, strategy=strategy_map[method])
        df_imr = imr.fit_transform(df.values)
        df = DataFrame(df_imr, index=data.index, columns=df.columns)
    elif method not in {"nan", "outline"}:
        raise ValueError("method는 'nan', 'outline', 'mean', 'median', 'most' 중 하나여야 합니다.")

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df[category_fields] = cate

    return df

# ===================================================================
# 중빈 이상치를 제거한 연처리된 데이터프레임을 반환한다
# ===================================================================
def drop_outliner(data: DataFrame, *fields: str, columns: list | None = None) -> DataFrame:
    """이상치를 결측치로 변환한 후 모두 삭제한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록
        columns (list, optional): 변환할 컬럼명 목록. args와 중복 사용 불가.

    Returns:
        DataFrame: 이상치가 삭제된 데이터프레임
    """
    # columns 인자가 있으면 args보다 우선한다.
    if columns is not None:
        if args: # type: ignore
            raise ValueError("args와 columns 인자는 중복 사용할 수 없습니다.")
        args = columns

    df = replace_outliner(data, "nan", *fields)
    return df.dropna()


# ===================================================================
# 범주 변수를 더미 변수(One-Hot 인코딩)로 변환한다
# ===================================================================
def get_dummies(data: DataFrame, *args: str, columns: list | None = None, drop_first: bool = True, dtype: str = "int") -> DataFrame:
    """명목형 변수를 더미 변수로 변환한다.

    컬럼명을 지정하면 그 컬럼들만 더미 변수로 변환하고,
    지정하지 않으면 숫자 타입이 아닌 모든 컬럼(문자열/명목형)을 자동으로 더미 변수로 변환한다.

    Args:
        data (DataFrame): 데이터프레임
        *args (str): 변환할 컬럼명 목록. 지정하지 않으면 숫자형이 아닌 모든 컬럼 자동 선택.
        columns (list, optional): 변환할 컬럼명 목록. args와 중복 사용 불가.
        drop_first (bool, optional): 첫 번째 더미 변수 제거 여부. 기본값 True.
        dtype (str, optional): 더미 변수 데이터 타입. 기본값 "int".

    Returns:
        DataFrame: 더미 변수로 변환된 데이터프레임

    Examples:
        ```python
        from hossam import *
        # 전체 비숫자 컬럼 자동 변환
        result = hs_prep.get_dummies(df)
        # 특정 컬럼만 변환
        result = hs_prep.get_dummies(df, 'cut', 'color', 'clarity')
        # 옵션 지정
        result = hs_prep.get_dummies(df, 'col1', drop_first=False, dtype='bool')
        ```
    """
    # columns 인자가 있으면 args보다 우선한다.
    if columns is not None:
        if args:
            raise ValueError("args와 columns 인자는 중복 사용할 수 없습니다.")
        args = columns # type: ignore

    if not args:
        # args가 없으면 숫자 타입이 아닌 모든 컬럼 자동 선택
        cols_to_convert = []
        for f in data.columns:
            if not pd.api.types.is_numeric_dtype(data[f]):
                cols_to_convert.append(f)
        args = cols_to_convert # type: ignore
    else:
        # args가 있으면 그 컬럼들만 사용 (존재 여부 확인)
        args = [c for c in args if c in data.columns] # type: ignore

    # pandas.get_dummies 사용 (재귀 문제 없음)
    return pd.get_dummies(data, columns=args, drop_first=drop_first, dtype=dtype) if args else data.copy() # type: ignore


# ===================================================================
# 범주형 변수(Categorical)를 순차적 레이블로 인코딩한다
# ===================================================================
def labelling(data: DataFrame, *fields: str) -> DataFrame:
    """명목형 변수를 라벨링한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 명목형 컬럼 목록

    Returns:
        DataFrame: 라벨링된 데이터프레임
    """
    df = data.copy()

    for f in fields:
        vc = sorted(list(df[f].unique()))
        label = {v: i for i, v in enumerate(vc)}
        df[f] = df[f].map(label).astype("int")

        # 라벨링 상황을 출력한다.
        i = []
        v = []
        for k in label:
            i.append(k)
            v.append(label[k])

        label_df = DataFrame({"label": v}, index=i)
        label_df.index.name = f
        pretty_table(label_df)

    return df


# ===================================================================
# 연속형 변수를 다양한 기준으로 구간화하여 명목형 변수로 추가한다
# ===================================================================
def bin_continuous(
    data: DataFrame,
    field: str,
    method: str = "natural_breaks",
    bins: int | list[float] | None = None,
    labels: list[str] | None = None,
    new_col: str | None = None,
    is_log_transformed: bool = False,
    apply_labels: bool = True,
) -> DataFrame:
    """연속형 변수를 다양한 알고리즘으로 구간화해 명목형 파생변수를 추가한다.

    지원 방법:
    - "natural_breaks"(기본): Jenks 자연 구간화. jenkspy 미사용 시 quantile로 대체
      기본 라벨: "X-Y" 형식 (예: "18-30", "30-40")
    - "quantile"/"qcut"/"equal_freq": 분위수 기반 동빈도
      기본 라벨: "X-Y" 형식
    - "equal_width"/"uniform": 동일 간격
      기본 라벨: "X-Y" 형식
    - "std": 평균±표준편차를 경계로 4구간 생성
      라벨: "low", "mid_low", "mid_high", "high"
    - "lifecourse"/"life_stage": 생애주기 5단계
      라벨: "아동", "청소년", "청년", "중년", "노년" (경계: 0, 13, 19, 40, 65)
    - "age_decade": 10대 단위 연령대
      라벨: "아동", "10대", "20대", "30대", "40대", "50대", "60대 이상"
    - "health_band"/"policy_band": 의료비 위험도 기반 연령대
      라벨: "18-29", "30-39", "40-49", "50-64", "65+"
    - 커스텀 구간: bins에 경계 리스트 전달 (예: [0, 30, 50, 100])

    Args:
        data (DataFrame): 입력 데이터프레임
        field (str): 구간화할 연속형 변수명
        method (str): 구간화 알고리즘 키워드 (기본값: "natural_breaks")
        bins (int|list[float]|None):
            - int: 생성할 구간 개수 (quantile, equal_width, natural_breaks에서 사용)
            - list: 경계값 리스트 (커스텀 구간화)
            - None: 기본값 사용 (quantile/equal_width는 4~5, natural_breaks는 5)
        labels (list[str]|None): 구간 레이블 목록
            - None: method별 기본 라벨 자동 생성
            - list: 사용자 정의 라벨 (구간 개수와 일치해야 함)
        new_col (str|None): 생성할 컬럼명
            - None: f"{field}_bin" 사용 (예: "age_bin")
        is_log_transformed (bool): 대상 컬럼이 로그 변환되어 있는지 여부
            - True: 지정된 컬럼을 역변환(exp)한 후 구간화
            - False: 원래 값 그대로 구간화 (기본값)
        apply_labels (bool): 구간에 숫자 인덱스를 적용할지 여부
            - True: 숫자 인덱스 사용 (0, 1, 2, 3, ...) (기본값)
            - False: 문자 라벨 적용 (예: "18~30", "아동")

    Returns:
        DataFrame: 원본에 구간화된 명목형 컬럼이 추가된 데이터프레임

    Examples:
        ```python
        from hossam import *

        # 동일 간격으로 5개 구간 생성 (숫자 인덱스):
        df = pd.DataFrame({'age': [20, 35, 50, 65]})
        result = hs_prep.bin_continuous(df, 'age', method='equal_width', bins=5)
        print(result['age_bin'])  # 0, 1, 2, ... (숫자 인덱스)

        # 문자 레이블 사용:
        result = hs_prep.bin_continuous(df, 'age', method='equal_width', bins=5, apply_labels=False)
        print(result['age_bin'])  # 20~30, 30~40, ... (문자 레이블)

        # 생애주기 기반 구간화:
        result = hs_prep.bin_continuous(df, 'age', method='lifecourse')
        print(result['age_bin'])  # 0, 1, 2, 3, 4 (숫자 인덱스)

        # 생애주기 문자 레이블:
        result = hs_prep.bin_continuous(df, 'age', method='lifecourse', apply_labels=False)
        print(result['age_bin'])  # 아동, 청소년, 청년, 중년, 노년

        # 의료비 위험도 기반 연령대 (health_band):
        result = hs_prep.bin_continuous(df, 'age', method='health_band', apply_labels=False)
        print(result['age_bin'])  # 18-29, 30-39, 40-49, 50-64, 65+

        # 로그 변환된 컬럼 역변환 후 구간화:
        df_log = pd.DataFrame({'charges_log': [np.log(1000), np.log(5000), np.log(50000)]})
        result = hs_prep.bin_continuous(df_log, 'charges_log', method='equal_width', is_log_transformed=True)
        print(result['charges_log_bin'])  # 0, 1, 2 (숫자 인덱스)
        ```
    """

    if field not in data.columns:
        return data

    df = data.copy()
    series = df[field].copy()

    # 로그 변환 역변환
    if is_log_transformed:
        series = np.exp(series)

    new_col = new_col or f"{field}_bin"
    method_key = (method or "").lower()

    def _cut(edges: list[float], default_labels: list[str] | None = None, right: bool = False, ordered: bool = True):
        nonlocal labels
        use_labels = None

        # apply_labels=True일 때 숫자 인덱스, False일 때 문자 레이블
        if apply_labels:
            # 숫자 인덱스 생성
            numeric_labels = list(range(len(edges) - 1))
            use_labels = numeric_labels
        else:
            # 문자 레이블 적용
            use_labels = labels if labels is not None else default_labels

        df[new_col] = pd.cut(
            series,
            bins=edges,
            labels=use_labels,
            right=right,
            include_lowest=True,
            ordered=False,  # 레이블이 있으므로 ordered=False 사용
        )
        df[new_col] = df[new_col].astype("category")

    # 생애주기 구분
    if method_key in {"lifecourse", "life_stage", "lifecycle", "life"}:
        edges = [0, 13, 19, 40, 65, np.inf]
        # 나이 구간을 함께 표기한 라벨 (apply_labels=False에서 사용)
        default_labels = [
            "아동(0~12)",
            "청소년(13~18)",
            "청년(19~39)",
            "중년(40~64)",
            "노년(65+)",
        ]
        _cut(edges, default_labels, right=False)
        return df

    # 연령대(10단위)
    if method_key in {"age_decade", "age10", "decade"}:
        edges = [0, 13, 20, 30, 40, 50, 60, np.inf]
        default_labels = ["아동", "10대", "20대", "30대", "40대", "50대", "60대 이상"]
        _cut(edges, default_labels, right=False)
        return df

    # 건강/제도 기준 (의료비 위험군 분류 기준)
    if method_key in {"health_band", "policy_band", "health"}:
        # 연령 데이터 최소값(예: 18세)과 레이블을 일치시킴
        edges = [0, 19, 30, 40, 50, 65, np.inf]
        default_labels = ["0~18", "19-29", "30-39", "40-49", "50-64", "65+"]
        _cut(edges, default_labels, right=False)
        return df

    # 표준편차 기반
    if method_key == "std":
        mu = series.mean()
        sd = series.std(ddof=0)
        edges = [-np.inf, mu - sd, mu, mu + sd, np.inf]
        default_labels = ["low", "mid_low", "mid_high", "high"]
        _cut(edges, default_labels, right=True)
        return df

    # 동일 간격
    if method_key in {"equal_width", "uniform"}:
        k = bins if isinstance(bins, int) and bins > 0 else 5
        _, edges = pd.cut(series, bins=k, include_lowest=True, retbins=True)

        # apply_labels=True: 숫자 인덱스 / False: 문자 레이블
        if apply_labels:
            # 숫자 인덱스 사용 (0, 1, 2, ...)
            numeric_labels = list(range(len(edges) - 1))
            df[new_col] = pd.cut(series, bins=edges, labels=numeric_labels, include_lowest=True, ordered=False) # type: ignore
        else:
            # 문자 레이블 적용
            if labels is None:
                auto_labels = []
                for i in range(len(edges) - 1):
                    left = f"{edges[i]:.2f}" if edges[i] != -np.inf else "-∞"
                    right = f"{edges[i+1]:.2f}" if edges[i+1] != np.inf else "∞"
                    # 정수값인 경우 소수점 제거
                    try:
                        left = str(int(float(left))) if float(left) == int(float(left)) else left
                        right = str(int(float(right))) if float(right) == int(float(right)) else right
                    except:
                        pass
                    auto_labels.append(f"{left}~{right}")
                df[new_col] = pd.cut(series, bins=edges, labels=auto_labels, include_lowest=True, ordered=False) # type: ignore
            else:
                df[new_col] = pd.cut(series, bins=edges, labels=labels, include_lowest=True, ordered=False) # type: ignore

        df[new_col] = df[new_col].astype("category")
        return df

    # 분위수 기반 동빈도
    if method_key in {"quantile", "qcut", "equal_freq"}:
        k = bins if isinstance(bins, int) and bins > 0 else 4
        # apply_labels=False일 때 기본 레이블을 사분위수 위치(Q1~)로 설정
        default_q_labels = labels if labels is not None else [f"Q{i+1}" for i in range(k)]
        try:
            if apply_labels:
                # 숫자 인덱스 사용
                numeric_labels = list(range(k))
                df[new_col] = pd.qcut(series, q=k, labels=numeric_labels, duplicates="drop")
            else:
                # 사분위수 위치 기반 문자 레이블(Q1, Q2, ...)
                df[new_col] = pd.qcut(series, q=k, labels=default_q_labels, duplicates="drop")
        except ValueError:
            _, edges = pd.cut(series, bins=k, include_lowest=True, retbins=True)
            # apply_labels=True: 숫자 인덱스 / False: 문자 레이블
            n_bins = len(edges) - 1
            if apply_labels:
                numeric_labels = list(range(n_bins))
                df[new_col] = pd.cut(series, bins=edges, labels=numeric_labels, include_lowest=True, ordered=False) # type: ignore
            else:
                if labels is None:
                    position_labels = [f"Q{i+1}" for i in range(n_bins)]
                    df[new_col] = pd.cut(series, bins=edges, labels=position_labels, include_lowest=True, ordered=False) # type: ignore
                else:
                    df[new_col] = pd.cut(series, bins=edges, labels=labels, include_lowest=True, ordered=False) # type: ignore
        df[new_col] = df[new_col].astype("category")
        return df

    # 자연 구간화 (Jenks) - 의존성 없으면 분위수로 폴백
    if method_key in {"natural_breaks", "natural", "jenks"}:
        k = bins if isinstance(bins, int) and bins > 1 else 5
        series_nonnull = series.dropna() # type: ignore
        k = min(k, max(2, series_nonnull.nunique()))
        edges = None
        try:
            edges = jenkspy.jenks_breaks(series_nonnull.to_list(), nb_class=k) # type: ignore
            edges[0] = -np.inf
            edges[-1] = np.inf
        except Exception:
            try:
                use_labels = labels if apply_labels else None
                df[new_col] = pd.qcut(series, q=k, labels=use_labels, duplicates="drop")
                df[new_col] = df[new_col].astype("category")
                return df
            except Exception:
                edges = None

        if edges:
            # apply_labels=True: 숫자 인덱스 / False: 문자 레이블
            if apply_labels:
                # 숫자 인덱스 사용
                numeric_labels = list(range(len(edges) - 1))
                df[new_col] = pd.cut(series, bins=edges, labels=numeric_labels, include_lowest=True, ordered=False)
                df[new_col] = df[new_col].astype("category")
            else:
                if labels is None:
                    auto_labels = []
                    for i in range(len(edges) - 1):
                        left = f"{edges[i]:.2f}" if edges[i] != -np.inf else "-∞"
                        right = f"{edges[i+1]:.2f}" if edges[i+1] != np.inf else "∞"
                        # 정수값인 경우 소수점 제거
                        try:
                            left = str(int(float(left))) if float(left) == int(float(left)) else left
                            right = str(int(float(right))) if float(right) == int(float(right)) else right
                        except:
                            pass
                        auto_labels.append(f"{left}~{right}")
                    _cut(edges, auto_labels, right=True, ordered=False)
                else:
                    _cut(edges, labels, right=True, ordered=False)
        else:
            _, cut_edges = pd.cut(series, bins=k, include_lowest=True, retbins=True)
            if apply_labels:
                # 숫자 인덱스 사용
                numeric_labels = list(range(len(cut_edges) - 1))
                df[new_col] = pd.cut(series, bins=cut_edges, labels=numeric_labels, include_lowest=True, ordered=False) # type: ignore
            else:
                if labels is None:
                    auto_labels = []
                    for i in range(len(cut_edges) - 1):
                        left = f"{cut_edges[i]:.2f}" if cut_edges[i] != -np.inf else "-∞"
                        right = f"{cut_edges[i+1]:.2f}" if cut_edges[i+1] != np.inf else "∞"
                        # 정수값인 경우 소수점 제거
                        try:
                            left = str(int(float(left))) if float(left) == int(float(left)) else left
                            right = str(int(float(right))) if float(right) == int(float(right)) else right
                        except:
                            pass
                        auto_labels.append(f"{left}~{right}")
                    df[new_col] = pd.cut(series, bins=cut_edges, labels=auto_labels, include_lowest=True, ordered=False) # type: ignore
                else:
                    df[new_col] = pd.cut(series, bins=cut_edges, labels=labels, include_lowest=True, ordered=False) # type: ignore
            df[new_col] = df[new_col].astype("category")
        return df

    # 커스텀 경계
    if isinstance(bins, list) and len(bins) >= 2:
        edges = sorted(bins)
        _cut(edges, labels, right=False)
        return df

    # 기본 폴백: 분위수 4구간
    df[new_col] = pd.qcut(series, q=4, labels=labels, duplicates="drop")
    df[new_col] = df[new_col].astype("category")
    return df


# ===================================================================
# 지정된 변수에 로그 먼저 변환을 적용한다
# ===================================================================
def log_transform(data: DataFrame, *fields: str, columns: list | None = None) -> DataFrame:
    """수치형 변수에 대해 로그 변환을 수행한다.

    자연로그(ln)를 사용하여 변환하며, 0 또는 음수 값이 있을 경우
    최소값을 기준으로 보정(shift)을 적용한다.

    Args:
        data (DataFrame): 변환할 데이터프레임.
        *fields (str): 변환할 컬럼명 목록. 지정하지 않으면 모든 수치형 컬럼을 처리.
        columns (list, optional): 변환할 컬럼명 목록. fields와 중복 사용 불가.

    Returns:
        DataFrame: 로그 변환된 데이터프레임.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        df = DataFrame({'x': [1, 10, 100], 'y': [2, 20, 200], 'z': ['a', 'b', 'c']})

        # 전체 수치형 컬럼에 대한 로그 변환:
        result = hs_prep.log_transform(df)
        print(result)

        # 특정 컬럼만 변환:
        result = hs_prep.log_transform(df, 'x', 'y')
        print(result)
        ```

    Notes:
        - 수치형이 아닌 컬럼은 자동으로 제외됩니다.
        - 0 또는 음수 값이 있는 경우 자동으로 보정됩니다.
        - 변환 공식: log(x + shift), 여기서 shift = 1 - min(x) (min(x) <= 0인 경우)
    """
    df = data.copy()

    if columns is not None:
        if fields:
            raise ValueError("fields와 columns 인자는 중복 사용할 수 없습니다.")
        fields = columns # type: ignore

    # 대상 컬럼 결정
    if not fields:
        # 모든 수치형 컬럼 선택
        target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        target_cols = list(fields)

    # 각 컬럼에 대해 로그 변환 수행
    for col in target_cols:
        # 컬럼이 존재하고 수치형인지 확인
        if col not in df.columns:
            continue

        if df[col].dtype not in ['int', 'int32', 'int64', 'float', 'float32', 'float64']:
            continue

        # 최소값 확인
        min_val = df[col].min()

        # 0 또는 음수가 있으면 shift 적용
        if min_val <= 0:
            shift = 1 - min_val
            df[col] = np.log(df[col] + shift)
        else:
            df[col] = np.log(df[col])

    return df


# ===================================================================
# 변수 간의 상호작용 항을 추가한 데이터프레임을 반환한다
# ===================================================================
def add_interaction(data: DataFrame, pairs: list[tuple[str, str]] | None = None) -> DataFrame:
    """데이터프레임에 상호작용(interaction) 항을 추가한다.

    수치형 및 명목형 변수 간의 상호작용 항을 생성하여 데이터프레임에 추가한다.
    - `수치형 * 수치형`: 두 변수의 곱셈 (col1*col2)
    - `수치형 * 명목형`: 명목형의 각 카테고리별 수치형 변수 생성 (col1*col2_category)
    - `명목형 * 명목형`: 두 명목형을 결합한 새 명목형 변수 생성 (col1_col2)

    Args:
        data (DataFrame): 원본 데이터프레임.
        pairs (list[tuple[str, str]], optional): 직접 지정할 교호작용 쌍의 리스트.
            예: [("age", "gender"), ("color", "cut")]
            None이면 모든 수치형 컬럼의 2-way 상호작용을 생성.

    Returns:
        DataFrame: 상호작용 항이 추가된 새 데이터프레임.

    Examples:
        ```python
        from hossam import *
        from padas import DataFrame

        # 수치형 변수들의 상호작용:
        df = DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        result = hs_prep.add_interaction(df)
        print(result.columns)  # x1, x2, x1*x2

        # 수치형과 명목형의 상호작용:
        df = DataFrame({'age': [20, 30, 40], 'gender': ['M', 'F', 'M']})
        result = hs_prep.add_interaction(df, pairs=[('age', 'gender')])
        print(result.columns)  # age, gender, age*gender_M, age*gender_F

        # 명목형끼리의 상호작용:
        df = DataFrame({'color': ['R', 'G', 'B'], 'cut': ['A', 'B', 'A']})
        result = hs_prep.add_interaction(df, pairs=[('color', 'cut')])
        print(result.columns)  # color, cut, color_cut
        ```
    """
    df = data.copy()

    # pairs가 제공되면 그것을 사용, 아니면 모든 수치형 컬럼의 2-way 상호작용 생성
    if pairs is not None:
        cols_to_interact = [(col1, col2) for col1, col2 in pairs
                           if col1 in df.columns and col2 in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_interact = list(combinations(numeric_cols, 2))

    # 상호작용 항 생성
    for col1, col2 in cols_to_interact:
        is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
        is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])

        # Case 1: 둘 다 수치형 -> 곱셈
        if is_col1_numeric and is_col2_numeric:
            interaction_col_name = f"{col1}*{col2}"
            df[interaction_col_name] = df[col1] * df[col2]

        # Case 2: 하나는 수치형, 하나는 명목형 -> 명목형의 각 카테고리별로 수치형 변수 생성
        elif is_col1_numeric and not is_col2_numeric:
            # col1은 수치형, col2는 명목형
            categories = df[col2].unique()
            for cat in categories:
                # 결측치 처리
                cat_str = str(cat) if pd.notna(cat) else "nan"
                interaction_col_name = f"{col1}*{col2}_{cat_str}"
                # 해당 카테고리인 경우 수치형 값, 아니면 0
                df[interaction_col_name] = df[col1] * (df[col2] == cat).astype(int)

        elif not is_col1_numeric and is_col2_numeric:
            # col1은 명목형, col2는 수치형
            categories = df[col1].unique()
            for cat in categories:
                cat_str = str(cat) if pd.notna(cat) else "nan"
                interaction_col_name = f"{col2}*{col1}_{cat_str}"
                df[interaction_col_name] = df[col2] * (df[col1] == cat).astype(int)

        # Case 3: 둘 다 명목형 -> 두 변수를 결합한 새로운 명목형 변수
        else:
            interaction_col_name = f"{col1}_{col2}"
            # 문자열로 변환 후 결합 (결측치 처리 포함)
            df[interaction_col_name] = (
                df[col1].astype(str).fillna("nan") + "_" +
                df[col2].astype(str).fillna("nan")
            )

    return df
