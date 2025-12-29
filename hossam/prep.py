# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import joblib
import numpy as np

# -------------------------------------------------------------
from pandas import DataFrame, get_dummies
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------
from .util import hs_pretty_table

# -------------------------------------------------------------
def hs_standard_scaler(
    data: any, yname: str | None = None, save_path: str | None = None, load_path: str | None = None
) -> DataFrame:
    """연속형 변수에 대해 Standard Scaling을 수행한다.

    - DataFrame 입력 시: 비수치형/종속변수를 분리한 후 스케일링하고 다시 합칩니다.
    - 배열 입력 시: 그대로 스케일링된 ndarray를 반환합니다.
    - `load_path`가 주어지면 기존 스케일러를 재사용하고, `save_path`가 주어지면 학습된 스케일러를 저장합니다.
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
        return sdata

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


# -------------------------------------------------------------
def hs_minmax_scaler(
    data: any, yname: str | None = None, save_path: str | None = None, load_path: str | None = None
) -> DataFrame:
    """연속형 변수에 대해 MinMax Scaling을 수행한다.

    DataFrame은 비수치/종속변수를 분리 후 스케일링하고 재결합하며, 배열 입력은 그대로 ndarray를 반환한다.
    `load_path` 제공 시 기존 스케일러를 사용하고, `save_path` 제공 시 학습 스케일러를 저장한다.
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
        return sdata

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


# -------------------------------------------------------------
def hs_set_category(data: DataFrame, *args: str) -> DataFrame:
    """카테고리 데이터를 설정한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        *args (str): 컬럼명 목록

    Returns:
        DataFrame: 카테고리 설정된 데이터프레임
    """
    df = data.copy()

    for k in args:
        df[k] = df[k].astype("category")

    return df


# -------------------------------------------------------------
def hs_unmelt(
    data: DataFrame, id_vars: str = "class", value_vars: str = "values"
) -> DataFrame:
    """두 개의 컬럼으로 구성된 데이터프레임에서 하나는 명목형, 나머지는 연속형일 경우
    명목형 변수의 값에 따라 고유한 변수를 갖는 데이터프레임으로 변환한다.

    Args:
        data (DataFrame): 데이터프레임
        id_vars (str, optional): 명목형 변수의 컬럼명. Defaults to 'class'.
        value_vars (str, optional): 연속형 변수의 컬럼명. Defaults to 'values'.

    Returns:
        DataFrame: 변환된 데이터프레임
    """
    result = data.groupby(id_vars)[value_vars].apply(list)
    mydict = {}

    for i in result.index:
        mydict[i] = result[i]

    return DataFrame(mydict)


# -------------------------------------------------------------
def hs_replace_missing_value(data: DataFrame, strategy: str = "mean") -> DataFrame:
    """SimpleImputer로 결측치를 대체한다.

    Args:
        data (DataFrame): 결측치가 포함된 데이터프레임
        strategy (str, optional): 결측치 대체 방식(mean, median, most_frequent, constant). Defaults to "mean".

    Returns:
        DataFrame: 결측치가 대체된 데이터프레임
    """

    allowed = {"mean", "median", "most_frequent", "constant"}
    if strategy not in allowed:
        raise ValueError(f"strategy는 {allowed} 중 하나여야 합니다.")

    imr = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imr = imr.fit_transform(data.values)
    return DataFrame(df_imr, index=data.index, columns=data.columns)


# -------------------------------------------------------------
def hs_outlier_table(data: DataFrame, *fields: str):
    """수치형 컬럼에 대한 사분위수 및 IQR 기반 이상치 경계를 계산한다.

    전달된 `fields`가 없으면 데이터프레임의 모든 수치형 컬럼을 대상으로 한다.
    결측치는 제외하고 사분위수를 계산한다.
    """

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


# -------------------------------------------------------------
def hs_replace_outliner(data: DataFrame, method: str = "nan", *fields: str) -> DataFrame:
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

    Returns:
        DataFrame: 이상치가 경계값으로 대체된 데이터 프레임
    """

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
    outliner_table = hs_outlier_table(df, *fields)

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

# -------------------------------------------------------------
def hs_drop_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 변환한 후 모두 삭제한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 삭제된 데이터프레임
    """

    df = hs_replace_outliner(data, "nan", *fields)
    return df.dropna()


# -------------------------------------------------------------

def hs_dummies(data: DataFrame, drop_first=True, dtype="int", *args: str) -> DataFrame:
    """명목형 변수를 더미 변수로 변환한다.

    Args:
        data (DataFrame): 데이터프레임
        *args (str): 명목형 컬럼 목록

    Returns:
        DataFrame: 더미 변수로 변환된 데이터프레임
    """
    if not args:
        args = []

        for f in data.columns:
            if data[f].dtypes == "category":
                args.append(f)
    else:
        args = list(args)

    return get_dummies(data, columns=args, drop_first=drop_first, dtype=dtype)



# -------------------------------------------------------------

def hs_labelling(data: DataFrame, *fields: str) -> DataFrame:
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
        hs_pretty_table(label_df)

    return df
