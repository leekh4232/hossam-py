# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import numpy as np
import joblib

# -------------------------------------------------------------
import joblib

# -------------------------------------------------------------
from pandas import DataFrame, get_dummies

# -------------------------------------------------------------
from tabulate import tabulate

# -------------------------------------------------------------
from scipy.stats import normaltest

# -------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor\

# -------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------

def my_normalize_data(
    mean: float, std: float, size: int = 100, round: int = 2
) -> np.ndarray:
    """정규분포를 따르는 데이터를 생성한다.

    Args:
        mean (float): 평균
        std (float): 표준편차
        size (int, optional): 데이터 크기. Defaults to 1000.

    Returns:
        np.ndarray: 정규분포를 따르는 데이터
    """
    p = 0
    x = []
    while p < 0.05:
        x = np.random.normal(mean, std, size).round(round)
        _, p = normaltest(x)

    return x


# -------------------------------------------------------------

def my_normalize_df(
    means: list = [0, 0, 0],
    stds: list = [1, 1, 1],
    sizes: list = [100, 100, 100],
    rounds: int = 2,
) -> DataFrame:
    """정규분포를 따르는 데이터프레임을 생성한다.

    Args:
        means (list): 평균 목록
        stds (list): 표준편차 목록
        sizes (list, optional): 데이터 크기 목록. Defaults to [100, 100, 100].
        rounds (int, optional): 반올림 자리수. Defaults to 2.

    Returns:
        DataFrame: 정규분포를 따르는 데이터프레임
    """
    data = {}
    for i in range(0, len(means)):
        data[f"X{i+1}"] = my_normalize_data(means[i], stds[i], sizes[i], rounds)

    return DataFrame(data)


# -------------------------------------------------------------

def my_pretty_table(data: DataFrame, headers: str = "keys") -> None:
    tabulate.WIDE_CHARS_MODE = False
    print(
        tabulate(
            data, headers="keys", tablefmt="simple", showindex=True, numalign="right"
        )
    )


# -------------------------------------------------------------


def my_standard_scaler(
    data: any, yname: str = None, save_path: str = None, load_path: str = None
) -> DataFrame:
    """데이터프레임의 연속형 변수에 대해 Standard Scaling을 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.
        save_path (str, optional): 저장할 경로. Defaults to None.
        load_path (str, optional): 불러올 경로. Defaults to None.

    Returns:
        DataFrame: 표준화된 데이터프레임
    """
    # 원본 데이터 프레임 복사
    df = data.copy()

    if data.__class__.__name__ == "DataFrame":
        # 종속변수만 별도로 분리
        if yname:
            y = df[yname]
            df = df.drop(yname, axis=1)

        # 카테고리 타입만 골라냄
        category_fields = []
        for f in df.columns:
            if df[f].dtypes not in [
                "int",
                "int32",
                "int64",
                "float",
                "float32",
                "float64",
            ]:
                category_fields.append(f)

        cate = df[category_fields]
        df = df.drop(category_fields, axis=1)

    # 표준화 수행
    if load_path:
        scaler = joblib.load(load_path)
        sdata = scaler.transform(df)

    else:
        scaler = StandardScaler()
        sdata = scaler.fit_transform(df)

    # 스케일러 저장 경로가 있을 경우
    if save_path:
        joblib.dump(value=scaler, filename=save_path)

    if data.__class__.__name__ != "DataFrame":
        return sdata

    # ----------------------------------
    std_df = DataFrame(data=sdata, index=data.index, columns=df.columns)

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        std_df[category_fields] = cate

    # 분리했던 종속 변수를 다시 결합
    if yname:
        std_df[yname] = y

    return std_df


# -------------------------------------------------------------

def my_minmax_scaler(
    data: DataFrame, yname: str = None, save_path: str = None, load_path: str = None
) -> DataFrame:
    """데이터프레임의 연속형 변수에 대해 MinMax Scaling을 수행한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        yname (str, optional): 종속변수의 컬럼명. Defaults to None.
        save_path (str, optional): 저장할 경로. Defaults to None.
        load_path (str, optional): 불러올 경로. Defaults to None.
    Returns:
        DataFrame: 표준화된 데이터프레임
    """
    # 원본 데이터 프레임 복사
    df = data.copy()

    # 종속변수만 별도로 분리
    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 표준화 수행
    if load_path:
        scaler = joblib.load(load_path)
        sdata = scaler.transform(df)

    else:
        scaler = MinMaxScaler()
        sdata = scaler.fit_transform(df)

    std_df = DataFrame(data=sdata, index=data.index, columns=df.columns)

    # 스케일러 저장 경로가 있을 경우
    if save_path:
        joblib.dump(scaler, save_path)

    if data.__class__.__name__ != "DataFrame":
        return sdata

    # ----------------------------------

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        std_df[category_fields] = cate

    # 분리했던 종속 변수를 다시 결합
    if yname:
        std_df[yname] = y

    return std_df


# -------------------------------------------------------------

def my_category(data: DataFrame, *args: str) -> DataFrame:
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


def my_unmelt(
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

def my_replace_missing_value(data: DataFrame, strategy: str = "mean") -> DataFrame:
    # 결측치 처리 규칙 생성
    imr = SimpleImputer(missing_values=np.nan, strategy=strategy)

    # 결측치 처리 규칙 적용 --> 2차원 배열로 반환됨
    df_imr = imr.fit_transform(data.values)

    # 2차원 배열을 데이터프레임으로 변환 후 리턴
    return DataFrame(df_imr, index=data.index, columns=data.columns)


# -------------------------------------------------------------

def my_outlier_table(data: DataFrame, *fields: str):
    """데이터프레임의 사분위수와 결측치 경계값을 구한다.
    함수 호출 전 상자그림을 통해 결측치가 확인된 필드에 대해서만 처리하는 것이 좋다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: IQ
    """
    if not fields:
        fields = data.columns

    result = []
    for f in fields:
        # 숫자 타입이 아니라면 건너뜀
        if data[f].dtypes not in [
            "int",
            "int32",
            "int64",
            "float",
            "float32",
            "float64",
        ]:
            continue

        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q2 = data[f].quantile(q=0.5)
        q3 = data[f].quantile(q=0.75)

        # 결측치 경계
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        iq = {
            "FIELD": f,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "IQR": iqr,
            "UP": up,
            "DOWN": down,
        }

        result.append(iq)

    return DataFrame(result).set_index("FIELD")


# -------------------------------------------------------------

def my_replace_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치 경계값을 넘어가는 데이터를 경계값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
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
    outliner_table = my_outlier_table(df, *fields)

    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = outliner_table.loc[f, "DOWN"]
        df.loc[df[f] > outliner_table.loc[f, "UP"], f] = outliner_table.loc[f, "UP"]

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df[category_fields] = cate

    return df


# -------------------------------------------------------------

def my_replace_outliner_to_nan(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 결측치로 대체된 데이터프레임
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
    outliner_table = my_outlier_table(df, *fields)

    # 이상치가 발견된 필드에 대해서만 처리
    for f in outliner_table.index:
        df.loc[df[f] < outliner_table.loc[f, "DOWN"], f] = np.nan
        df.loc[df[f] > outliner_table.loc[f, "UP"], f] = np.nan

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df[category_fields] = cate

    return df


# -------------------------------------------------------------

def my_replace_outliner_to_mean(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 평균값으로 대체한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 평균값으로 대체된 데이터프레임
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

    # 이상치를 결측치로 대체한다.
    if not fields:
        fields = df.columns

    df2 = my_replace_outliner_to_nan(df, *fields)

    # 결측치를 평균값으로 대체한다.
    df3 = my_replace_missing_value(df2, "mean")

    # 분리했던 카테고리 타입을 다시 병합
    if category_fields:
        df3[category_fields] = cate

    return df3


# -------------------------------------------------------------

def my_drop_outliner(data: DataFrame, *fields: str) -> DataFrame:
    """이상치를 결측치로 변환한 후 모두 삭제한다.

    Args:
        data (DataFrame): 데이터프레임
        *fields (str): 컬럼명 목록

    Returns:
        DataFrame: 이상치가 삭제된 데이터프레임
    """

    df = my_replace_outliner_to_nan(data, *fields)
    return df.dropna()


# -------------------------------------------------------------

def my_dummies(data: DataFrame, drop_first=True, dtype="int", *args: str) -> DataFrame:
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

def my_trend(x: any, y: any, degree=2, value_count=100) -> tuple:
    """x, y 데이터에 대한 추세선을 구한다.

    Args:
        x (_type_): 산점도 그래프에 대한 x 데이터
        y (_type_): 산점도 그래프에 대한 y 데이터
        degree (int, optional): 추세선 방정식의 차수. Defaults to 2.
        value_count (int, optional): x 데이터의 범위 안에서 간격 수. Defaults to 100.

    Returns:
        tuple: (v_trend, t_trend)
    """
    # [ a, b, c ] ==> ax^2 + bx + c
    coeff = np.polyfit(x, y, degree)

    if type(x) == "list":
        minx = min(x)
        maxx = max(x)
    else:
        minx = x.min()
        maxx = x.max()

    v_trend = np.linspace(minx, maxx, value_count)

    t_trend = coeff[-1]
    for i in range(0, degree):
        t_trend += coeff[i] * v_trend ** (degree - i)

    return (v_trend, t_trend)


# -------------------------------------------------------------

def my_labelling(data: DataFrame, *fields: str) -> DataFrame:
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
        my_pretty_table(label_df)

    return df

# -------------------------------------------------------------

def my_vif_filter(
    data: DataFrame,
    yname: str = None,
    ignore: list = [],
    threshold: float = 10,
    verbose: bool = False,
) -> DataFrame:
    """독립변수 간 다중공선성을 검사하여 VIF가 threshold 이상인 변수를 제거한다.

    Args:
        data (DataFrame): 데이터프레임
        yname (str, optional): 종속변수 컬럼명. Defaults to None.
        ignore (list, optional): 제외할 컬럼 목록. Defaults to [].
        threshold (float, optional): VIF 임계값. Defaults to 10.
        verbose (bool, optional): True일 경우 VIF를 출력한다. Defaults to False.

    Returns:
        DataFrame: VIF가 threshold 이하인 변수만 남은 데이터프레임
    """
    df = data.copy()

    if yname:
        y = df[yname]
        df = df.drop(yname, axis=1)

    # 카테고리 타입만 골라냄
    category_fields = []
    for f in df.columns:
        if df[f].dtypes not in ["int", "int32", "int64", "float", "float32", "float64"]:
            category_fields.append(f)

    cate = df[category_fields]
    df = df.drop(category_fields, axis=1)

    # 제외할 필드를 제거
    if ignore:
        ignore_df = df[ignore]
        df = df.drop(ignore, axis=1)

    # VIF 계산
    while True:
        xnames = list(df.columns)
        vif = {}

        for x in xnames:
            vif[x] = variance_inflation_factor(df, xnames.index(x))

        if verbose:
            print(vif)

        maxkey = max(vif, key=vif.get)

        if vif[maxkey] <= threshold:
            break

        df = df.drop(maxkey, axis=1)

    # 출력 옵션이 False일 경우 최종 값만 출력
    if not verbose:
        print(vif)

    # 분리했던 명목형 변수를 다시 결합
    if category_fields:
        df[category_fields] = cate

    # 분리했던 제외할 필드를 다시 결합
    if ignore:
        df[ignore] = ignore_df

    # 분리했던 종속 변수를 다시 결합
    if yname:
        df[yname] = y

    return df
