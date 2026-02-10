# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import overload, Tuple, Literal, LiteralString, Union, Any

# -------------------------------------------------------------
import numpy as np
from typing import Tuple
from itertools import combinations
from pandas import DataFrame, Series, concat
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from scipy.stats import (
    shapiro,
    normaltest,
    bartlett,
    levene,
    ttest_1samp,    # type: ignore
    ttest_ind as scipy_ttest_ind,
    ttest_rel,      # type: ignore
    wilcoxon,
    pearsonr,
    spearmanr,
    chi2,
    jarque_bera
)

import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.discrete.discrete_model import BinaryResults

from pingouin import anova, pairwise_tukey, welch_anova, pairwise_gameshowell

from .hs_plot import ols_residplot, ols_qqplot, get_default_ax, finalize_plot
from .hs_prep import unmelt
from .hs_util import pretty_table

# ===================================================================
# MCAR(결측치 무작위성) 검정
# ===================================================================
def mcar_test(data: DataFrame, columns: list | str | None = None) -> DataFrame:
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",")]

    cols = data.columns if columns is None else columns
    df = data[cols]

    # 결측치가 있는 컬럼만 사용
    cols_with_na = [c for c in df.columns if df[c].isna().any()]
    if len(cols_with_na) < 2:
        raise ValueError("MCAR 검정은 결측치가 있는 변수가 최소 2개 이상 필요합니다.")

    X = df[cols_with_na].to_numpy()
    n, p = X.shape

    # complete cases로 평균·공분산 추정
    complete = ~np.isnan(X).any(axis=1)
    if complete.sum() < p + 1:
        raise ValueError("완전관측치(complete cases)가 부족하여 MCAR 검정을 수행할 수 없습니다.")

    mu = X[complete].mean(axis=0)
    S = np.cov(X[complete], rowvar=False)
    S_inv = np.linalg.pinv(S)

    chi_sq = 0.0
    dfree = 0

    for i in range(n):
        obs = ~np.isnan(X[i])
        if obs.sum() == p:
            continue  # complete case는 제외
        diff = X[i, obs] - mu[obs]
        S_obs = S[np.ix_(obs, obs)]
        S_obs_inv = np.linalg.pinv(S_obs)

        chi_sq += diff @ S_obs_inv @ diff
        dfree += obs.sum()

    dfree -= p  # Little's adjustment

    p_value = 1 - chi2.cdf(chi_sq, dfree)
    is_mcar = p_value > 0.05

    return DataFrame([{
        "statistic": chi_sq,
        "dof": dfree,
        "p-value": p_value,
        "is_mcar": is_mcar,
        "interpretation": (
            "결측치는 완전 무작위(MCAR)로 판단됨 → 결측치 삭제 가능"
            if is_mcar else
            "결측치는 완전 무작위(MCAR)가 아님 → 삭제 시 편향 가능"
        )
    }])

# ===================================================================
# 결측치 분석 (Missing Values Analysis)
# ===================================================================
def missing_values(data: DataFrame, *fields: str, columns: list | None = None) -> DataFrame:
    """데이터프레임의 결측치 정보를 컬럼 단위로 반환한다.

    각 컬럼의 결측치 수와 전체 대비 비율을 계산하여 데이터프레임으로 반환한다.

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        *fields (str): 분석할 컬럼명 목록. 지정하지 않으면 모든 컬럼을 처리.

    Returns:
        DataFrame: 각 컬럼별 결측치 정보를 포함한 데이터프레임.
            인덱스는 FIELD(컬럼명)이며, 다음 컬럼을 포함:

            - missing_count (int): 결측치의 수
            - missing_rate (float): 전체 행에서 결측치의 비율(%)

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        # 전체 컬럼에 대한 결측치 확인:
        df = DataFrame({'x': [1, 2, None, 4], 'y': [10, None, None, 40]})
        result = hs_stats.missing_values(df)
        print(result)

        # 특정 컬럼만 분석:
        result = hs_stats.missing_values(df, 'x', 'y')
        print(result)
        ```
    """
    if columns is not None:
        if fields:  # type: ignore
            raise ValueError("fields와 columns 인자는 중복 사용할 수 없습니다.")
        fields = columns # type: ignore

    target_fields: list | None = list(fields) if fields else list(data.columns) # type: ignore

    result = []
    for f in target_fields:
        missing_count = data[f].isna().sum()
        missing_rate = (missing_count / len(data)) * 100

        iq = {
            "field": f,
            "missing_count": missing_count,
            "missing_rate": missing_rate
        }

        result.append(iq)

    return DataFrame(result).set_index("field")


# ===================================================================
# 이상치 분석 (Outlier Analysis)
# ===================================================================
def outlier_table(data: DataFrame, *fields: str, columns: list | None = None):
    """데이터프레임의 사분위수와 이상치 경계값, 왜도를 구한다.

    수업에서 사용된 hs_outlier_table() 함수를 개선한 버전    

    Tukey의 방법을 사용하여 각 숫자형 컬럼에 대한 사분위수(Q1, Q2, Q3)와
    이상치 판단을 위한 하한(DOWN)과 상한(UP) 경계값을 계산한다.
    함수 호출 전 상자그림을 통해 이상치가 확인된 필드에 대해서만 처리하는 것이 좋다.

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        *fields (str): 분석할 컬럼명 목록. 지정하지 않으면 모든 숫자형 컬럼을 처리.
        columns (list | None): 분석할 컬럼명 목록. 지정하지 않으면 모든 숫자형 컬럼을 처리.

    Returns:
        DataFrame: 각 필드별 사분위수 및 이상치 경계값을 포함한 데이터프레임.
            인덱스는 FIELD(컬럼명)이며, 다음 컬럼을 포함:

            - q1 (float): 제1사분위수 (25th percentile)
            - q2 (float): 제2사분위수 (중앙값, 50th percentile)
            - q3 (float): 제3사분위수 (75th percentile)
            - iqr (float): 사분위 범위 (q3 - q1)
            - up (float): 이상치 상한 경계값 (q3 + 1.5 * iqr)
            - down (float): 이상치 하한 경계값 (q1 - 1.5 * iqr)
            - min (float): 최소값
            - max (float): 최대값
            - skew (float): 왜도
            - outlier_count (int): 이상치 개수
            - outlier_rate (float): 이상치 비율(%)

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({'x': [1, 2, 3, 100], 'y': [10, 20, 30, 40]})

        # 전체 숫자형 컬럼에 대한 이상치 경계 확인:
        result = hs_stats.outlier_table(df)
        print(result)

        # 특정 컬럼만 분석:
        result = hs_stats.outlier_table(df, 'x', 'y')
        print(result[['q1', 'q3', 'up', 'down']])
        ```

    Notes:
        - DOWN 미만이거나 UP 초과인 값은 이상치(outlier)로 간주됩니다.
        - 숫자형이 아닌 컬럼은 자동으로 제외됩니다.
        - Tukey의 1.5 * IQR 규칙을 사용합니다 (상자그림의 표준 방법).
    """
    if columns is not None:
        if fields:  # type: ignore
            raise ValueError("fields와 columns 인자는 중복 사용할 수 없습니다.")
        fields = columns # type: ignore

    num_columns = list(data.select_dtypes(include=np.number).columns)

    target_fields: list | None = list(fields) if fields else num_columns # type: ignore

    result = []
    for f in target_fields:
        # 숫자 타입이 아니라면 건너뜀
        if f not in num_columns:
            continue

        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q2 = data[f].quantile(q=0.5)
        q3 = data[f].quantile(q=0.75)
        min_value = data[f].min()
        max_value = data[f].max()

        # 이상치 경계 (Tukey's fences)
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        # 왜도
        skew = data[f].skew()

        # 이상치 개수 및 비율
        outlier_count = ((data[f] < down) | (data[f] > up)).sum()
        outlier_rate = (outlier_count / len(data)) * 100

        # 왜도
        skew = data[f].skew()

        # 이상치 개수 및 비율
        outlier_count = ((data[f] < down) | (data[f] > up)).sum()
        outlier_rate = (outlier_count / len(data)) * 100

        # 분포 특성 판정 (왜도 기준)
        abs_skew = abs(skew)    # type: ignore
        if abs_skew < 0.5:      # type: ignore
            dist = "거의 대칭"
        elif abs_skew < 1.0:    # type: ignore
            if skew > 0:        # type: ignore
                dist = "약한 우측 꼬리"
            else:
                dist = "약한 좌측 꼬리"
        elif abs_skew < 2.0:    # type: ignore
            if skew > 0:        # type: ignore
                dist = "중간 우측 꼬리"
            else:
                dist = "중간 좌측 꼬리"
        else:
            if skew > 0:        # type: ignore
                dist = "극단 우측 꼬리"
            else:
                dist = "극단 좌측 꼬리"

        # 로그변환 필요성 판정
        if abs_skew < 0.5:      # type: ignore
            log_need = "낮음"
        elif abs_skew < 1.0:    # type: ignore
            log_need = "중간"
        else:
            log_need = "높음"

        iq = {
            "field": f,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": iqr,
            "up": up,
            "down": down,
            "min": min_value,
            "max": max_value,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "skew": skew,
            "dist": dist,
            "log_need": log_need
        }

        result.append(iq)

    return DataFrame(result).set_index("field")


# ===================================================================
# 확장된 기술통계량 (Extended Descriptive Statistics)
# ===================================================================
def describe(data: DataFrame, *fields: str, columns: list | None = None):
    """데이터프레임의 연속형 변수의 단위 및 현실성을 평가하기 위해 확장된 기술통계량을 반환한다.

    각 연속형(숫자형) 컬럼의 기술통계량(describe)을 구하고, 이에 사분위수 범위(IQR),
    이상치 경계값(UP, DOWN), 왜도(skew), 이상치 개수 및 비율, 분포 특성, 로그변환 필요성을
    추가하여 반환한다.

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        *fields (str): 분석할 컬럼명 목록. 지정하지 않으면 모든 숫자형 컬럼을 처리.
        columns (list, optional): 반환할 통계량 컬럼 목록. None이면 모든 통계량 반환.

    Returns:
        DataFrame: 각 필드별 확장된 기술통계량을 포함한 데이터프레임.
            행은 다음과 같은 통계량을 포함:

            - count (float): 비결측치의 수
            - na_count (int): 결측치의 수
            - na_rate (float): 결측치 비율(%)
            - mean (float): 평균값
            - std (float): 표준편차
            - min (float): 최소값
            - 25% (float): 제1사분위수 (Q1)
            - 50% (float): 제2사분위수 (중앙값)
            - 75% (float): 제3사분위수 (Q3)
            - max (float): 최대값
            - iqr (float): 사분위 범위 (Q3 - Q1)
            - up (float): 이상치 상한 경계값 (Q3 + 1.5 * IQR)
            - down (float): 이상치 하한 경계값 (Q1 - 1.5 * IQR)
            - skew (float): 왜도
            - outlier_count (int): 이상치 개수
            - outlier_rate (float): 이상치 비율(%)
            - dist (str): 분포 특성 ("극단 우측 꼬리", "거의 대칭" 등)
            - log_need (str): 로그변환 필요성 ("높음", "중간", "낮음")

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({
            'x': [1, 2, 3, 4, 5, 100],
            'y': [10, 20, 30, 40, 50, 60],
            'z': ['a', 'b', 'c', 'd', 'e', 'f']
        })

        # 전체 숫자형 컬럼에 대한 확장된 기술통계:
        result = hs_stats.describe(df)
        print(result)

        # 특정 컬럼만 분석:
        result = hs_stats.describe(df, 'x', 'y')
        print(result)
        ```

    Notes:
        - 숫자형이 아닌 컬럼은 자동으로 제외됩니다.
        - 결과는 필드(컬럼)가 행으로, 통계량이 열로 구성됩니다.
        - Tukey의 1.5 * IQR 규칙을 사용하여 이상치를 판정합니다.
        - 분포 특성은 왜도 값으로 판정합니다.
        - 로그변환 필요성은 왜도의 절댓값 크기로 판정합니다.
    """
    if columns is not None:
        if fields:  # type: ignore
            raise ValueError("fields와 columns 인자는 중복 사용할 수 없습니다.")
        fields = columns # type: ignore

    num_columns = list(data.select_dtypes(include=np.number).columns)

    target_fields: list | None = list(fields) if fields else num_columns # type: ignore

    # 기술통계량 구하기
    desc = data[target_fields].describe().T

    # 각 컬럼별 결측치 수(na_count) 추가
    na_counts = data[target_fields].isnull().sum()
    desc.insert(1, 'na_count', na_counts)

    # 결측치 비율(na_rate) 추가
    desc.insert(2, 'na_rate', (na_counts / len(data)) * 100)

    # 추가 통계량 계산
    additional_stats = []
    for f in target_fields: # type: ignore
        # 숫자 타입이 아니라면 건너뜀
        if f not in num_columns:
            continue

        # 사분위수
        q1 = data[f].quantile(q=0.25)
        q3 = data[f].quantile(q=0.75)

        # 이상치 경계 (Tukey's fences)
        iqr = q3 - q1
        down = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        # 왜도
        skew = data[f].skew()

        # 이상치 개수 및 비율
        outlier_count = ((data[f] < down) | (data[f] > up)).sum()
        outlier_rate = (outlier_count / len(data)) * 100

        # 분포 특성 판정 (왜도 기준)
        abs_skew = abs(skew)    # type: ignore
        if abs_skew < 0.5:      # type: ignore
            dist = "거의 대칭"
        elif abs_skew < 1.0:    # type: ignore
            if skew > 0:        # type: ignore
                dist = "약한 우측 꼬리"
            else:
                dist = "약한 좌측 꼬리"
        elif abs_skew < 2.0:    # type: ignore
            if skew > 0:        # type: ignore
                dist = "중간 우측 꼬리"
            else:
                dist = "중간 좌측 꼬리"
        else:
            if skew > 0:        # type: ignore
                dist = "극단 우측 꼬리"
            else:
                dist = "극단 좌측 꼬리"

        # 로그변환 필요성 판정
        if abs_skew < 0.5:      # type: ignore
            log_need = "낮음"
        elif abs_skew < 1.0:    # type: ignore
            log_need = "중간"
        else:
            log_need = "높음"

        additional_stats.append({
            'field': f,
            'iqr': iqr,
            'up': up,
            'down': down,
            'outlier_count': outlier_count,
            'outlier_rate': outlier_rate,
            'skew': skew,
            'dist': dist,
            'log_need': log_need
        })

    additional_df = DataFrame(additional_stats).set_index('field')

    # 결과 병합
    result = concat([desc, additional_df], axis=1)

    return result


# ===================================================================
# 범주형 변수 요약 (Categorical Variable Summary)
# ===================================================================
def category_describe(data: DataFrame, *fields: str, columns: list | None = None):
    """데이터프레임의 명목형(범주형) 변수에 대한 분포 편향을 요약한다.

    각 명목형 컬럼의 최다 범주와 최소 범주의 정보를 요약하여 데이터프레임으로 반환한다.

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        *fields (str): 분석할 컬럼명 목록. 지정하지 않으면 모든 명목형 컬럼을 처리.
        columns (list | None, optional): 분석할 컬럼명 목록. fields와 중복 사용 불가. 기본값은 None.

    Returns:
        tuple[DataFrame, DataFrame]: 각 컬럼별 최다/최소 범주 정보를 포함한 데이터프레임과
            각 범주별 빈도/비율 정보를 포함한 데이터프레임을 튜플로 반환.
            다음 컬럼을 포함:

            - 변수 (str): 컬럼명
            - 최다_범주: 가장 빈도가 높은 범주값
            - 최다_비율(%) (float): 최다 범주의 비율
            - 최소_범주: 가장 빈도가 낮은 범주값
            - 최소_비율(%) (float): 최소 범주의 비율

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({
            'cut': ['Ideal', 'Premium', 'Good', 'Ideal', 'Premium'],
            'color': ['E', 'F', 'G', 'E', 'F'],
            'price': [100, 200, 150, 300, 120]
        })

        # 전체 명목형 컬럼에 대한 요약:
        result, summary = hs_stats.category_describe(df)

        # 특정 컬럼만 분석:
        result, summary = hs_stats.category_describe(df, 'cut', 'color')
        ```

    Notes:
        - 숫자형 컬럼은 자동으로 제외됩니다.
        - NaN 값도 하나의 범주로 포함됩니다.
    """
    # columns 인자가 있으면 fields보다 우선한다.
    if columns is not None:
        if fields:  # type: ignore
            raise ValueError("fields와 columns 인자는 중복 사용할 수 없습니다.")
        fields = columns # type: ignore

    num_columns = list(data.select_dtypes(include=np.number).columns)

    target_fields = list(fields) if fields else columns

    if not target_fields:
        # 명목형(범주형) 컬럼 선택: object, category, bool 타입
        target_fields = data.select_dtypes(include=['object', 'category', 'bool']).columns # type: ignore

    result = []
    summary = []
    for f in target_fields:
        # 숫자형 컬럼은 건너뜀
        if f in num_columns:
            continue

        # 각 범주값의 빈도수 계산 (NaN 포함)
        value_counts = data[f].value_counts(dropna=False)

        # 범주별 빈도/비율 정보 추가 (category_table 기능)
        for category, count in value_counts.items():
            rate = (count / len(data)) * 100
            result.append({
                "변수": f,
                "범주": category,
                "빈도": count,
                "비율(%)": round(rate, 2)
            })

        if len(value_counts) == 0:
            continue

        # 최다/최소 범주 정보 추가 (category_describe 기능)
        max_category = value_counts.index[0]
        max_count = value_counts.iloc[0]
        max_rate = (max_count / len(data)) * 100
        min_category = value_counts.index[-1]
        min_count = value_counts.iloc[-1]
        min_rate = (min_count / len(data)) * 100
        summary.append({
            "변수": f,
            "최다_범주": max_category,
            "최다_비율(%)": round(max_rate, 2),
            "최소_범주": min_category,
            "최소_비율(%)": round(min_rate, 2)
        })

    return DataFrame(result), DataFrame(summary).set_index("변수")

# ===================================================================
# 정규성 검정 (Normal Test)
# ===================================================================
def normal_test(data: DataFrame, columns: list | str | None = None, method: str = "n") -> DataFrame:
    """지정된 컬럼(또는 모든 수치형 컬럼)에 대해 정규성 검정을 수행하고 결과를 DataFrame으로 반환한다.

    정규성 검정의 귀무가설은 "데이터가 정규분포를 따른다"이므로, p-value > 0.05일 때
    귀무가설을 기각하지 않으며 정규성을 충족한다고 해석한다.

    Args:
        data (DataFrame): 검정 대상 데이터를 포함한 데이터프레임.
        columns (list | str | None, optional): 검정 대상 컬럼명.
            - None 또는 빈 리스트: 모든 수치형 컬럼에 대해 검정 수행.
            - 컬럼명 리스트: 지정된 컬럼에 대해서만 검정 수행.
            - 콤마로 구분된 문자열: "A, B, C" 형식으로 컬럼명 지정 가능.
            기본값은 None.
        method (str, optional): 정규성 검정 방법.
            - "n": D'Agostino and Pearson's Omnibus test (표본 크기 20 이상 권장)
            - "s": Shapiro-Wilk test (표본 크기 5000 이하 권장)
            기본값은 "n".

    Returns:
        DataFrame: 각 컬럼의 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - method (str): 사용된 검정 방법
            - column (str): 컬럼명
            - statistic (float): 검정 통계량
            - p-value (float): 유의확률
            - is_normal (bool): 정규성 충족 여부 (p-value > 0.05)

    Raises:
        ValueError: 메서드가 "n" 또는 "s"가 아닐 경우.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        import numpy as np

        df = DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.exponential(2, 100)
        })

        # 모든 수치형 컬럼 검정
        result = hs_stats.normal_test(df, method='n')

        # 특정 컬럼만 검정 (리스트)
        result = hs_stats.normal_test(df, columns=['x'], method='n')

        # 특정 컬럼만 검정 (문자열)
        result = hs_stats.normal_test(df, columns='x, y', method='n')
        ```
    """
    if method not in ["n", "s"]:
        raise ValueError(f"method는 'n' 또는 's'여야 합니다. 입력값: {method}")

    # columns가 문자열인 경우 리스트로 변환
    if isinstance(columns, str):
        columns = [col.strip() for col in columns.split(',')]

    # 컬럼 선택: 지정된 컬럼 또는 모든 수치형 컬럼
    if columns is None or len(columns) == 0:
        # 모든 수치형 컬럼 선택 (bool 제외)
        numeric_df = data.select_dtypes(include=[np.number])
        target_cols = [c for c in numeric_df.columns if not is_bool_dtype(numeric_df[c])]
    else:
        # 지정된 컬럼 사용
        target_cols = columns

    results = []

    for c in target_cols:
        # NaN 값 제거 (통계 검정 수행)
        col_data = data[c].dropna()

        if len(col_data) == 0:
            results.append({
                "method": method,
                "column": c,
                "statistic": np.nan,
                "p-value": np.nan,
                "is_normal": False
            })
            continue

        try:
            if method == "n":
                method_name = "normaltest"
                s, p = normaltest(col_data)
            else:
                method_name = "shapiro"
                s, p = shapiro(col_data)

            results.append({
                "method": method_name,
                "column": c,
                "statistic": s,
                "p-value": p,
                "is_normal": p > 0.05
            })
        except Exception as e:
            # 검정 실패 시 NaN으로 기록
            results.append({
                "method": method,
                "column": c,
                "statistic": np.nan,
                "p-value": np.nan,
                "is_normal": False
            })

    result_df = DataFrame(results)
    return result_df


# ===================================================================
# 등분산성 검정
# ===================================================================
def equal_var_test(data: DataFrame, columns: list | str | None = None, normal_dist: bool | None = None) -> DataFrame:
    """수치형 컬럼들의 분산이 같은지 검정하고 결과를 DataFrame으로 반환한다.

    등분산성 검정의 귀무가설은 "모든 그룹의 분산이 같다"이므로, p-value > 0.05일 때
    귀무가설을 기각하지 않으며 등분산성을 충족한다고 해석한다.

    Args:
        data (DataFrame): 검정 대상 데이터를 포함한 데이터프레임.
        columns (list | str | None, optional): 검정 대상 컬럼명.
            - None 또는 빈 리스트: 모든 수치형 컬럼에 대해 검정 수행.
            - 컬럼명 리스트: 지정된 컬럼에 대해서만 검정 수행.
            - 콤마로 구분된 문자열: "A, B, C" 형식으로 컬럼명 지정 가능.
            기본값은 None.
        normal_dist (bool | None, optional): 등분산성 검정 방법.
            - True: Bartlett 검정 (데이터가 정규분포를 따를 때, 모든 표본이 같은 크기일 때 권장)
            - False: Levene 검정 (정규분포를 따르지 않을 때 더 강건함)
            - None: normal_test()를 이용하여 자동으로 정규성을 판별 후 적절한 검정 방법 선택.
              모든 컬럼이 정규분포를 따르면 Bartlett, 하나라도 따르지 않으면 Levene 사용.
            기본값은 None.

    Returns:
        DataFrame: 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - method (str): 사용된 검정 방법 (Bartlett 또는 Levene)
            - statistic (float): 검정 통계량
            - p-value (float): 유의확률
            - is_equal_var (bool): 등분산성 충족 여부 (p-value > 0.05)
            - n_columns (int): 검정에 사용된 컬럼 수
            - columns (str): 검정에 포함된 컬럼명 (쉼표로 구분)
            - normality_checked (bool): normal_dist가 None이었는지 여부 (자동 판별 사용 여부)

    Raises:
        ValueError: 수치형 컬럼이 2개 미만일 경우 (검정에 최소 2개 필요).

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        import numpy as np

        df = DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100),
            'z': np.random.normal(0, 2, 100)
        })

        # 모든 수치형 컬럼 자동 판별
        result = hs_stats.equal_var_test(df)

        # 특정 컬럼만 검정 (리스트)
        result = hs_stats.equal_var_test(df, columns=['x', 'y'])

        # 특정 컬럼만 검정 (문자열)
        result = hs_stats.equal_var_test(df, columns='x, y')

        # 명시적 지정
        result = hs_stats.equal_var_test(df, normal_dist=True)
        ```
    """
    # columns가 문자열인 경우 리스트로 변환
    if isinstance(columns, str):
        columns = [col.strip() for col in columns.split(',')]

    # 컬럼 선택: 지정된 컬럼 또는 모든 수치형 컬럼
    if columns is None or len(columns) == 0:
        # 모든 수치형 컬럼 선택 (bool 제외)
        numeric_df = data.select_dtypes(include=[np.number])
        numeric_cols = [c for c in numeric_df.columns if not is_bool_dtype(numeric_df[c])]
    else:
        # 지정된 컬럼 사용
        numeric_cols = columns

    if len(numeric_cols) < 2:
        raise ValueError(f"등분산성 검정에는 최소 2개의 수치형 컬럼이 필요합니다. 현재: {len(numeric_cols)}")

    # 각 컬럼별로 NaN을 제거하여 필드 리스트 구성
    fields = []
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            fields.append(col_data)

    if len(fields) < 2:
        raise ValueError("NaN을 제거한 후 최소 2개의 유효한 컬럼이 필요합니다.")

    # normal_dist가 None이면 자동으로 정규성 판별
    normality_checked = False
    if normal_dist is None:
        normality_checked = True
        normality_result = normal_test(data[numeric_cols], method="n")
        # 모든 컬럼이 정규분포를 따르는지 확인
        all_normal = normality_result["is_normal"].all()
        normality_method = normality_result["method"].iloc[0]
        normal_dist = all_normal    # type: ignore

    try:
        if normal_dist:
            method_name = "Bartlett"
            s, p = bartlett(*fields)
        else:
            method_name = "Levene"
            s, p = levene(*fields)

        result_df = DataFrame([{
            "normality_method": normality_method,
            "normality_checked": normal_dist,
            "method": method_name,
            "statistic": s,
            "p-value": p,
            "is_equal_var": p > 0.05,
            "n_columns": len(fields),
            "columns": ", ".join(numeric_cols[:len(fields)])
        }])

        return result_df

    except Exception as e:
        # 검정 실패 시 NaN으로 기록
        method_name = "Bartlett" if normal_dist else "Levene"
        result_df = DataFrame([{
            "method": method_name,
            "statistic": np.nan,
            "p-value": np.nan,
            "is_equal_var": False,
            "n_columns": len(fields),
            "columns": ", ".join(numeric_cols[:len(fields)]),
            "normality_checked": normality_checked
        }])
        return result_df


# ===================================================================
# 일표본 T검정
# ===================================================================
def ttest_1samp(data, mean_value: float = 0.0) -> DataFrame:
    """연속형 데이터에 대해 일표본 t-검정을 수행하고 결과를 반환한다.

    일표본 t-검정은 표본 평균이 특정 값(mean_value)과 같은지를 검정한다.
    귀무가설(H0): 모집단 평균 = mean_value
    대립가설(H1): alternative에 따라 달라짐 (!=, <, >)

    Args:
        data (array-like): 검정 대상 연속형 데이터 (리스트, Series, ndarray 등).
        mean_value (float, optional): 귀무가설의 기준값(비교 대상 평균값).
            기본값은 0.0.

    Returns:
        DataFrame: 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - alternative (str): 대립가설 방향 (two-sided, less, greater)
            - statistic (float): t-통계량
            - p-value (float): 유의확률
            - H0 (bool): 귀무가설 채택 여부 (p-value > 0.05)
            - H1 (bool): 대립가설 채택 여부 (p-value <= 0.05)
            - interpretation (str): 검정 결과 해석 문자열

    Examples:
        ```python
        from hossam import *
        from pandas import Series
        import numpy as np

        # 리스트 데이터로 검정
        data = [5.1, 4.9, 5.3, 5.0, 4.8]
        result = hs_stats.ttest_1samp(data, mean_value=5.0)

        # Series 데이터로 검정
        s = Series(np.random.normal(5, 1, 100))
        result = hs_stats.ttest_1samp(s, mean_value=5)
        ```
    """
    # 데이터를 Series로 변환하고 이름 감지
    if isinstance(data, Series):
        col_data = data.dropna()
    else:
        col_data = Series(data).dropna()

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []

    # 각 대립가설 방향에 대해 t-검정 수행
    for a in alternative:
        try:
            s, p = ttest_1samp(col_data, mean_value, alternative=a) # type: ignore

            itp = None

            if a == "two-sided":
                itp = "μ {0} {1}".format("==" if p > 0.05 else "!=", mean_value)
            elif a == "less":
                itp = "μ {0} {1}".format(">=" if p > 0.05 else "<", mean_value)
            else:
                itp = "μ {0} {1}".format("<=" if p > 0.05 else ">", mean_value)

            result.append({
                "alternative": a,
                "statistic": round(s, 3),
                "p-value": round(p, 4),
                "H0": p > 0.05,
                "H1": p <= 0.05,
                "interpretation": itp,
            })
        except Exception as e:
            result.append({
                "alternative": a,
                "statistic": np.nan,
                "p-value": np.nan,
                "H0": False,
                "H1": False,
                "interpretation": f"검정 실패: {str(e)}"
            })

    rdf = DataFrame(result)
    rdf.set_index(["field", "alternative"], inplace=True)
    return rdf


# ===================================================================
# 독립표본 t-검정 또는 Welch's t-test
# ===================================================================
def ttest_ind(
        data: DataFrame | None = None,
        x : Series | list | np.ndarray | str | None = None,
        y : Series | list | np.ndarray | str | None = None,
        equal_var: bool | None = None
) -> DataFrame:
    """두 독립 집단의 평균 차이를 검정한다 (독립표본 t-검정 또는 Welch's t-test).

    수업에서 사용한 hs_ttest_ind() 함수를 확장한 버전이다.

    독립표본 t-검정은 두 독립된 집단의 평균이 같은지를 검정한다.
    귀무가설(H0): μ1 = μ2 (두 집단의 평균이 같다)

    Args:
        data (DataFrame | None, optional): x와 y가 컬럼명인 경우 사용할 데이터프레임.
            기본값은 None.
        x (Series | list | np.ndarray | str | None, optional): 첫 번째 집단의 데이터 또는
            data가 주어진 경우 연속형 변수의 컬럼명. 기본값은 None.
        y (Series | list | np.ndarray | str | None, optional): 두 번째 집단의 데이터 또는
            data가 주어진 경우 명목형 변수의 컬럼명. 기본값은 None.
        equal_var (bool | None, optional): 등분산성 가정 여부.
            - True: 독립표본 t-검정 (등분산 가정)
            - False: Welch's t-test (등분산 가정하지 않음, 더 강건함)
            - None: equal_var_test()로 자동 판별
            기본값은 None.

    Returns:
        DataFrame: 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - test (str): 사용된 검정 방법
            - alternative (str): 대립가설 방향
            - statistic (float): t-통계량
            - p-value (float): 유의확률
            - H0 (bool): 귀무가설 채택 여부
            - H1 (bool): 대립가설 채택 여부
            - interpretation (str): 검정 결과 해석

    Examples:
        ```python
        from hossam import *
        from pandas import Series, DataFrame
        import numpy as np

        # 리스트로 검정
        group1 = [5.1, 4.9, 5.3, 5.0, 4.8]
        group2 = [5.5, 5.7, 5.4, 5.6, 5.8]
        result = hs_stats.ttest_ind(group1, group2)

        # Series로 검정
        s1 = Series(np.random.normal(5, 1, 100))
        s2 = Series(np.random.normal(5.5, 1, 100))
        result = hs_stats.ttest_ind(s1, s2, equal_var=False)
        ```
    """
    # data가 주어지고 x, y가 컬럼명인 경우 데이터 추출
    if data is not None and isinstance(x, str) and isinstance(y, str):
        df = unmelt(data=data, value_vars=x, id_vars=y)
        x = df[df.columns[0]]
        y = df[df.columns[1]]

    # 데이터를 Series로 변환
    if isinstance(x, Series):
        x_data = x.dropna()
    else:
        x_data = Series(x).dropna()

    if isinstance(y, Series):
        y_data = y.dropna()
    else:
        y_data = Series(y).dropna()

    # 데이터 유효성 검사
    if len(x_data) < 2 or len(y_data) < 2:
        raise ValueError(f"각 집단에 최소 2개 이상의 데이터가 필요합니다. x: {len(x_data)}, y: {len(y_data)}")

    # equal_var가 None이면 자동으로 등분산성 판별
    var_checked = False
    if equal_var is None:
        var_checked = True
        # 두 데이터를 DataFrame으로 구성하여 등분산성 검정
        temp_df = DataFrame({'x': x_data, 'y': y_data})
        var_result = equal_var_test(temp_df)
        normality_method = var_result["normality_method"].iloc[0]
        normality_checked = var_result["normality_checked"].iloc[0]
        equal_var_method = var_result["method"].iloc[0]
        equal_var = var_result["is_equal_var"].iloc[0]

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ(x) {0} μ(y)"

    for a in alternative:
        try:
            s, p = scipy_ttest_ind(x_data, y_data, equal_var=equal_var, alternative=a)  # type: ignore
            n = "t-test_ind" if equal_var else "Welch's t-test"

            # 검정 결과 해석
            itp = None

            if a == "two-sided":
                itp = fmt.format("==" if p > 0.05 else "!=")    # type: ignore
            elif a == "less":
                itp = fmt.format(">=" if p > 0.05 else "<")     # type: ignore
            else:
                itp = fmt.format("<=" if p > 0.05 else ">")     # type: ignore

            result.append({
                "test": n,
                "alternative": a,
                "interpretation": itp,
                normality_method: normality_checked,
                equal_var_method: equal_var,
                n: round(s, 3),   # type: ignore
                "p-value": round(p, 4),     # type: ignore
                "H0": p > 0.05,             # type: ignore
                "H1": p <= 0.05,            # type: ignore
            })
        except Exception as e:
            result.append({
                "test": "t-test_ind" if equal_var else "Welch's t-test",
                "alternative": a,
                "interpretation": f"검정 실패: {str(e)}",
                normality_method: normality_checked,
                equal_var_method: equal_var,
                n: np.nan,
                "p-value": np.nan,
                "H0": False,
                "H1": False
            })

    rdf = DataFrame(result)
    rdf.set_index(["test", "alternative"], inplace=True)
    return rdf


# ===================================================================
# 대응표본 t-검정 또는 Wilcoxon test
# ===================================================================
def ttest_rel(x, y, normality: bool | None = None) -> DataFrame:
    """대응표본 t-검정 또는 Wilcoxon signed-rank test를 수행한다.

    대응표본 t-검정은 동일 개체에서 측정된 두 시점의 평균 차이를 검정한다.
    귀무가설(H0): 두 시점의 평균 차이가 0이다.

    Args:
        x (array-like): 첫 번째 측정값의 연속형 데이터 (리스트, Series, ndarray 등).
        y (array-like): 두 번째 측정값의 연속형 데이터 (리스트, Series, ndarray 등).
        normality (bool | None, optional): 정규성 가정 여부.
            - True: 대응표본 t-검정 (차이의 정규분포 가정)
            - False: Wilcoxon signed-rank test (비모수 검정, 더 강건함)
            - None: 차이의 정규성을 자동으로 검정하여 판별
            기본값은 None.

    Returns:
        DataFrame: 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - test (str): 사용된 검정 방법
            - alternative (str): 대립가설 방향
            - statistic (float): 검정 통계량
            - p-value (float): 유의확률
            - H0 (bool): 귀무가설 채택 여부
            - H1 (bool): 대립가설 채택 여부
            - interpretation (str): 검정 결과 해석

    Examples:
        ```python
        from hossam import *
        from pandas import Series
        import numpy as np

        # 리스트로 검정
        before = [5.1, 4.9, 5.3, 5.0, 4.8]
        after = [5.5, 5.2, 5.7, 5.3, 5.1]
        result = hs_stats.ttest_rel(before, after)

        # Series로 검정
        s1 = Series(np.random.normal(5, 1, 100))
        s2 = Series(np.random.normal(5.3, 1, 100))
        result = hs_stats.ttest_rel(s1, s2, parametric=False)
        ```
    """
    # 데이터를 Series로 변환
    if isinstance(x, Series):
        x_data = x.dropna()
    else:
        x_data = Series(x).dropna()

    if isinstance(y, Series):
        y_data = y.dropna()
    else:
        y_data = Series(y).dropna()

    # 대응표본이므로 같은 길이여야 함
    if len(x_data) != len(y_data):
        raise ValueError(f"대응표본은 같은 길이여야 합니다. x: {len(x_data)}, y: {len(y_data)}")

    # 데이터 유효성 검사
    if len(x_data) < 2:
        raise ValueError(f"최소 2개 이상의 대응 데이터가 필요합니다. 현재: {len(x_data)}")

    # parametric이 None이면 차이의 정규성을 자동으로 검정
    if normality is None:
        tmp_df = DataFrame({'x': x_data, 'y': y_data})
        normality_result = normal_test(tmp_df, method="n")
        # 모든 컬럼이 정규분포를 따르는지 확인
        all_normal = normality_result["is_normal"].all()
        normality_method = normality_result["method"].iloc[0]
        normality = all_normal    # type: ignore

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ(x) {0} μ(y)"

    if normality:
        s, p = ttest_rel(x_data, y_data, alternative=a) # type: ignore
    else:
        # Wilcoxon signed-rank test (대응표본용 비모수 검정)
        n = "Wilcoxon signed-rank"

    for a in alternative:
        try:
            if normality:
                s, p = ttest_rel(x_data, y_data, alternative=a) # type: ignore
            else:
                # Wilcoxon signed-rank test (대응표본용 비모수 검정)
                s, p = wilcoxon(x_data, y_data, alternative=a)

            itp = None

            if a == "two-sided":
                itp = fmt.format("==" if p > 0.05 else "!=")    # type: ignore
            elif a == "less":
                itp = fmt.format(">=" if p > 0.05 else "<")     # type: ignore
            else:
                itp = fmt.format("<=" if p > 0.05 else ">")     # type: ignore

            result.append({
                "test": n,
                "alternative": a,
                normality_method: normality,
                "interpretation": itp,
                "statistic": round(s, 3) if not np.isnan(s) else s, # type: ignore
                "p-value": round(p, 4) if not np.isnan(p) else p,   # type: ignore
                "H0": p > 0.05,     # type: ignore
                "H1": p <= 0.05,    # type: ignore
            })
        except Exception as e:
            result.append({
                "test": n,
                "alternative": a,
                normality_method: normality,
                "interpretation": f"검정 실패: {str(e)}",
                "statistic": np.nan,
                "p-value": np.nan,
                "H0": False,
                "H1": False
            })

    rdf = DataFrame(result)
    rdf.set_index(["test", "alternative"], inplace=True)
    return rdf




# ===================================================================
# 일원 분산분석 (One-way ANOVA)
# ===================================================================
#def oneway_anova(data: DataFrame, dv: str, between: str, alpha: float = 0.05) -> tuple[DataFrame, str, DataFrame | None, str]:
def oneway_anova(data: DataFrame, dv: str, between: str, alpha: float = 0.05, posthoc: bool = False) -> DataFrame | tuple[DataFrame, DataFrame] :
    """일원분산분석(One-way ANOVA)을 일괄 처리한다.

    수업에서 사용된 hs_oneway_anova() 함수를 개선한 버전    

    정규성 및 등분산성 검정을 자동으로 수행한 후,
    그 결과에 따라 적절한 ANOVA 방식을 선택하여 분산분석을 수행한다.
    ANOVA 결과가 유의하면 자동으로 사후검정을 실시한다.

    분석 흐름:
    1. 정규성 검정 (각 그룹별로 normaltest 수행)
    2. 등분산성 검정 (정규성 만족 시 Bartlett, 불만족 시 Levene)
    3. ANOVA 수행 (등분산 만족 시 parametric ANOVA, 불만족 시 Welch's ANOVA)
    4. ANOVA p-value ≤ alpha 일 때 사후검정 (등분산 만족 시 Tukey HSD, 불만족 시 Games-Howell)

    Args:
        data (DataFrame): 분석 대상 데이터프레임. 종속변수와 그룹 변수를 포함해야 함.
        dv (str): 종속변수(Dependent Variable) 컬럼명.
        between (str): 그룹 구분 변수 컬럼명.
        alpha (float, optional): 유의수준. 기본값 0.05.
        posthoc (bool, optional): 사후검정 수행 여부. 기본값 False.

    Returns:
        tuple:
            - anova_df (DataFrame): ANOVA 또는 Welch 결과 테이블(Source, ddof1, ddof2, F, p-unc, np2 등 포함).
            - posthoc_df (DataFrame|None): 사후검정 결과(Tukey HSD 또는 Games-Howell). ANOVA가 유의할 때만 생성.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({
            'score': [5.1, 4.9, 5.3, 5.0, 4.8, 5.5, 5.2, 5.7, 5.3, 5.1],
            'group': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })

        anova_df, posthoc_df = hs_stats.oneway_anova(df, dv='score', between='group')

        # 사후검정결과는 ANOVA가 유의할 때만 생성됨
        if posthoc_df is not None:
            print(posthoc_report)
            print(posthoc_df.head())
        ```

    Raises:
        ValueError: dv 또는 between 컬럼이 데이터프레임에 없을 경우.
    """
    # 컬럼 유효성 검사
    if dv not in data.columns:
        raise ValueError(f"'{dv}' 컬럼이 데이터프레임에 없습니다.")
    if between not in data.columns:
        raise ValueError(f"'{between}' 컬럼이 데이터프레임에 없습니다.")

    df_filtered = data[[dv, between]].dropna()

    # ============================================
    # 1. 정규성 검정 (각 그룹별로 수행)
    # ============================================
    group_names = sorted(df_filtered[between].unique())
    normality_satisfied = True

    for group in group_names:
        group_values = df_filtered[df_filtered[between] == group][dv].dropna()
        if len(group_values) > 0:
            s, p = normaltest(group_values)
            if p <= alpha:
                normality_satisfied = False
                break

    # ============================================
    # 2. 등분산성 검정 (그룹별로 수행)
    # ============================================
    # 각 그룹별로 데이터 분리
    group_data_dict = {}
    for group in group_names:
        group_data_dict[group] = df_filtered[df_filtered[between] == group][dv].dropna().values

    # 등분산 검정 수행
    if len(group_names) > 1:
        if normality_satisfied:
            # 정규성을 만족하면 Bartlett 검정
            s, p = bartlett(*group_data_dict.values())
        else:
            # 정규성을 만족하지 않으면 Levene 검정
            s, p = levene(*group_data_dict.values())
        equal_var_satisfied = p > alpha
    else:
        # 그룹이 1개인 경우 등분산성 검정 불가능
        equal_var_satisfied = True

    # ============================================
    # 3. ANOVA 수행
    # ============================================
    anova_df: DataFrame
    anova_method: str

    if equal_var_satisfied:
        # 등분산을 만족할 때 일반적인 ANOVA 사용
        anova_method = "ANOVA"
        anova_df = anova(data=df_filtered, dv=dv, between=between)
        en = "Bartlett"
    else:
        # 등분산을 만족하지 않을 때 Welch's ANOVA 사용
        anova_method = "Welch"
        anova_df = welch_anova(data=df_filtered, dv=dv, between=between)
        en = "Levene"

    # ANOVA 결과에 메타정보 추가
    anova_df.insert(1, 'normality', normality_satisfied)
    anova_df.insert(2, en, equal_var_satisfied)
    anova_df[anova_method] = anova_df['p-unc'] <= alpha if 'p-unc' in anova_df.columns else False  # type: ignore

    if posthoc == False:
        return anova_df

    # ANOVA 결과가 유의한지 확인
    p_unc = float(anova_df.loc[0, 'p-unc']) # type: ignore
    anova_significant = p_unc <= alpha

    # ANOVA 보고 문장 생성
    # def _safe_get(col: str, default: float = np.nan) -> float:
    #     try:
    #         return float(anova_df.loc[0, col]) if col in anova_df.columns else default  # type: ignore
    #     except Exception:
    #         return default

    # df1 = _safe_get('ddof1')
    # df2 = _safe_get('ddof2')
    # fval = _safe_get('F')
    # eta2 = _safe_get('np2')

    # anova_sig_text = "그룹별 평균이 다를 가능성이 높습니다." if anova_significant else "그룹별 평균 차이에 대한 근거가 부족합니다."
    # assumption_text = f"정규성은 {'대체로 만족' if normality_satisfied else '충족되지 않았고'}, 등분산성은 {'충족되었다' if equal_var_satisfied else '충족되지 않았다'}고 판단됩니다."

    # anova_report = (
    #     f"{between}별로 {dv} 평균을 비교한 {anova_method} 결과: F({df1:.3f}, {df2:.3f}) = {fval:.3f}, p = {p_unc:.4f}. "
    #     f"해석: {anova_sig_text} {assumption_text}"
    # )

    # if not np.isnan(eta2):
    #     anova_report += f" 효과 크기(η²p) ≈ {eta2:.3f}, 값이 클수록 그룹 차이가 뚜렷함을 의미합니다."

    # ============================================
    # 4. 사후검정 (ANOVA 유의할 때만)
    # ============================================
    posthoc_df: DataFrame
    posthoc_method: str
    #posthoc_report = "ANOVA 결과가 유의하지 않아 사후검정을 진행하지 않았습니다."

    if anova_significant:
        if equal_var_satisfied:
            # 등분산을 만족하면 Tukey HSD 사용
            posthoc_method = "Tukey HSD"
            posthoc_df = pairwise_tukey(data=df_filtered, dv=dv, between=between)
        else:
            # 등분산을 만족하지 않으면 Games-Howell 사용
            posthoc_method = "Games-Howell"
            posthoc_df = pairwise_gameshowell(df_filtered, dv=dv, between=between)

        # 사후검정 결과에 메타정보 추가
        # posthoc_df.insert(0, 'normality', normality_satisfied)
        # posthoc_df.insert(1, 'equal_var', equal_var_satisfied)
        posthoc_df.insert(0, 'method', posthoc_method)  # type: ignore

        # p-value 컬럼 탐색
        p_cols = [c for c in ["p-tukey", "pval", "p-adjust", "p_adj", "p-corr", "p", "p-unc", "pvalue", "p_value"] if c in posthoc_df.columns]  # type: ignore
        p_col = p_cols[0] if p_cols else None
        
        # 유의성 여부 컬럼 추가
        posthoc_df['significant'] = posthoc_df[p_col] <= alpha  if p_col else False # type: ignore

        # if p_col:
        #     sig_pairs_df = posthoc_df[posthoc_df[p_col] <= alpha]
        #     sig_count = len(sig_pairs_df)
        #     total_count = len(posthoc_df)
        #     pair_samples = []
        #     if not sig_pairs_df.empty and {'A', 'B'}.issubset(sig_pairs_df.columns):
        #         pair_samples = [f"{row['A']} vs {row['B']}" for _, row in sig_pairs_df.head(3).iterrows()]

        #     if sig_count > 0:
        #         posthoc_report = (
        #             f"{posthoc_method} 사후검정에서 {sig_count}/{total_count}쌍이 의미 있는 차이를 보였습니다 (alpha={alpha})."
        #         )
        #         if pair_samples:
        #             posthoc_report += " 예: " + ", ".join(pair_samples) + " 등."
        #     else:
        #         posthoc_report = f"{posthoc_method} 사후검정에서 추가로 유의한 쌍은 발견되지 않았습니다."
        # else:
        #     posthoc_report = f"{posthoc_method} 결과는 생성했지만 p-value 정보를 찾지 못해 유의성을 확인할 수 없습니다."

    # ============================================
    # 5. 결과 반환
    # ============================================
    #return anova_df, anova_report, posthoc_df, posthoc_report
    return anova_df, posthoc_df


# ===================================================================
# 이원 분산분석 (Two-way ANOVA: 두 범주형 독립변수)
# ===================================================================
def twoway_anova(
    data: DataFrame,
    dv: str,
    factor_a: str,
    factor_b: str,
    alpha: float = 0.05,
) -> tuple[DataFrame, str, DataFrame | None, str]:
    """두 범주형 요인에 대한 이원분산분석을 수행하고 해석용 보고문을 반환한다.

    분석 흐름:
    1) 각 셀(요인 조합)별 정규성 검정
    2) 전체 셀을 대상으로 등분산성 검정 (정규성 충족 시 Bartlett, 불충족 시 Levene)
    3) 두 요인 및 교호작용을 포함한 2원 ANOVA 수행
    4) 유의한 요인에 대해 Tukey HSD 사후검정(요인별) 실행

    Args:
        data (DataFrame): 종속변수와 두 개의 범주형 요인을 포함한 데이터프레임.
        dv (str): 종속변수 컬럼명.
        factor_a (str): 첫 번째 요인 컬럼명.
        factor_b (str): 두 번째 요인 컬럼명.
        alpha (float, optional): 유의수준. 기본 0.05.

    Returns:
        tuple:
            - anova_df (DataFrame): 2원 ANOVA 결과(각 요인과 상호작용의 F, p, η²p 포함).
            - anova_report (str): 두 요인 및 상호작용의 유의성/가정 충족 여부를 요약한 문장.
            - posthoc_df (DataFrame|None): 유의한 요인에 대한 Tukey 사후검정 결과(요인명, A, B, p 포함). 없으면 None.
            - posthoc_report (str): 사후검정 유무 및 유의 쌍 요약 문장.

    Raises:
        ValueError: 입력 컬럼이 데이터프레임에 없을 때.
    """
    # 컬럼 유효성 검사
    for col in [dv, factor_a, factor_b]:
        if col not in data.columns:
            raise ValueError(f"'{col}' 컬럼이 데이터프레임에 없습니다.")

    df_filtered = data[[dv, factor_a, factor_b]].dropna()

    # 1) 셀별 정규성 검정
    normality_satisfied = True
    for (a, b), subset in df_filtered.groupby([factor_a, factor_b], observed=False):
        vals = subset[dv].dropna()
        if len(vals) > 0:
            _, p = normaltest(vals)
            if p <= alpha:
                normality_satisfied = False
                break

    # 2) 등분산성 검정 (셀 단위)
    cell_values = [g[dv].dropna().values for _, g in df_filtered.groupby([factor_a, factor_b], observed=False)]
    if len(cell_values) > 1:
        if normality_satisfied:
            _, p_var = bartlett(*cell_values)
        else:
            _, p_var = levene(*cell_values)
        equal_var_satisfied = p_var > alpha
    else:
        equal_var_satisfied = True

    # 3) 2원 ANOVA 수행 (pingouin anova with between factors)
    anova_df = anova(data=df_filtered, dv=dv, between=[factor_a, factor_b], effsize="np2")
    anova_df.insert(0, "normality", normality_satisfied)
    anova_df.insert(1, "equal_var", equal_var_satisfied)
    if 'p-unc' in anova_df.columns:
        anova_df['significant'] = anova_df['p-unc'] <= alpha

    # 보고문 생성
    def _safe(row, col, default=np.nan):
        try:
            return float(row[col])
        except Exception:
            return default

    # 요인별 문장
    reports = []
    sig_flags = {}
    for _, row in anova_df.iterrows():
        term = row.get("Source", "")
        fval = _safe(row, "F")
        pval = _safe(row, "p-unc")
        eta2 = _safe(row, "np2")
        sig = pval <= alpha
        sig_flags[term] = sig
        if term.lower() == "residual":
            continue
        effect_name = term.replace("*", "와 ")
        msg = f"{effect_name}: F={fval:.3f}, p={pval:.4f}. 해석: "
        msg += "유의한 차이가 있습니다." if sig else "유의한 차이를 찾지 못했습니다."
        if not np.isnan(eta2):
            msg += f" 효과 크기(η²p)≈{eta2:.3f}."
        reports.append(msg)

    assumption_text = f"정규성은 {'대체로 만족' if normality_satisfied else '충족되지 않음'}, 등분산성은 {'충족' if equal_var_satisfied else '충족되지 않음'}으로 판단했습니다."
    anova_report = " ".join(reports) + " " + assumption_text

    # 4) 사후검정: 유의한 요인(교호작용 제외) 대상, 수준이 2 초과일 때만 실행
    posthoc_df_list = []
    interaction_name = f"{factor_a}*{factor_b}".lower()
    interaction_name_spaced = f"{factor_a} * {factor_b}".lower()

    for factor, sig in sig_flags.items():
        if factor is None:
            continue
        factor_lower = str(factor).lower()

        # 교호작용(residual 포함) 혹은 비유의 항은 건너뛴다
        if factor_lower in ["residual", interaction_name, interaction_name_spaced] or not sig:
            continue

        # 실제 컬럼이 아니면 건너뛴다 (ex: "A * B" 같은 교호작용 이름)
        if factor not in df_filtered.columns:
            continue

        levels = df_filtered[factor].unique()
        if len(levels) <= 2:
            continue
        tukey_df = pairwise_tukey(data=df_filtered, dv=dv, between=factor)
        tukey_df.insert(0, "factor", factor)
        posthoc_df_list.append(tukey_df)

    posthoc_df = None
    posthoc_report = "사후검정이 필요하지 않거나 유의한 요인이 없습니다."
    if posthoc_df_list:
        posthoc_df = concat(posthoc_df_list, ignore_index=True)
        p_cols = [c for c in ["p-tukey", "pval", "p-adjust", "p_adj", "p-corr", "p", "p-unc", "pvalue", "p_value"] if c in posthoc_df.columns]
        p_col = p_cols[0] if p_cols else None
        if p_col:
            posthoc_df['significant'] = posthoc_df[p_col] <= alpha
            sig_df = posthoc_df[posthoc_df[p_col] <= alpha]
            sig_count = len(sig_df)
            total_count = len(posthoc_df)
            examples = []
            if not sig_df.empty and {"A", "B"}.issubset(sig_df.columns):
                examples = [f"{row['A']} vs {row['B']}" for _, row in sig_df.head(3).iterrows()]
            if sig_count > 0:
                posthoc_report = f"사후검정(Tukey)에서 {sig_count}/{total_count}쌍이 의미 있는 차이를 보였습니다."
                if examples:
                    posthoc_report += " 예: " + ", ".join(examples) + " 등."
            else:
                posthoc_report = "사후검정 결과 추가로 유의한 쌍은 없었습니다."
        else:
            posthoc_report = "사후검정 결과를 생성했으나 p-value 정보를 찾지 못했습니다."

    return anova_df, anova_report, posthoc_df, posthoc_report



# ===================================================================
# 종속변수에 대한 편상관계수 및 효과크기 분석 (Correlation & Effect Size)
# ===================================================================
def corr_effect_size(data: DataFrame, dv: str, *fields: str, alpha: float = 0.05) -> DataFrame:
    """종속변수와의 편상관계수 및 효과크기를 계산한다.

    각 독립변수와 종속변수 간의 상관계수를 계산하되, 정규성과 선형성을 검사하여
    Pearson 또는 Spearman 상관계수를 적절히 선택한다.
    Cohen's d (효과크기)를 계산하여 상관 강도를 정량화한다.

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        dv (str): 종속변수 컬럼 이름.
        *fields (str): 독립변수 컬럼 이름들. 지정하지 않으면 수치형 컬럼 중 dv 제외 모두 사용.
        alpha (float, optional): 유의수준. 기본 0.05.

    Returns:
        DataFrame: 다음 컬럼을 포함한 데이터프레임:
            - Variable (str): 독립변수 이름
            - Correlation (float): 상관계수 (Pearson 또는 Spearman)
            - Corr_Type (str): 선택된 상관계수 종류 ('Pearson' 또는 'Spearman')
            - P-value (float): 상관계수의 유의확률
            - Cohens_d (float): 표준화된 효과크기
            - Effect_Size (str): 효과크기 분류 ('Large', 'Medium', 'Small', 'Negligible')

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({'age': [20, 30, 40, 50],
                   'bmi': [22, 25, 28, 30],
                   'charges': [1000, 2000, 3000, 4000]})

        result = hs_stats.corr_effect_size(df, 'charges', 'age', 'bmi')
        ```
    """

    # fields가 지정되지 않으면 수치형 컬럼 중 dv 제외 모두 사용
    if not fields:
        fields = [col for col in data.columns if is_numeric_dtype(data[col]) and col != dv] # type: ignore

    # dv가 수치형인지 확인
    if not is_numeric_dtype(data[dv]):
        raise ValueError(f"Dependent variable '{dv}' must be numeric type")

    results = []

    for var in fields:
        if not is_numeric_dtype(data[var]):
            continue

        # 결측치 제거
        valid_idx = data[[var, dv]].notna().all(axis=1)
        x = data.loc[valid_idx, var].values
        y = data.loc[valid_idx, dv].values

        if len(x) < 3:
            continue

        # 정규성 검사 (Shapiro-Wilk: n <= 5000 권장, 그 외 D'Agostino)
        method_x = 's' if len(x) <= 5000 else 'n'
        method_y = 's' if len(y) <= 5000 else 'n'

        normal_x_result = normal_test(data[[var]], columns=[var], method=method_x)
        normal_y_result = normal_test(data[[dv]], columns=[dv], method=method_y)

        # 정규성 판정 (p > alpha면 정규분포 가정)
        normal_x = normal_x_result.loc[var, 'p-val'] > alpha if var in normal_x_result.index else False     # type: ignore
        normal_y = normal_y_result.loc[dv, 'p-val'] > alpha if dv in normal_y_result.index else False   # type: ignore

        # Pearson (모두 정규) vs Spearman (하나라도 비정규)
        if normal_x and normal_y:
            r, p = pearsonr(x, y)
            corr_type = 'Pearson'
        else:
            r, p = spearmanr(x, y)
            corr_type = 'Spearman'

        # Cohen's d 계산 (상관계수에서 효과크기로 변환)
        # d = 2*r / sqrt(1-r^2)
        if r ** 2 < 1:    # type: ignore
            d = (2 * r) / np.sqrt(1 - r ** 2) # type: ignore
        else:
            d = 0

        # 효과크기 분류 (Cohen's d 기준)
        # Small: 0.2 < |d| <= 0.5
        # Medium: 0.5 < |d| <= 0.8
        # Large: |d| > 0.8
        abs_d = abs(d)
        if abs_d > 0.8:
            effect_size = 'Large'
        elif abs_d > 0.5:
            effect_size = 'Medium'
        elif abs_d > 0.2:
            effect_size = 'Small'
        else:
            effect_size = 'Negligible'

        results.append({
            'Variable': var,
            'Correlation': r,
            'Corr_Type': corr_type,
            'P-value': p,
            'Cohens_d': d,
            'Effect_Size': effect_size
        })

    result_df = DataFrame(results)

    # 상관계수로 정렬 (절댓값 기준 내림차순)
    if len(result_df) > 0:
        result_df = result_df.sort_values('Correlation', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

    return result_df


# ===================================================================
# 쌍별 상관분석 (선형성/이상치 점검 후 Pearson/Spearman 자동 선택)
# ===================================================================
def corr_pairwise(
    data: DataFrame,
    fields: list[str] | None = None,
    alpha: float = 0.05,
    z_thresh: float = 3.0,
    min_n: int = 8,
    #linearity_power: tuple[int, ...] = (2,)
) -> tuple[DataFrame, DataFrame]:
    """각 변수 쌍에 대해 선형성·이상치 여부를 점검한 뒤 Pearson/Spearman을 자동 선택해 상관을 요약한다.

    절차:
    1) z-score 기준(|z|>z_thresh)으로 각 변수의 이상치 존재 여부를 파악
    2) 단순회귀 y~x에 대해 Ramsey RESET(linearity_power)로 선형성 검정 (모든 p>alpha → 선형성 충족)
    3) 선형성 충족이고 양쪽 변수에서 |z|>z_thresh 이상치가 없으면 Pearson, 그 외엔 Spearman 선택
    4) 상관계수/유의확률, 유의성 여부, 강도(strong/medium/weak/no correlation) 기록

    Args:
        data (DataFrame): 분석 대상 데이터프레임.
        fields (list[str]|None): 분석할 숫자형 컬럼 이름 리스트. None이면 모든 숫자형 컬럼 사용. 기본값 None.
        alpha (float, optional): 유의수준. 기본 0.05.
        z_thresh (float, optional): 이상치 판단 임계값(|z| 기준). 기본 3.0.
        min_n (int, optional): 쌍별 최소 표본 크기. 미만이면 계산 생략. 기본 8.
        #linearity_power (tuple[int,...], optional): RESET 검정에서 포함할 차수 집합. 기본 (2,).

    Returns:
        tuple[DataFrame, DataFrame]: 두 개의 데이터프레임을 반환.
            [0] result_df: 각 변수쌍별 결과 테이블. 컬럼:
                var_a, var_b, n, linearity(bool), outlier_flag(bool), chosen('pearson'|'spearman'),
                corr, pval, significant(bool), strength(str)
            [1] corr_matrix: 상관계수 행렬 (행과 열에 변수명, 값에 상관계수)

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame

        df = DataFrame({'x1': [1,2,3,4,5], 'x2': [2,4,5,4,6], 'x3': [10,20,25,24,30]})
        # 전체 숫자형 컬럼에 대해 상관분석
        result_df, corr_matrix = hs_stats.corr_pairwise(df)
        # 특정 컬럼만 분석
        result_df, corr_matrix = hs_stats.corr_pairwise(df, fields=['x1', 'x2'])
        ```
    """

    # 0) 컬럼 선정 (숫자형만)
    if fields is None:
        # None이면 모든 숫자형 컬럼 사용
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # fields 리스트에서 데이터에 있는 것만 선택하되, 숫자형만 필터링
        cols = [c for c in fields if c in data.columns and is_numeric_dtype(data[c])]

    # 사용 가능한 컬럼이 2개 미만이면 상관분석 불가능
    if len(cols) < 2:
        empty_df = DataFrame(columns=["var_a", "var_b", "n", "linearity", "outlier_flag", "chosen", "corr", "pval", "significant", "strength"])
        return empty_df, DataFrame()

    # z-score 기반 이상치 유무 계산
    z_outlier_flags = {}
    for c in cols:
        col = data[c].dropna()
        if col.std(ddof=1) == 0:
            z_outlier_flags[c] = False
            continue
        z = (col - col.mean()) / col.std(ddof=1)
        z_outlier_flags[c] = (z.abs() > z_thresh).any()

    rows = []

    for a, b in combinations(cols, 2):
        # 공통 관측치 사용
        pair_df = data[[a, b]].dropna()
        if len(pair_df) < min_n:
            # 표본이 너무 적으면 계산하지 않음
            rows.append(
                {
                    "var_a": a,
                    "var_b": b,
                    "n": len(pair_df),
                    "linearity": False,
                    "outlier_flag": True,
                    "chosen": None,
                    "corr": np.nan,
                    "pval": np.nan,
                    "significant": False,
                    "strength": "no correlation",
                }
            )
            continue

        x = pair_df[a]
        y = pair_df[b]

        # 상수열/분산 0 체크 → 상관계수 계산 불가
        if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
            rows.append(
                {
                    "var_a": a,
                    "var_b": b,
                    "n": len(pair_df),
                    "linearity": False,
                    "outlier_flag": True,
                    "chosen": None,
                    "corr": np.nan,
                    "pval": np.nan,
                    "significant": False,
                    "strength": "no correlation",
                }
            )
            continue

        # 1) 선형성: Ramsey RESET (지정 차수 전부 p>alpha 여야 통과)
        linearity_ok = False
        try:
            X_const = sm.add_constant(x)
            model = sm.OLS(y, X_const).fit()
            # pvals = []
            # for pwr in linearity_power:
            #     reset = linear_reset(model, power=pwr, use_f=True)
            #     pvals.append(reset.pvalue)
            # # 모든 차수에서 유의하지 않을 때 선형성 충족으로 간주
            # if len(pvals) > 0:
            #     linearity_ok = all([pv > alpha for pv in pvals])

            reset = linear_reset(model)
            linearity_ok = reset.pvalue > alpha
        except Exception:
            linearity_ok = False

        # 2) 이상치 플래그 (두 변수 중 하나라도 z-outlier 있으면 True)
        outlier_flag = bool(z_outlier_flags.get(a, False) or z_outlier_flags.get(b, False))

        # 3) 상관 계산: 선형·무이상치면 Pearson, 아니면 Spearman
        try:
            if linearity_ok and not outlier_flag:
                chosen = "pearson"
                corr_val, pval = pearsonr(x, y)
            else:
                chosen = "spearman"
                corr_val, pval = spearmanr(x, y)
        except Exception:
            chosen = None
            corr_val, pval = np.nan, np.nan

        # 4) 유의성, 강도
        significant = False if np.isnan(pval) else pval <= alpha    # type: ignore
        abs_r = abs(corr_val) if not np.isnan(corr_val) else 0      # type: ignore
        if abs_r > 0.7:
            strength = "strong"
        elif abs_r > 0.3:
            strength = "medium"
        elif abs_r > 0:
            strength = "weak"
        else:
            strength = "no correlation"

        rows.append(
            {
                "var_a": a,
                "var_b": b,
                "n": len(pair_df),
                "linearity": linearity_ok,
                "outlier_flag": outlier_flag,
                "chosen": chosen,
                "corr": corr_val,
                "pval": pval,
                "significant": significant,
                "strength": strength,
            }
        )

    result_df = DataFrame(rows)


    # 5) 상관행렬 생성 (result_df 기반)
    # 모든 변수를 행과 열로 하는 대칭 행렬 생성
    corr_matrix = DataFrame(np.nan, index=cols, columns=cols)
    # 대각선: 1 (자기상관)
    for c in cols:
        corr_matrix.loc[c, c] = 1.0
    # 쌍별 상관계수 채우기 (대칭성 유지)
    if not result_df.empty:
        for _, row in result_df.iterrows():
            a, b, corr_val = row["var_a"], row["var_b"], row["corr"]
            corr_matrix.loc[a, b] = corr_val
            corr_matrix.loc[b, a] = corr_val  # 대칭성

    return result_df, corr_matrix



# ===================================================================
# 독립변수간 다중공선성 제거
# ===================================================================
def vif_filter(
    data: DataFrame,
    yname: str | None = None,
    ignore: list | None = None,
    threshold: float = 10.0,
    verbose: bool = False,
) -> DataFrame:
    """독립변수 간 다중공선성을 검사하여 VIF가 threshold 이상인 변수를 반복적으로 제거한다.

    Args:
        data (DataFrame): 데이터프레임
        yname (str, optional): 종속변수 컬럼명. Defaults to None.
        ignore (list | None, optional): 제외할 컬럼 목록. Defaults to None.
        threshold (float, optional): VIF 임계값. Defaults to 10.0.
        verbose (bool, optional): True일 경우 각 단계의 VIF를 출력한다. Defaults to False.

    Returns:
        DataFrame: VIF가 threshold 이하인 변수만 남은 데이터프레임 (원본 컬럼 순서 유지)

    Examples:
        ```python
        # 기본 사용 예
        from hossam import *
        filtered = hs_stats.vif_filter(df, yname="target", ignore=["id"], threshold=10.0)
        ```
    """

    df = data.copy()

    # y 분리 (있다면)
    y = None
    if yname and yname in df.columns:
        y = df[yname]
        df = df.drop(columns=[yname])

    # 제외할 목록 정리
    ignore = ignore or []
    ignore_cols_present = [c for c in ignore if c in df.columns]

    # VIF 대상 수치형 컬럼 선택 (bool은 연속형이 아니므로 제외)
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = [c for c in numeric_df.columns if not is_bool_dtype(numeric_df[c])]

    # VIF 대상 X 구성 (수치형에서 제외 목록 제거)
    X = df[numeric_cols]
    if ignore_cols_present:
        X = X.drop(columns=ignore_cols_present, errors="ignore")

    # 수치형 변수가 없으면 바로 반환
    if X.shape[1] == 0:
        result = data.copy()
        return result

    def _compute_vifs(X_: DataFrame, verbose: bool = False) -> DataFrame:
        # NA 제거 후 상수항 추가
        X_clean = X_.dropna()
        
        if X_clean.shape[0] == 0:
            # 데이터가 모두 NA인 경우 VIF 계산 불가: NaN 반환
            return DataFrame({col: [np.nan] for col in X_.columns})
        
        if X_clean.shape[1] == 1:
            # 단일 예측변수의 경우 다른 설명변수가 없으므로 VIF는 1로 간주
            return DataFrame({col: [1.0] for col in X_clean.columns})
        
        exog = sm.add_constant(X_clean, prepend=True)
        vifs = {}
        
        for i, col in enumerate(X_clean.columns, start=0):
            # exog의 첫 열은 상수항이므로 변수 인덱스는 +1
            try:
                vifs[col] = float(variance_inflation_factor(exog.values, i + 1))# type: ignore
            except Exception:
                # 계산 실패 시 무한대로 처리하여 우선 제거 대상으로
                vifs[col] = float("inf")
        
        vdf = DataFrame(list(vifs.items()), columns=["Variable", "VIF"])
        vdf.sort_values("VIF", ascending=False, inplace=True)

        if verbose:
            pretty_table(vdf)  # type: ignore
            print()

        return vdf

    # 반복 제거 루프
    i = 0
    while True:
        if X.shape[1] == 0:
            break

        print(f"📇 VIF 제거 반복 {i+1}회차\n")
        vifs = _compute_vifs(X, verbose=verbose)
        
        # 모든 변수가 임계값 이하이면 종료
        max_vif = vifs.iloc[0]["VIF"]
        max_key = vifs.iloc[0]["Variable"]

        if np.isnan(max_vif) or max_vif <= threshold:
            if i == 0:
                print("▶ 모든 변수의 VIF가 임계값 이하입니다. 제거할 변수가 없습니다.\n")
            else:
                print("▶ 모든 변수의 VIF가 임계값 이하가 되어 종료합니다. 제거된 변수 {0}개\n".format(i))
            break

        # 가장 큰 VIF 변수 제거
        X = X.drop(columns=[max_key])
        print(f"제거된 변수: {max_key} (VIF={max_vif:.2f})")
        i += 1

    # 원본 컬럼 순서 유지하며 제거된 수치형 컬럼만 제외
    kept_numeric_cols = list(X.columns)
    removed_numeric_cols = [c for c in numeric_cols if c not in kept_numeric_cols]
    result = data.drop(columns=removed_numeric_cols, errors="ignore")

    return result



# ===================================================================
# x, y 데이터에 대한 추세선을 구한다.
# ===================================================================
def trend(x: Any, y: Any, degree: int = 1, value_count: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """x, y 데이터에 대한 추세선을 구한다.

    Args:
        x (_type_): 산점도 그래프에 대한 x 데이터
        y (_type_): 산점도 그래프에 대한 y 데이터
        degree (int, optional): 추세선 방정식의 차수. Defaults to 1.
        value_count (int, optional): x 데이터의 범위 안에서 간격 수. Defaults to 100.

    Returns:
        tuple: (v_trend, t_trend)

    Examples:
        ```python
        # 2차 다항 회귀 추세선
        from hossam import *
        vx, vy = hs_stats.trend(x, y, degree=2, value_count=200)
        print(len(vx), len(vy)) # 200, 200
        ```
    """
    # [ a, b, c ] ==> ax^2 + bx + c
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim == 0 or y_arr.ndim == 0:
        raise ValueError("x, y는 1차원 이상의 배열이어야 합니다.")

    coeff = np.polyfit(x_arr, y_arr, degree)

    minx = np.min(x_arr)
    maxx = np.max(x_arr)
    v_trend = np.linspace(minx, maxx, value_count)

    # np.polyval 사용으로 간결하게 추세선 계산
    t_trend = np.polyval(coeff, v_trend)

    return (v_trend, t_trend)


# ===================================================================
# 선형회귀 요약 리포트
# ===================================================================
@overload
def ols_report(
    fit: RegressionResultsWrapper,
    data: DataFrame,
    full: Literal[False],
    alpha: float = 0.05
) -> tuple[DataFrame, DataFrame]: ...

@overload
def ols_report(
    fit: RegressionResultsWrapper,
    data: DataFrame,
    full: Literal[True],
    alpha: float = 0.05
) -> tuple[
    DataFrame,
    DataFrame,
    str,
    LiteralString,
    list[str],
    str
]: ...

def ols_report(
    fit: RegressionResultsWrapper,
    data: DataFrame,
    full: bool = False,
    alpha: float = 0.05,
    logvar: bool = False
    ):
    """선형회귀 적합 결과를 요약 리포트로 변환한다.

    Args:
        fit (RegressionResultsWrapper): statsmodels OLS 등 선형회귀 결과 객체 (`fit.summary()`를 지원해야 함).
        data (DataFrame): 종속변수와 독립변수를 모두 포함한 DataFrame.
        full (bool): True이면 6개 값 반환, False이면 회귀계수 테이블(rdf)만 반환. 기본값 True.
        alpha (float): 유의수준. 기본값 0.05.
        logvar (bool): 종속변수에 로그가 적용되어 있는지 여부 (True일 때 독립변수 해석에 로그를 반영함)

    Returns:
        tuple: full=True일 때 다음 요소를 포함한다.
            - 성능 지표 표 (`pdf`, DataFrame): R, R², Adj. R², F, p-value, Durbin-Watson.
            - 회귀계수 표 (`rdf`, DataFrame): 변수별 B, 표준오차, Beta, t, p-value, significant, 공차, VIF.
            - 적합도 요약 (`result_report`, str): R, R², F, p-value, Durbin-Watson 등 핵심 지표 문자열.
            - 모형 보고 문장 (`model_report`, str): F-검정 유의성에 기반한 서술형 문장.
            - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수에 대한 서술형 문장.
            - 회귀식 문자열 (`equation_text`, str): 상수항과 계수를 포함한 회귀식 표현.

        full=False일 때:
            - 성능 지표 표 (`pdf`, DataFrame): R, R², Adj. R², F, p-value, Durbin-Watson.
            - 회귀계수 표 (`rdf`, DataFrame)

    Examples:
        ```python
        from hossam import *

        df = hs_util.load_data("some_data.csv")
        fit = hs_stats.ols(df, yname="target")

        # 전체 리포트
        pdf, rdf, result_report, model_report, variable_reports, eq = hs_stats.ols_report(fit, data, full=True)

        # 간단한 버전 (성능지표, 회귀계수 테이블만)
        pdf, rdf = hs_stats.ols_report(fit, data)
        ```
    """

    # summary2() 결과에서 실제 회귀계수 DataFrame 추출
    summary_obj = fit.summary2()
    tbl = summary_obj.tables[1]  # 회귀계수 테이블은 tables[1]에 위치

    # 종속변수 이름
    yname = fit.model.endog_names

    # 독립변수 이름(상수항 제외)
    xnames = [n for n in fit.model.exog_names if n != "const"]

    # 독립변수 부분 데이터 (VIF 계산용)
    indi_df = data.filter(xnames)

    # 독립변수 결과를 누적
    variables = []

    # VIF 계산 (상수항 포함 설계행렬 사용)
    vif_dict = {}
    indi_df_const = sm.add_constant(indi_df, has_constant="add")
    for i, col in enumerate(indi_df.columns, start=1):  # 상수항이 0이므로 1부터 시작
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                vif_value = variance_inflation_factor(indi_df_const.values, i)  # type: ignore
                # inf나 매우 큰 값 처리
                if np.isinf(vif_value) or vif_value > 1e10:
                    vif_dict[col] = np.inf
                else:
                    vif_dict[col] = vif_value
        except:
            vif_dict[col] = np.inf

    for idx, row in tbl.iterrows():
        name = idx
        if name not in xnames:
            continue

        b = float(row['Coef.'])
        se = float(row['Std.Err.'])
        t = float(row['t'])
        p = float(row['P>|t|'])

        # 표준화 회귀계수(β) 계산
        beta = b * (data[name].std(ddof=1) / data[yname].std(ddof=1))

        # VIF 값
        vif = vif_dict.get(name, np.nan)

        # 유의확률과 별표 표시
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        # 한 변수에 대한 보고 정보 추가
        var_row = {
            "종속변수": yname,  # 종속변수 이름
            "독립변수": name,  # 독립변수 이름
            "B(비표준화 계수)": np.round(b, 4),  # 비표준화 회귀계수(B)
        }
        # logvar가 True면 exp(B) 컬럼 추가
        if 'logvar' in locals() and logvar:
            var_row["exp(B)"] = np.round(np.exp(b), 4)

        var_row.update({
            "표준오차": np.round(se, 4),  # 계수 표준오차
            "β(표준화 계수)": np.round(beta, 4),  # 표준화 회귀계수(β)
            "t": f"{np.round(t, 4)}{stars}",  # t-통계량(+별표)
            "유의확률": np.round(p, 4),  # 계수 유의확률
            #"significant": p <= alpha,  # 유의성 여부 (boolean)
            #"공차": 1 / vif,  # 공차(Tolerance = 1/VIF)
            "vif": np.round(vif, 4),  # 분산팽창계수
        })
        variables.append(var_row)

    rdf = DataFrame(variables)

    # summary 표에서 적합도 정보를 key-value로 추출
    result_dict = {}
    summary_main = fit.summary()
    for i in [0, 2]:
        for item in summary_main.tables[i].data:
            n = len(item)
            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i + 1].strip()
                if not key or not value:
                    continue
                result_dict[key] = value

    r2 = float(result_dict.get('R-squared', np.nan))
    adj_r2 = float(result_dict.get('Adj. R-squared', np.nan))
    r = np.sqrt(r2) if r2 >= 0 else np.nan

    # 적합도 보고 문자열 구성
    result_report = f"𝑅({r:.3f}), 𝑅^2({r2:.3f}), Adj 𝑅^2({adj_r2:.3f}), 𝐹({float(result_dict['F-statistic']):.3f}), 유의확률({float(result_dict['Prob (F-statistic)']):.3f}), Durbin-Watson({float(result_dict['Durbin-Watson']):.3f})"

    # 모형 보고 문장 구성
    tpl = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %0.3f, p %s 0.05)."
    model_report = tpl % (
        rdf["종속변수"][0],
        ",".join(list(rdf["독립변수"])),
        (
            "유의하다"
            if float(result_dict["Prob (F-statistic)"]) <= 0.05
            else "유의하지 않다"
        ),
        result_dict["Df Model"],
        result_dict["Df Residuals"],
        float(result_dict["F-statistic"]),
        "<=" if float(result_dict["Prob (F-statistic)"]) <= 0.05 else ">",
    )

    # 변수별 보고 문장 리스트 구성
    variable_reports = []
    s_normal = "%s가 1 증가하면 %s(이)가 %.3f만큼 변하는 것으로 나타남. (p %s 0.05, %s)"
    s_log = "%s가 1 증가하면 %s(이)가 약 %.3f배 변하는 것으로 나타남. (p %s 0.05, %s)"

    for i in rdf.index:
        row = rdf.iloc[i]
        if logvar:
            effect = np.exp(float(row["B(비표준화 계수)"]))
            variable_reports.append(
                s_log
                % (
                    row["독립변수"],
                    row["종속변수"],
                    effect,
                    "<=" if float(row["유의확률"]) < 0.05 else ">",
                    "유의함" if float(row["유의확률"]) < 0.05 else "유의하지 않음",
                )
            )
        else:
            variable_reports.append(
                s_normal
                % (
                    row["독립변수"],
                    row["종속변수"],
                    float(row["B(비표준화 계수)"]),
                    "<=" if float(row["유의확률"]) < 0.05 else ">",
                    "유의함" if float(row["유의확률"]) < 0.05 else "유의하지 않음",
                )
            )

    # -----------------------------
    # 회귀식 자동 출력
    # -----------------------------
    intercept = fit.params["const"]
    terms = []

    for name in xnames:
        coef = fit.params[name]
        sign = "+" if coef >= 0 else "-"
        terms.append(f" {sign} {abs(coef):.3f}·{name}")

    equation_text = f"{yname} = {intercept:.3f}" + "".join(terms)

    # 성능 지표 표 생성 (pdf)
    pdf = DataFrame(
        {
            "R": [r],
            "R²": [r2],
            "Adj. R²": [adj_r2],
            "F": [float(result_dict.get('F-statistic', np.nan))],
            "p-value": [float(result_dict.get('Prob (F-statistic)', np.nan))],
            "Durbin-Watson": [float(result_dict.get('Durbin-Watson', np.nan))],
        }
    )

    if full:
        return pdf, rdf, result_report, model_report, variable_reports, equation_text
    else:
        return pdf, rdf


# ===================================================================
# 선형회귀
# ===================================================================
@overload
def ols(
    df: DataFrame,
    yname: str,
    report: Literal[False]
) -> RegressionResultsWrapper: ...

@overload
def ols(
    df: DataFrame,
    yname: str,
    report: Literal["summary"]
) -> tuple[RegressionResultsWrapper, DataFrame, DataFrame]: ...

@overload
def ols(
    df: DataFrame,
    yname: str,
    report: Literal["full"]
) -> tuple[
    RegressionResultsWrapper,
    DataFrame,
    DataFrame,
    str,
    str,
    list[str],
    str
]: ...

def ols(df: DataFrame, yname: str, report: Literal[False, "summary", "full"] = "summary") -> Union[
    Tuple[RegressionResultsWrapper, DataFrame, DataFrame],
    Tuple[
        RegressionResultsWrapper,
        DataFrame,
        DataFrame,
        str,
        str,
        list[str],
        str
    ],
    RegressionResultsWrapper
]:
    """선형회귀분석을 수행하고 적합 결과를 반환한다.

    OLS(Ordinary Least Squares) 선형회귀분석을 실시한다.
    필요시 상세한 통계 보고서를 함께 제공한다.

    Args:
        df (DataFrame): 종속변수와 독립변수를 모두 포함한 데이터프레임.
        yname (str): 종속변수 컬럼명.
        report (bool | str): 리포트 모드 설정. 다음 값 중 하나:
            - False (기본값): 리포트 미사용. fit 객체만 반환.
            - 1 또는 'summary': 요약 리포트 반환 (full=False).
            - 2 또는 'full': 풀 리포트 반환 (full=True).
            - True: 풀 리포트 반환 (2와 동일).

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: report=False일 때.
            선형회귀 적합 결과 객체. fit.summary()로 상세 결과 확인 가능.

        tuple (6개): report=1 또는 'summary'일 때.
            (fit, rdf, result_report, model_report, variable_reports, equation_text) 형태로 (pdf 제외).

        tuple (7개): report=2, 'full' 또는 True일 때.
            (fit, pdf, rdf, result_report, model_report, variable_reports, equation_text) 형태로:
            - fit: 선형회귀 적합 결과 객체
            - pdf: 성능 지표 표 (DataFrame): R, R², F, p-value, Durbin-Watson
            - rdf: 회귀계수 표 (DataFrame)
            - result_report: 적합도 요약 (str)
            - model_report: 모형 보고 문장 (str)
            - variable_reports: 변수별 보고 문장 리스트 (list[str])
            - equation_text: 회귀식 문자열 (str)

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        import numpy as np

        df = DataFrame({
            'target': np.random.normal(100, 10, 100),
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100)
        })

        # 적합 결과만 반환
        fit = hs_stats.ols(df, 'target')

        # 요약 리포트 반환
        fit, pdf, rdf = hs_stats.ols(df, 'target', report='summary')

        # 풀 리포트 반환
        fit, pdf, rdf, result_report, model_report, var_reports, eq = hs_stats.ols(df, 'target', report='full')
        ```
    """
    x = df.drop(yname, axis=1)
    y = df[yname]

    X_const = sm.add_constant(x)
    linear_model = sm.OLS(y, X_const)
    linear_fit = linear_model.fit()

    # report 파라미터에 따른 처리
    if not report or report is False:
        # 리포트 미사용
        return linear_fit
    elif report == 'full':
        # 풀 리포트 (full=True)
        pdf, rdf, result_report, model_report, variable_reports, equation_text = ols_report(linear_fit, df, full=True, alpha=0.05)  # type: ignore
        return linear_fit, pdf, rdf, result_report, model_report, variable_reports, equation_text
    else:
        # 요약 리포트 (full=False) -> report == 1 or report == 'summary':
        pdf, rdf = ols_report(linear_fit, df, full=False, alpha=0.05)   # type: ignore
        return linear_fit, pdf, rdf


# ===================================================================
# 선형성 검정 (Linearity Test)
# ===================================================================
def ols_linearity_test(fit: RegressionResultsWrapper, power: int = 2, alpha: float = 0.05,  plot: bool = False, title: str | None = None, save_path: str | None = None) -> DataFrame:
    """회귀모형의 선형성을 Ramsey RESET 검정으로 평가한다.

    적합된 회귀모형에 대해 Ramsey RESET(Regression Specification Error Test) 검정을 수행하여
    모형의 선형성 가정이 타당한지를 검정한다. 귀무가설은 '모형이 선형이다'이다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
             OLS 또는 WLS 모형이어야 한다.
        power (int, optional): RESET 검정에 사용할 거듭제곱 수. 기본값 2.
                               power=2일 때 예측값의 제곱항이 추가됨.
                               power가 클수록 더 높은 차수의 비선형성을 감지.
        alpha (float, optional): 유의수준. 기본값 0.05

    Returns:
        DataFrame: 선형성 검정 결과를 포함한 데이터프레임.
                   - 검정통계량: F-statistic
                   - p-value: 검정의 p값
                   - 유의성: alpha 기준 결과 (bool)
                   - 해석: 선형성 판정 (문자열)

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.logit(df, 'target')
        result = hs_stats.ols_linearity_test(fit)
        ```

    Notes:
        - p-value > alpha: 선형성 가정을 만족 (귀무가설 채택)
        - p-value <= alpha: 선형성 가정 위반 가능 (귀무가설 기각)
    """
    import re

    # Ramsey RESET 검정 수행
    reset_result = linear_reset(fit, power=power)

    # ContrastResults 객체에서 결과 추출
    test_stat = None
    p_value = None

    try:
        # 문자열 표현에서 숫자 추출 시도
        result_str = str(reset_result)

        # 정규식으로 숫자값들 추출 (F-statistic과 p-value)
        numbers = re.findall(r'\d+\.?\d*[eE]?-?\d*', result_str)

        if len(numbers) >= 2:
            # 일반적으로 첫 번째는 F-statistic, 마지막은 p-value
            test_stat = float(numbers[0])
            p_value = float(numbers[-1])
    except (ValueError, IndexError, AttributeError):
        pass

    # 정규식 실패 시 직접 속성 접근
    if test_stat is None or p_value is None:
        attr_pairs = [
            ('statistic', 'pvalue'),
            ('test_stat', 'pvalue'),
            ('fvalue', 'pvalue'),
        ]

        for attr_stat, attr_pval in attr_pairs:
            if hasattr(reset_result, attr_stat) and hasattr(reset_result, attr_pval):
                try:
                    test_stat = float(getattr(reset_result, attr_stat))
                    p_value = float(getattr(reset_result, attr_pval))
                    break
                except (ValueError, TypeError):
                    continue

    # 여전히 값을 못 찾으면 에러
    if test_stat is None or p_value is None:
        raise ValueError(f"linear_reset 결과를 해석할 수 없습니다. 반환값: {reset_result}")

    # 유의성 판정
    significant = p_value <= alpha

    # 해석 문구
    if significant:
        interpretation = f"선형성 가정 위반 (p={p_value:.4f} <= {alpha})"
    else:
        interpretation = f"선형성 가정 만족 (p={p_value:.4f} > {alpha})"

    # 결과를 DataFrame으로 반환
    result_df = DataFrame({
        "검정": ["Ramsey RESET"],
        "검정통계량 (F)": [f"{test_stat:.4f}"],
        "p-value": [f"{p_value:.4f}"],
        "유의수준": [alpha],
        "선형성_위반": [significant],  # True: 선형성 위반, False: 선형성 만족
        "해석": [interpretation]
    })

    if plot:
        ols_residplot(fit, lowess=True, mse=True, title=title, save_path=save_path)

    return result_df


# ===================================================================
# 정규성 검정 (Normality Test)
# ===================================================================
def ols_normality_test(fit: RegressionResultsWrapper, alpha: float = 0.05, plot: bool = False, title: str | None = None, save_path: str | None = None) -> DataFrame:
    """회귀모형 잔차의 정규성을 검정한다.

    회귀모형의 잔차가 정규분포를 따르는지 Shapiro-Wilk 검정과 Jarque-Bera 검정으로 평가한다.
    정규성 가정은 회귀분석의 추론(신뢰구간, 가설검정)이 타당하기 위한 중요한 가정이다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
        alpha (float, optional): 유의수준. 기본값 0.05.
        plot (bool, optional): True이면 Q-Q 플롯을 출력. 기본값 False.
        title (str, optional): 플롯 제목. 기본값 None.
        save_path (str, optional): 플롯을 저장할 경로. 기본값 None

    Returns:
        DataFrame: 정규성 검정 결과를 포함한 데이터프레임.
                   - 검정명: 사용된 검정 방법
                   - 검정통계량: 검정 통계량 값
                   - p-value: 검정의 p값
                   - 유의수준: 설정된 유의수준
                   - 정규성_위반: alpha 기준 결과 (bool)
                   - 해석: 정규성 판정 (문자열)

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.logit(df, 'target')
        result = hs_stats.ols_normality_test(fit)
        ```

    Notes:
        - Shapiro-Wilk: 샘플 크기가 작을 때 (< 5000) 강력한 검정
        - Jarque-Bera: 왜도(skewness)와 첨도(kurtosis) 기반 검정
        - p-value > alpha: 정규성 가정 만족 (귀무가설 채택)
        - p-value <= alpha: 정규성 가정 위반 (귀무가설 기각)
    """
    # fit 객체에서 잔차 추출
    residuals = fit.resid
    n = len(residuals)

    results = []

    # 1. Shapiro-Wilk 검정 (샘플 크기 < 5000일 때 권장)
    if n < 5000:
        try:
            stat_sw, p_sw = shapiro(residuals)
            significant_sw = p_sw <= alpha

            if significant_sw:
                interpretation_sw = f"정규성 위반 (p={p_sw:.4f} <= {alpha})"
            else:
                interpretation_sw = f"정규성 만족 (p={p_sw:.4f} > {alpha})"

            results.append({
                "검정": "Shapiro-Wilk",
                "검정통계량": f"{stat_sw:.4f}",
                "p-value": f"{p_sw:.4f}",
                "유의수준": alpha,
                "정규성_위반": significant_sw,
                "해석": interpretation_sw
            })
        except Exception as e:
            pass

    # 2. Jarque-Bera 검정 (항상 수행)
    try:
        stat_jb, p_jb = jarque_bera(residuals)
        significant_jb = p_jb <= alpha  # type: ignore

        if significant_jb:
            interpretation_jb = f"정규성 위반 (p={p_jb:.4f} <= {alpha})"
        else:
            interpretation_jb = f"정규성 만족 (p={p_jb:.4f} > {alpha})"

        results.append({
            "검정": "Jarque-Bera",
            "검정통계량": f"{stat_jb:.4f}",
            "p-value": f"{p_jb:.4f}",
            "유의수준": alpha,
            "정규성_위반": significant_jb,
            "해석": interpretation_jb
        })
    except Exception as e:
        pass

    # 결과를 DataFrame으로 반환
    if not results:
        raise ValueError("정규성 검정을 수행할 수 없습니다.")


    if plot:
        ols_qqplot(fit, title=title, save_path=save_path)

    result_df = DataFrame(results)
    return result_df


# ===================================================================
# 등분산성 검정 (Homoscedasticity Test)
# ===================================================================
def ols_variance_test(fit: RegressionResultsWrapper, alpha: float = 0.05, plot: bool = False, title: str | None = None, save_path: str | None = None) -> DataFrame:
    """회귀모형의 등분산성 가정을 검정한다.

    잔차의 분산이 예측값의 수준에 관계없이 일정한지 Breusch-Pagan 검정과 White 검정으로 평가한다.
    등분산성 가정은 회귀분석의 추론(표준오차, 검정통계량)이 정확하기 위한 중요한 가정이다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
        alpha (float, optional): 유의수준. 기본값 0.05.
        plot (bool, optional): True이면 Q-Q 플롯을 출력. 기본값 False.
        title (str, optional): 플롯 제목. 기본값 None.
        save_path (str, optional): 플롯을 저장할 경로. 기본값 None

    Returns:
        DataFrame: 등분산성 검정 결과를 포함한 데이터프레임.
                   - 검정명: 사용된 검정 방법
                   - 검정통계량: 검정 통계량 값 (LM 또는 F)
                   - p-value: 검정의 p값
                   - 유의수준: 설정된 유의수준
                   - 등분산성_위반: alpha 기준 결과 (bool)
                   - 해석: 등분산성 판정 (문자열)

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.logit(df, 'target')
        result = hs_stats.ols_variance_test(fit)
        ```

    Notes:
        - Breusch-Pagan: 잔차 제곱과 독립변수의 선형관계 검정
        - White: 잔차 제곱과 독립변수 및 그 제곱, 교차항의 관계 검정
        - p-value > alpha: 등분산성 가정 만족 (귀무가설 채택)
        - p-value <= alpha: 이분산성 존재 (귀무가설 기각)
    """

    # fit 객체에서 필요한 정보 추출
    exog = fit.model.exog  # 설명변수 (상수항 포함)
    resid = fit.resid      # 잔차

    results = []

    # 1. Breusch-Pagan 검정
    try:
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, exog)
        significant_bp = lm_pvalue <= alpha

        if significant_bp:
            interpretation_bp = f"등분산성 위반 (p={lm_pvalue:.4f} <= {alpha})"
        else:
            interpretation_bp = f"등분산성 만족 (p={lm_pvalue:.4f} > {alpha})"

        results.append({
            "검정": "Breusch-Pagan",
            "검정통계량 (LM)": f"{lm:.4f}",
            "p-value": f"{lm_pvalue:.4f}",
            "유의수준": alpha,
            "등분산성_위반": significant_bp,
            "해석": interpretation_bp
        })
    except Exception as e:
        pass

    # 2. White 검정
    try:
        lm, lm_pvalue, fvalue, f_pvalue = het_white(resid, exog)
        significant_white = lm_pvalue <= alpha

        if significant_white:
            interpretation_white = f"등분산성 위반 (p={lm_pvalue:.4f} <= {alpha})"
        else:
            interpretation_white = f"등분산성 만족 (p={lm_pvalue:.4f} > {alpha})"

        results.append({
            "검정": "White",
            "검정통계량 (LM)": f"{lm:.4f}",
            "p-value": f"{lm_pvalue:.4f}",
            "유의수준": alpha,
            "등분산성_위반": significant_white,
            "해석": interpretation_white
        })
    except Exception as e:
        pass

    # 결과를 DataFrame으로 반환
    if not results:
        raise ValueError("등분산성 검정을 수행할 수 없습니다.")

    if plot:
        ols_residplot(fit, lowess=True, mse=True, title=title, save_path=save_path)

    result_df = DataFrame(results)
    return result_df


# ===================================================================
# 독립성 검정 (Independence Test - Durbin-Watson)
# ===================================================================
def ols_independence_test(fit: RegressionResultsWrapper, alpha: float = 0.05) -> DataFrame:
    """회귀모형의 독립성 가정(자기상관 없음)을 검정한다.

    Durbin-Watson 검정을 사용하여 잔차의 1차 자기상관 여부를 검정한다.
    시계열 데이터나 순서가 있는 데이터에서 주로 사용된다.

    Args:
        fit: statsmodels 회귀분석 결과 객체 (RegressionResultsWrapper).
        alpha (float, optional): 유의수준. 기본값은 0.05.

    Returns:
        DataFrame: 검정 결과 데이터프레임.
            - 검정: 검정 방법명
            - 검정통계량(DW): Durbin-Watson 통계량 (0~4 범위, 2에 가까울수록 자기상관 없음)
            - 독립성_위반: 자기상관 의심 여부 (True/False)
            - 해석: 검정 결과 해석

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.logit(df, 'target')
        result = hs_stats.ols_independence_test(fit)
        ```

    Notes:
        - Durbin-Watson 통계량 해석:
          * 2에 가까우면: 자기상관 없음 (독립성 만족)
          * 0에 가까우면: 양의 자기상관 (독립성 위반)
          * 4에 가까우면: 음의 자기상관 (독립성 위반)
        - 일반적으로 1.5~2.5 범위를 자기상관 없음으로 판단
        - 시계열 데이터나 관측치에 순서가 있는 경우 중요한 검정
    """
    # Durbin-Watson 통계량 계산
    dw_stat = durbin_watson(fit.resid)

    # 자기상관 판단 (1.5 < DW < 2.5 범위를 독립성 만족으로 판단)
    is_autocorrelated = dw_stat < 1.5 or dw_stat > 2.5

    # 해석 메시지 생성
    if dw_stat < 1.5:
        interpretation = f"DW={dw_stat:.4f} < 1.5 (양의 자기상관)"
    elif dw_stat > 2.5:
        interpretation = f"DW={dw_stat:.4f} > 2.5 (음의 자기상관)"
    else:
        interpretation = f"DW={dw_stat:.4f} (독립성 가정 만족)"

    # 결과 데이터프레임 생성
    result_df = DataFrame(
        {
            "검정": ["Durbin-Watson"],
            "검정통계량(DW)": [dw_stat],
            "독립성_위반": [is_autocorrelated],
            "해석": [interpretation],
        }
    )

    return result_df



def ols_tests(fit: RegressionResultsWrapper, alpha: float = 0.05, plot: bool = False, title: str | None = None, save_path: str | None = None) -> None:
    """회귀모형의 가정 검정을 종합적으로 수행한다.

    선형성, 정규성, 등분산성, 독립성 검정을 순차적으로 실시하고 결과를 하나의 데이터프레임으로 반환한다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
        alpha (float, optional): 유의수준. 기본값 0.05.
        plot (bool, optional): True이면 정규성 검정 시 Q-Q 플롯을 출력. 기본값 False.
        title (str, optional): 플롯 제목. 기본값 None.
        save_path (str, optional): 플롯을 저장할 경로. 기본값 None.

    Returns:
        None

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.ols(df, 'target')
        hs_stats.ols_tests(fit)
        ```
    """
    # 각 검정 함수 호출
    linearity_df = ols_linearity_test(fit, alpha=alpha)
    normality_df = ols_normality_test(fit, alpha=alpha, plot=False)
    variance_df = ols_variance_test(fit, alpha=alpha, plot=False)
    independence_df = ols_independence_test(fit, alpha=alpha)

    from IPython.display import display
    display(linearity_df)
    display(normality_df)
    display(variance_df)
    display(independence_df)

    if plot:
        fig, ax = get_default_ax(rows=1, cols=2, title=title)
        ols_qqplot(fit, ax=ax[0])
        ols_residplot(fit, lowess=True, mse=True, ax=ax[1])
        finalize_plot(ax, outparams=True)


# ===================================================================
# 로지스틱 회귀 요약 리포트
# ===================================================================
def logit_report(
    fit: BinaryResultsWrapper,
    data: DataFrame,
    threshold: float = 0.5,
    full: Union[bool, str, int] = False,
    alpha: float = 0.05
) -> Union[
    Tuple[DataFrame, DataFrame],
    Tuple[
        DataFrame,
        DataFrame,
        str,
        str,
        list[str],
        np.ndarray
    ]
]:
    """로지스틱 회귀 적합 결과를 상세 리포트로 변환한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.summary()`와 예측 확률을 지원해야 함).
        data (DataFrame): 종속변수와 독립변수를 모두 포함한 DataFrame.
        threshold (float): 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
        full (bool | str | int): True이면 6개 값 반환, False이면 주요 2개(cdf, rdf)만 반환. 기본값 False.
        alpha (float): 유의수준. 기본값 0.05.

    Returns:
        tuple: full=True일 때 다음 요소를 포함한다.
            - 성능 지표 표 (`cdf`, DataFrame): McFadden Pseudo R², Accuracy, Precision, Recall, FPR, TNR, AUC, F1.
            - 회귀계수 표 (`rdf`, DataFrame): B, 표준오차, z, p-value, significant, OR, 95% CI, VIF 등.
            - 적합도 및 예측 성능 요약 (`result_report`, str): Pseudo R², LLR χ², p-value, Accuracy, AUC.
            - 모형 보고 문장 (`model_report`, str): LLR p-value에 기반한 서술형 문장.
            - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수의 오즈비 해석 문장.
            - 혼동행렬 (`cm`, ndarray): 예측 결과와 실제값의 혼동행렬 [[TN, FP], [FN, TP]].

        full=False일 때:
            - 성능 지표 표 (`cdf`, DataFrame)
            - 회귀계수 표 (`rdf`, DataFrame)

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        import numpy as np

        df = DataFrame({
            'target': np.random.binomial(1, 0.5, 100),
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100)
        })

        # 로지스틱 회귀 적합
        fit = hs_stats.logit(df, yname="target")

        # 전체 리포트
        cdf, rdf, result_report, model_report, variable_reports, cm = hs_stats.logit_report(fit, df, full=True)

        # 간단한 버전 (주요 테이블만)
        cdf, rdf = hs_stats.logit_report(fit, df)
        ```
    """

    # -----------------------------
    # 성능평가지표
    # -----------------------------
    yname = fit.model.endog_names
    y_true = data[yname]
    y_pred = fit.predict(fit.model.exog)
    y_pred_fix = (y_pred >= threshold).astype(int)

    # 혼동행렬
    cm = confusion_matrix(y_true, y_pred_fix)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred_fix)  # 정확도
    pre = precision_score(y_true, y_pred_fix)  # 정밀도
    tpr = recall_score(y_true, y_pred_fix)  # 재현율
    fpr = fp / (fp + tn)  # 위양성율
    tnr = 1 - fpr  # 특이성
    f1 = f1_score(y_true, y_pred_fix)  # f1-score
    ras = roc_auc_score(y_true, y_pred)  # auc score

    cdf = DataFrame(
        {
            "설명력(P-Rsqe)": [fit.prsquared],
            "정확도(Accuracy)": [acc],
            "정밀도(Precision)": [pre],
            "재현율(Recall,TPR)": [tpr],
            "위양성율(Fallout,FPR)": [fpr],
            "특이성(Specif city,TNR)": [tnr],
            "RAS(auc score)": [ras],
            "F1": [f1],
        }
    )

    # -----------------------------
    # 회귀계수 표 구성 (OR 중심)
    # -----------------------------
    tbl = fit.summary2().tables[1]

    # 독립변수 이름(상수항 제외)
    xnames = [n for n in fit.model.exog_names if n != "const"]

    # 독립변수
    x = data[xnames]

    variables = []

    # VIF 계산 (상수항 포함 설계행렬 사용)
    vif_dict = {}
    x_const = sm.add_constant(x, has_constant="add")
    for i, col in enumerate(x.columns, start=1):  # 상수항이 0이므로 1부터 시작
        vif_dict[col] = variance_inflation_factor(x_const.values, i)    # type: ignore

    for idx, row in tbl.iterrows():
        name = idx
        if name not in xnames:
            continue

        beta = float(row['Coef.'])
        se = float(row['Std.Err.'])
        z = float(row['z'])
        p = float(row['P>|z|'])

        or_val = np.exp(beta)
        ci_low = np.exp(beta - 1.96 * se)
        ci_high = np.exp(beta + 1.96 * se)

        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        variables.append(
            {
                "종속변수": yname,
                "독립변수": name,
                "B(β)": beta,
                "표준오차": se,
                "z": f"{z:.3f}{stars}",
                "p-value": p,
                "significant": p <= alpha,
                "OR": or_val,
                "CI_lower": ci_low,
                "CI_upper": ci_high,
                "VIF": vif_dict.get(name, np.nan),
            }
        )

    rdf = DataFrame(variables)

    # ---------------------------------
    # 모델 적합도 + 예측 성능 지표
    # ---------------------------------
    auc = roc_auc_score(y_true, y_pred)

    result_report = (
        f"Pseudo R²(McFadden) = {fit.prsquared:.3f}, "
        f"LLR χ²({int(fit.df_model)}) = {fit.llr:.3f}, "
        f"p-value = {fit.llr_pvalue:.4f}, "
        f"Accuracy = {acc:.3f}, "
        f"AUC = {auc:.3f}"
    )

    # -----------------------------
    # 모형 보고 문장
    # -----------------------------
    tpl = (
        "%s에 대하여 %s로 예측하는 로지스틱 회귀분석을 실시한 결과, "
        "모형은 통계적으로 %s(χ²(%s) = %.3f, p %s 0.05)하였다."
    )

    model_report = tpl % (
        yname,
        ", ".join(xnames),
        "유의" if fit.llr_pvalue <= 0.05 else "유의하지 않음",
        int(fit.df_model),
        fit.llr,
        "<=" if fit.llr_pvalue <= 0.05 else ">",
    )

    # -----------------------------
    # 변수별 보고 문장
    # -----------------------------
    variable_reports = []

    s = (
        "%s의 오즈비는 %.3f(p %s 0.05)로, "
        "%s 발생 odds에 %s 영향을 미치는 것으로 나타났다."
    )

    for _, row in rdf.iterrows():
        variable_reports.append(
            s
            % (
                row["독립변수"],
                row["OR"],
                "<=" if row["p-value"] < 0.05 else ">",
                row["종속변수"],
                "유의미한" if row["p-value"] < 0.05 else "유의하지 않은",
            )
        )

    if full:
        return cdf, rdf, result_report, model_report, variable_reports, cm
    else:
        return cdf, rdf


# ===================================================================
# 로지스틱 회귀
# ===================================================================
def logit(
    df: DataFrame,
    yname: str,
    report: bool | str = 'summary'
) -> Union[
    BinaryResultsWrapper,
    Tuple[
        BinaryResultsWrapper,
        DataFrame,
        DataFrame
    ],
    Tuple[
        BinaryResultsWrapper,
        DataFrame,
        DataFrame,
        str,
        str,
        list[str]
    ]
]:
    """로지스틱 회귀분석을 수행하고 적합 결과를 반환한다.

    종속변수가 이항(binary) 형태일 때 로지스틱 회귀분석을 실시한다.
    필요시 상세한 통계 보고서를 함께 제공한다.

    Args:
        df (DataFrame): 종속변수와 독립변수를 모두 포함한 데이터프레임.
        yname (str): 종속변수 컬럼명. 이항 변수여야 한다.
        report: 리포트 모드 설정. 다음 값 중 하나:
            - False (기본값): 리포트 미사용. fit 객체만 반환.
            - 1 또는 'summary': 요약 리포트 반환 (full=False).
            - 2 또는 'full': 풀 리포트 반환 (full=True).
            - True: 풀 리포트 반환 (2와 동일).

    Returns:
        statsmodels.genmod.generalized_linear_model.BinomialResults: report=False일 때.
            로지스틱 회귀 적합 결과 객체. fit.summary()로 상세 결과 확인 가능.

        tuple (4개): report=1 또는 'summary'일 때.
            (fit, rdf, result_report, variable_reports) 형태로 (cdf 제외).

        tuple (6개): report=2, 'full' 또는 True일 때.
            (fit, cdf, rdf, result_report, model_report, variable_reports) 형태로:
            - fit: 로지스틱 회귀 적합 결과 객체
            - cdf: 성능 지표 표 (DataFrame)
            - rdf: 회귀계수 표 (DataFrame)
            - result_report: 적합도 및 예측 성능 요약 (str)
            - model_report: 모형 보고 문장 (str)
            - variable_reports: 변수별 보고 문장 리스트 (list[str])

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame
        import numpy as np

        df = DataFrame({
            'target': np.random.binomial(1, 0.5, 100),
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100)
        })

        # 적합 결과만 반환
        fit = hs_stats.logit(df, 'target')

        # 요약 리포트 반환
        fit, rdf, result_report, var_reports = hs_stats.logit(df, 'target', report='summary')

        # 풀 리포트 반환
        fit, cdf, rdf, result_report, model_report, var_reports = hs_stats.logit(df, 'target', report='full')
        ```
    """
    x = df.drop(yname, axis=1)
    y = df[yname]

    X_const = sm.add_constant(x)
    logit_model = sm.Logit(y, X_const)
    logit_fit = logit_model.fit(disp=False)

    # report 파라미터에 따른 처리
    if not report or report is False:
        # 리포트 미사용
        return logit_fit
    elif report == 'full':
        # 풀 리포트 (full=True)
        cdf, rdf, result_report, model_report, variable_reports, cm = logit_report(logit_fit, df, threshold=0.5, full=True, alpha=0.05) # type: ignore
        return logit_fit, cdf, rdf, result_report, model_report, variable_reports
    else:
        # 요약 리포트 (report == 'summary')
        cdf, rdf = logit_report(logit_fit, df, threshold=0.5, full=False, alpha=0.05)   # type: ignore
        # 요약에서는 result_report와 variable_reports만 포함
        # 간단한 버전으로 result와 variable_reports만 생성
        return logit_fit, cdf, rdf


# ===================================================================
# 모델 예측 (Model Prediction)
# ===================================================================
def predict(fit, data: DataFrame | Series) -> DataFrame | Series | float:
    """회귀 또는 로지스틱 모형을 이용하여 예측값을 생성한다.

    statsmodels의 RegressionResultsWrapper(선형회귀) 또는
    BinaryResultsWrapper(로지스틱 회귀) 객체를 받아 데이터에 대한
    예측값을 생성하고 반환한다.

    모형 학습 시 상수항이 포함되었다면, 예측 데이터에도 자동으로
    상수항을 추가하여 차원을 맞춘다.

    로지스틱 회귀의 경우 예측 확률과 함께 분류 해석을 포함한다.

    Args:
        fit: 학습된 회귀 모형 객체.
             - statsmodels.regression.linear_model.RegressionResultsWrapper (선형회귀)
             - statsmodels.discrete.discrete_model.BinaryResultsWrapper (로지스틱 회귀)
        data (DataFrame|Series): 예측에 사용할 설명변수.
                                 - DataFrame: 여러 개의 관측치
                                 - Series: 단일 관측치 또는 변수 하나
                                 원본 모형 학습 시 사용한 특성과 동일해야 함.
                                 (상수항 제외, 자동으로 추가됨)

    Returns:
        DataFrame|Series|float: 예측값.
                          - DataFrame 입력:
                            - 선형회귀: 예측값 컬럼을 포함한 DataFrame
                            - 로지스틱: 확률, 분류, 해석 컬럼을 포함한 DataFrame
                          - Series 입력: 단일 예측값 (float)

    Raises:
        ValueError: fit 객체가 지원되지 않는 타입인 경우.
        Exception: 데이터와 모형의 특성 불일치로 인한 predict 실패.

    Examples:
        ```python
        from hossam import *

        df = hs_util.load_data("some_data.csv")
        fit1 = hs_stats.ols(df, yname="target")

        pred = hs_stats.predict(fit1, df_new[['x1', 'x2']])  # DataFrame 반환

        # 로지스틱 회귀 (상수항 자동 추가)
        fit2 = hs_stats.logit(df, yname="target")
        pred_prob = hs_stats.predict(fit2, df_new[['x1', 'x2']])  # DataFrame 반환 (해석 포함)
        ```
    """
    # fit 객체의 타입 확인
    fit_type = type(fit).__name__

    # RegressionResultsWrapper인지 BinaryResults인지 확인
    is_linear = isinstance(fit, RegressionResultsWrapper)
    is_logit = isinstance(fit, BinaryResults) or 'BinaryResult' in fit_type

    if not (is_linear or is_logit):
        raise ValueError(
            f"지원되지 않는 fit 객체 타입입니다: {fit_type}\n"
            "RegressionResultsWrapper 또는 BinaryResults를 사용하세요."
        )

    # Series를 DataFrame으로 변환
    if isinstance(data, Series):
        data_to_predict = data.to_frame().T
        is_series = True
    else:
        data_to_predict = data.copy()
        is_series = False

    try:
        # 모형의 매개변수 수와 입력 데이터의 특성 수를 비교하여 상수항 필요 여부 판단
        n_params = len(fit.params)
        n_features = data_to_predict.shape[1]

        # 상수항이 필요한 경우 자동으로 추가
        if n_params == n_features + 1:
            data_to_predict = sm.add_constant(data_to_predict, has_constant='skip')
        elif n_params != n_features:
            raise ValueError(
                f"특성 수 불일치: 모형은 {n_params}개의 매개변수를 기대하지만, "
                f"입력 데이터는 {n_features}개의 특성을 제공했습니다. "
                f"(상수항 자동 감지 후에도 불일치)"
            )

        # 예측값 생성
        predictions = fit.predict(data_to_predict)

        # Series 입력인 경우 단일 값 반환
        if is_series:
            return float(predictions.iloc[0])   # type: ignore

        # DataFrame 입력인 경우
        if isinstance(data, DataFrame):
            result_df = DataFrame({'예측값': predictions}, index=data.index)

            # 로지스틱 회귀인 경우 추가 정보 포함
            if is_logit:
                # 확률 확인
                result_df['확률(%)'] = (predictions * 100).round(2)
                # 분류 (0.5 기준)
                result_df['분류'] = (predictions >= 0.5).astype(int)
                # 해석 추가
                result_df['해석'] = result_df['분류'].apply(
                    lambda x: '양성(1)' if x == 1 else '음성(0)'
                )
                # 신뢰도 평가
                result_df['신뢰도'] = result_df['확률(%)'].apply(
                    lambda x: f"{abs(x - 50):.1f}% 확실"
                )

            return result_df

        return predictions

    except Exception as e:
        raise Exception(
            f"예측 과정에서 오류가 발생했습니다.\n"
            f"모형 학습 시 사용한 특성과 입력 데이터의 특성이 일치하는지 확인하세요.\n"
            f"원본 오류: {str(e)}"
        )