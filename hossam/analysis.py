# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import numpy as np
from typing import Tuple
from pandas import DataFrame
from pandas.api.types import is_bool_dtype

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
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    wilcoxon
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------

def hs_normal_test(data: DataFrame, columns: list | str | None = None, method: str = "n") -> DataFrame:
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
        >>> from hossam.analysis import hs_normal_test
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'x': np.random.normal(0, 1, 100),
        ...     'y': np.random.exponential(2, 100)
        ... })
        >>> # 모든 수치형 컬럼 검정
        >>> result = hs_normal_test(df, method='n')
        >>> # 특정 컬럼만 검정 (리스트)
        >>> result = hs_normal_test(df, columns=['x'], method='n')
        >>> # 특정 컬럼만 검정 (문자열)
        >>> result = hs_normal_test(df, columns='x, y', method='n')
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
            else:  # method == "s"
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


# -------------------------------------------------------------

def hs_equal_var_test(data: DataFrame, columns: list | str | None = None, normal_dist: bool | None = None) -> DataFrame:
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
            - None: hs_normal_test()를 이용하여 자동으로 정규성을 판별 후 적절한 검정 방법 선택.
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
        >>> from hossam.analysis import hs_equal_var_test
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'x': np.random.normal(0, 1, 100),
        ...     'y': np.random.normal(0, 1, 100),
        ...     'z': np.random.normal(0, 2, 100)
        ... })
        >>> # 모든 수치형 컬럼 자동 판별
        >>> result = hs_equal_var_test(df)
        >>> # 특정 컬럼만 검정 (리스트)
        >>> result = hs_equal_var_test(df, columns=['x', 'y'])
        >>> # 특정 컬럼만 검정 (문자열)
        >>> result = hs_equal_var_test(df, columns='x, y')
        >>> # 명시적 지정
        >>> result = hs_equal_var_test(df, normal_dist=True)
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
        normality_result = hs_normal_test(data[numeric_cols], method="n")
        # 모든 컬럼이 정규분포를 따르는지 확인
        all_normal = normality_result["is_normal"].all()
        normal_dist = all_normal

    try:
        if normal_dist:
            method_name = "Bartlett"
            s, p = bartlett(*fields)
        else:
            method_name = "Levene"
            s, p = levene(*fields)

        result_df = DataFrame([{
            "method": method_name,
            "statistic": s,
            "p-value": p,
            "is_equal_var": p > 0.05,
            "n_columns": len(fields),
            "columns": ", ".join(numeric_cols[:len(fields)]),
            "normality_checked": normality_checked
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


# -------------------------------------------------------------

def hs_ttest_1samp(data: DataFrame, columns: list | str | None = None, mean_value: float = 0.0) -> DataFrame:
    """지정된 컬럼(또는 모든 수치형 컬럼)에 대해 일표본 t-검정을 수행하고 결과를 반환한다.

    일표본 t-검정은 표본 평균이 특정 값(mean_value)과 같은지를 검정한다.
    귀무가설(H0): 모집단 평균 = mean_value
    대립가설(H1): alternative에 따라 달라짐 (!=, <, >)

    Args:
        data (DataFrame): 검정 대상 데이터를 포함한 데이터프레임.
        columns (list | str | None, optional): 검정 대상 컬럼명.
            - None 또는 빈 리스트: 모든 수치형 컬럼에 대해 검정 수행.
            - 컬럼명 리스트: 지정된 컬럼에 대해서만 검정 수행.
            - 콤마로 구분된 문자열: "A, B, C" 형식으로 컬럼명 지정 가능.
            기본값은 None.
        mean_value (float, optional): 귀무가설의 기준값(비교 대상 평균값).
            기본값은 0.0.

    Returns:
        DataFrame: 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
            - field (str): 컬럼명
            - alternative (str): 대립가설 방향 (two-sided, less, greater)
            - statistic (float): t-통계량
            - p-value (float): 유의확률
            - H0 (bool): 귀무가설 채택 여부 (p-value > 0.05)
            - H1 (bool): 대립가설 채택 여부 (p-value <= 0.05)
            - interpretation (str): 검정 결과 해석 문자열

    Examples:
        >>> from hossam.analysis import hs_ttest_1samp
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'x': np.random.normal(5, 1, 100),
        ...     'y': np.random.normal(0, 1, 100)
        ... })
        >>> # 모든 수치형 컬럼에 대해 평균이 0인지 검정
        >>> result = hs_ttest_1samp(df, mean_value=0)
        >>> # 특정 컬럼만 검정 (리스트)
        >>> result = hs_ttest_1samp(df, columns=['x'], mean_value=5)
        >>> # 특정 컬럼만 검정 (문자열)
        >>> result = hs_ttest_1samp(df, columns='x, y', mean_value=5)
    """
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

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []

    for c in target_cols:
        # NaN 값 제거
        col_data = data[c].dropna()

        # 데이터가 없거나 분산이 0인 경우 건너뜀
        if len(col_data) == 0 or col_data.std(ddof=1) == 0:
            for a in alternative:
                result.append({
                    "field": c,
                    "alternative": a,
                    "statistic": np.nan,
                    "p-value": np.nan,
                    "H0": False,
                    "H1": False,
                    "interpretation": f"검정 불가 (데이터 부족 또는 분산=0)"
                })
            continue

        for a in alternative:
            try:
                s, p = ttest_1samp(col_data, mean_value, alternative=a)

                itp = None

                if a == "two-sided":
                    itp = "μ {0} {1}".format("==" if p > 0.05 else "!=", mean_value)
                elif a == "less":
                    itp = "μ {0} {1}".format(">=" if p > 0.05 else "<", mean_value)
                else:
                    itp = "μ {0} {1}".format("<=" if p > 0.05 else ">", mean_value)

                result.append({
                    "field": c,
                    "alternative": a,
                    "statistic": round(s, 3),
                    "p-value": round(p, 3),
                    "H0": p > 0.05,
                    "H1": p <= 0.05,
                    "interpretation": itp,
                })
            except Exception as e:
                result.append({
                    "field": c,
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


# -------------------------------------------------------------

def hs_ttest_ind(
    data: DataFrame, xname: str, yname: str, equal_var: bool | None = None
) -> DataFrame:
    """두 독립 집단의 평균 차이를 검정한다 (독립표본 t-검정 또는 Welch's t-test).

    독립표본 t-검정은 두 독립된 집단의 평균이 같은지를 검정한다.
    귀무가설(H0): μ1 = μ2 (두 집단의 평균이 같다)

    Args:
        data (DataFrame): 검정 대상 데이터를 포함한 데이터프레임.
        xname (str): 첫 번째 집단의 컬럼명.
        yname (str): 두 번째 집단의 컬럼명.
        equal_var (bool | None, optional): 등분산성 가정 여부.
            - True: 독립표본 t-검정 (등분산 가정)
            - False: Welch's t-test (등분산 가정하지 않음, 더 강건함)
            - None: hs_equal_var_test()로 자동 판별
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
        >>> from hossam.analysis import hs_ttest_ind
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'group1': np.random.normal(5, 1, 100),
        ...     'group2': np.random.normal(5.5, 1, 100)
        ... })
        >>> # 자동 등분산성 판별
        >>> result = hs_ttest_ind(df, 'group1', 'group2')
        >>> # 명시적 지정
        >>> result = hs_ttest_ind(df, 'group1', 'group2', equal_var=False)
    """
    # NaN 제거
    x_data = data[xname].dropna()
    y_data = data[yname].dropna()

    # 데이터 유효성 검사
    if len(x_data) < 2 or len(y_data) < 2:
        raise ValueError(f"각 집단에 최소 2개 이상의 데이터가 필요합니다. {xname}: {len(x_data)}, {yname}: {len(y_data)}")

    # equal_var가 None이면 자동으로 등분산성 판별
    var_checked = False
    if equal_var is None:
        var_checked = True
        var_result = hs_equal_var_test(data[[xname, yname]])
        equal_var = var_result["is_equal_var"].iloc[0]

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"

    for a in alternative:
        try:
            s, p = ttest_ind(x_data, y_data, equal_var=equal_var, alternative=a)
            n = "t-test_ind" if equal_var else "Welch's t-test"

            # 검정 결과 해석
            itp = None

            if a == "two-sided":
                itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
            elif a == "less":
                itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
            else:
                itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)

            result.append({
                "test": n,
                "alternative": a,
                "statistic": round(s, 3),
                "p-value": round(p, 3),
                "H0": p > 0.05,
                "H1": p <= 0.05,
                "interpretation": itp,
                "equal_var_checked": var_checked
            })
        except Exception as e:
            result.append({
                "test": "t-test_ind" if equal_var else "Welch's t-test",
                "alternative": a,
                "statistic": np.nan,
                "p-value": np.nan,
                "H0": False,
                "H1": False,
                "interpretation": f"검정 실패: {str(e)}",
                "equal_var_checked": var_checked
            })

    rdf = DataFrame(result)
    rdf.set_index(["test", "alternative"], inplace=True)
    return rdf


# -------------------------------------------------------------

def hs_ttest_rel(
    data: DataFrame, xname: str, yname: str, equal_var: bool | None = None
) -> DataFrame:
    """대응표본 t-검정 또는 Wilcoxon signed-rank test를 수행한다.

    대응표본 t-검정은 동일 개체에서 측정된 두 시점의 평균 차이를 검정한다.
    귀무가설(H0): 두 시점의 평균 차이가 0이다.

    Args:
        data (DataFrame): 검정 대상 데이터를 포함한 데이터프레임.
        xname (str): 첫 번째 측정값의 컬럼명.
        yname (str): 두 번째 측정값의 컬럼명.
        equal_var (bool | None, optional): 정규성/등분산성 가정 여부.
            - True: 대응표본 t-검정 (정규분포 가정)
            - False: Wilcoxon signed-rank test (비모수 검정, 더 강건함)
            - None: hs_equal_var_test()로 자동 판별
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
        >>> from hossam.analysis import hs_ttest_rel
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'before': np.random.normal(5, 1, 100),
        ...     'after': np.random.normal(5.3, 1, 100)
        ... })
        >>> # 자동 정규성 판별
        >>> result = hs_ttest_rel(df, 'before', 'after')
        >>> # 명시적으로 비모수 검정
        >>> result = hs_ttest_rel(df, 'before', 'after', equal_var=False)
    """
    # NaN 제거 (대응표본이므로 행 단위로 제거)
    valid_idx = data[[xname, yname]].dropna().index
    x_data = data.loc[valid_idx, xname]
    y_data = data.loc[valid_idx, yname]

    # 데이터 유효성 검사
    if len(x_data) < 2:
        raise ValueError(f"최소 2개 이상의 대응 데이터가 필요합니다. 현재: {len(x_data)}")

    # equal_var가 None이면 자동으로 등분산성 판별
    var_checked = False
    if equal_var is None:
        var_checked = True
        var_result = hs_equal_var_test(data[[xname, yname]])
        equal_var = var_result["is_equal_var"].iloc[0]

    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"

    for a in alternative:
        try:
            if equal_var:
                s, p = ttest_rel(x_data, y_data, alternative=a)
                n = "t-test_paired"
            else:
                # Wilcoxon signed-rank test (대응표본용 비모수 검정)
                s, p = wilcoxon(x_data, y_data, alternative=a)
                n = "Wilcoxon signed-rank"

            itp = None

            if a == "two-sided":
                itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
            elif a == "less":
                itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
            else:
                itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)

            result.append({
                "test": n,
                "alternative": a,
                "statistic": round(s, 3) if not np.isnan(s) else s,
                "p-value": round(p, 3) if not np.isnan(p) else p,
                "H0": p > 0.05,
                "H1": p <= 0.05,
                "interpretation": itp,
                "equal_var_checked": var_checked
            })
        except Exception as e:
            result.append({
                "test": "t-test_paired" if equal_var else "Wilcoxon signed-rank",
                "alternative": a,
                "statistic": np.nan,
                "p-value": np.nan,
                "H0": False,
                "H1": False,
                "interpretation": f"검정 실패: {str(e)}",
                "equal_var_checked": var_checked
            })

    rdf = DataFrame(result)
    rdf.set_index(["test", "alternative"], inplace=True)

    return rdf


# -------------------------------------------------------------
def hs_vif_filter(
    data: DataFrame,
    yname: str = None,
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
        기본 사용 예:

        >>> from hossam.analysis import hs_vif_filter
        >>> filtered = hs_vif_filter(df, yname="target", ignore=["id"], threshold=10.0)
        >>> filtered.head()
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

    def _compute_vifs(X_: DataFrame) -> dict:
        # NA 제거 후 상수항 추가
        X_clean = X_.dropna()
        if X_clean.shape[0] == 0:
            # 데이터가 모두 NA인 경우 VIF 계산 불가: NaN 반환
            return {col: np.nan for col in X_.columns}
        if X_clean.shape[1] == 1:
            # 단일 예측변수의 경우 다른 설명변수가 없으므로 VIF는 1로 간주
            return {col: 1.0 for col in X_clean.columns}
        exog = sm.add_constant(X_clean, prepend=True)
        vifs = {}
        for i, col in enumerate(X_clean.columns, start=0):
            # exog의 첫 열은 상수항이므로 변수 인덱스는 +1
            try:
                vifs[col] = float(variance_inflation_factor(exog.values, i + 1))
            except Exception:
                # 계산 실패 시 무한대로 처리하여 우선 제거 대상으로
                vifs[col] = float("inf")
        return vifs

    # 반복 제거 루프
    while True:
        if X.shape[1] == 0:
            break
        vifs = _compute_vifs(X)
        if verbose:
            print(vifs)
        # 모든 변수가 임계값 이하이면 종료
        max_key = max(vifs, key=lambda k: (vifs[k] if not np.isnan(vifs[k]) else -np.inf))
        max_vif = vifs[max_key]
        if np.isnan(max_vif) or max_vif <= threshold:
            break
        # 가장 큰 VIF 변수 제거
        X = X.drop(columns=[max_key])

    # 출력 옵션이 False일 경우 최종 값만 출력
    if not verbose:
        final_vifs = _compute_vifs(X) if X.shape[1] > 0 else {}
        print(final_vifs)

    # 원본 컬럼 순서 유지하며 제거된 수치형 컬럼만 제외
    kept_numeric_cols = list(X.columns)
    removed_numeric_cols = [c for c in numeric_cols if c not in kept_numeric_cols]
    result = data.drop(columns=removed_numeric_cols, errors="ignore")

    return result


# -------------------------------------------------------------
def hs_trend(x: any, y: any, degree: int = 1, value_count: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """x, y 데이터에 대한 추세선을 구한다.

    Args:
        x (_type_): 산점도 그래프에 대한 x 데이터
        y (_type_): 산점도 그래프에 대한 y 데이터
        degree (int, optional): 추세선 방정식의 차수. Defaults to 1.
        value_count (int, optional): x 데이터의 범위 안에서 간격 수. Defaults to 100.

    Returns:
        tuple: (v_trend, t_trend)

    Examples:
        2차 다항 회귀 추세선:

        >>> from hossam.analysis import hs_trend
        >>> vx, vy = hs_trend(x, y, degree=2, value_count=200)
        >>> len(vx), len(vy)
        (200, 200)
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


# -------------------------------------------------------------
def hs_linear_report(fit, data):
    """선형회귀 적합 결과를 요약 리포트로 변환한다.

    Args:
        fit: statsmodels OLS 등 선형회귀 결과 객체 (`fit.summary()`를 지원해야 함).
        data: 종속변수와 독립변수를 모두 포함한 DataFrame.

    Returns:
        tuple: 다음 요소를 포함한다.
            - 회귀계수 표 (`rdf`, DataFrame): 변수별 B, 표준오차, Beta, t, p-value, 공차, VIF.
            - 적합도 요약 (`result_report`, str): R, R², F, p-value, Durbin-Watson 등 핵심 지표 문자열.
            - 모형 보고 문장 (`model_report`, str): F-검정 유의성에 기반한 서술형 문장.
            - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수에 대한 서술형 문장.
            - 회귀식 문자열 (`equation_text`, str): 상수항과 계수를 포함한 회귀식 표현.

    Examples:
        >>> import statsmodels.api as sm
        >>> y = data['target']
        >>> X = sm.add_constant(data[['x1', 'x2']])
        >>> fit = sm.OLS(y, X).fit()
        >>> rdf, result_report, model_report, variable_reports, eq = hs_linear_report(fit, data)
        >>> print(eq)
    """

    tbl = fit.summary()

    # 종속변수 이름
    yname = fit.model.endog_names

    # 독립변수 이름(상수항 제외)
    xnames = [n for n in fit.model.exog_names if n != "const"]

    # 독립변수 부분 데이터 (VIF 계산용)
    indi_df = data.filter(xnames)

    # 독립변수 결과를 누적
    variables = []
    for i, v in enumerate(tbl.tables[1].data):
        # 한 행의 변수명 추출 후 목록에 있는지 확인
        name = v[0].strip()
        if name not in xnames:
            continue

        # VIF 계산: 상수항을 포함한 설계행렬에서 대상 변수의 열 인덱스를 사용
        indi_df_const = sm.add_constant(indi_df, has_constant="add")
        j = list(indi_df_const.columns).index(name)
        vif = variance_inflation_factor(indi_df_const.values, j)

        # 유의확률과 별표 표시 함수
        p = float(v[4].strip())
        stars = lambda p: (
            "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        )

        # 한 변수에 대한 보고 정보 추가
        variables.append(
            {
                "종속변수": yname,  # 종속변수 이름
                "독립변수": name,  # 독립변수 이름
                "B": v[1].strip(),  # 비표준화 회귀계수(B)
                "표준오차": v[2].strip(),  # 계수 표준오차
                "Beta": float(fit.params[name])
                * (
                    data[name].std(ddof=1) / data[yname].std(ddof=1)
                ),  # 표준화 회귀계수(β)
                "t": "%s%s" % (v[3].strip(), stars(p)),  # t-통계량(+별표)
                "p-value": p,  # 계수 유의확률
                "공차": 1 / vif,  # 공차(Tolerance = 1/VIF)
                "vif": vif,  # 분산팽창계수
            }
        )

    rdf = DataFrame(variables)

    # summary 표에서 적합도 정보를 key-value로 추출
    result_dict = {}
    for i in [0, 2]:
        for item in tbl.tables[i].data:
            n = len(item)
            for i in range(0, n, 2):
                key = item[i].strip()[:-1]
                value = item[i + 1].strip()
                if not key or not value:
                    continue
                result_dict[key] = value

    # 적합도 보고 문자열 구성
    result_report = f"𝑅({result_dict['R-squared']}), 𝑅^2({result_dict['Adj. R-squared']}), 𝐹({result_dict['F-statistic']}), 유의확률({result_dict['Prob (F-statistic)']}), Durbin-Watson({result_dict['Durbin-Watson']})"

    # 모형 보고 문장 구성
    tpl = "%s에 대하여 %s로 예측하는 회귀분석을 실시한 결과, 이 회귀모형은 통계적으로 %s(F(%s,%s) = %s, p %s 0.05)."
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
        result_dict["F-statistic"],
        "<=" if float(result_dict["Prob (F-statistic)"]) <= 0.05 else ">",
    )

    # 변수별 보고 문장 리스트 구성
    variable_reports = []
    s = "%s의 회귀계수는 %s(p %s 0.05)로, %s에 대하여 %s 예측변인인 것으로 나타났다."

    for i in rdf.index:
        row = rdf.iloc[i]
        variable_reports.append(
            s
            % (
                row["독립변수"],
                row["B"],
                "<=" if float(row["p-value"]) < 0.05 else ">",
                row["종속변수"],
                "유의미한" if float(row["p-value"]) < 0.05 else "유의하지 않은",
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

    return rdf, result_report, model_report, variable_reports, equation_text


# -------------------------------------------------------------
def hs_logit_report(fit, data, threshold=0.5):
    """로지스틱 회귀 적합 결과를 상세 리포트로 변환한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.summary()`와 예측 확률을 지원해야 함).
        data: 종속변수와 독립변수를 모두 포함한 DataFrame.
        threshold: 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.

    Returns:
        tuple: 다음 요소를 포함한다.
            - 성능 지표 표 (`cdf`, DataFrame): McFadden Pseudo R², Accuracy, Precision, Recall, FPR, TNR, AUC, F1.
            - 회귀계수 표 (`rdf`, DataFrame): B, 표준오차, z, p-value, OR, 95% CI, VIF 등.
            - 적합도 및 예측 성능 요약 (`result_report`, str): Pseudo R², LLR χ², p-value, Accuracy, AUC.
            - 모형 보고 문장 (`model_report`, str): LLR p-value에 기반한 서술형 문장.
            - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수의 오즈비 해석 문장.

    Examples:
        >>> import statsmodels.api as sm
        >>> y = data['target']
        >>> X = sm.add_constant(data[['x1', 'x2']])
        >>> fit = sm.Logit(y, X).fit(disp=0)
        >>> cdf, rdf, result_report, model_report, variable_reports = hs_logit_report(fit, data, threshold=0.5)
        >>> print(variable_reports[0])
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
    tbl = fit.summary()

    # 독립변수 이름(상수항 제외)
    xnames = [n for n in fit.model.exog_names if n != "const"]

    # 독립변수
    x = data[xnames]

    variables = []

    # VIF 계산 (상수항 포함 설계행렬 사용)
    vif_dict = {}
    x_const = sm.add_constant(x, has_constant="add")
    for col in x.columns:
        col_idx = list(x_const.columns).index(col)
        vif_dict[col] = variance_inflation_factor(x_const.values, col_idx)

    for v in tbl.tables[1].data:
        name = v[0].strip()
        if name not in xnames:
            continue

        beta = float(v[1])
        se = float(v[2])
        z = float(v[3])
        p = float(v[4])

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

    return cdf, rdf, result_report, model_report, variable_reports