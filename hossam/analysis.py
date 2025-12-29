# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import numpy as np
from typing import Tuple
from pandas import DataFrame
from pandas.api.types import is_bool_dtype
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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