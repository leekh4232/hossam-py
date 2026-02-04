from IPython.display import display

from pandas import DataFrame, merge
import seaborn as sb
import numpy as np

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import learning_curve

# 성능 평가 지표 모듈
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from .hs_plot import create_figure, finalize_plot


# --------------------------------------------------------
# VIF 기반 다중공선성 제거기
# --------------------------------------------------------
class VIFSelector(BaseEstimator, TransformerMixin):
    """
    VIF(Variance Inflation Factor) 기반 다중공선성 제거기

    Args:
        threshold (float): VIF 임계값 (기본값: 10.0
        check_cols (list or None): VIF 계산에 사용할 열 목록 (기본값: None, 모든 열 사용)

    Attributes:
        drop_cols_ (list): 제거된 열 목록
        vif_cols_ (list): VIF 계산에 사용된 열 목록

    """

    def __init__(self, threshold=10.0, check_cols=None):
        self.threshold = threshold
        self.check_cols = check_cols

    def _compute_vifs(self, X: DataFrame):
        exog = sm.add_constant(X, prepend=True)

        vifs = {}
        for i, col in enumerate(X.columns):
            try:
                vifs[col] = float(variance_inflation_factor(exog.values, i + 1))
            except Exception:
                vifs[col] = float("inf")

        vdf = DataFrame(vifs.items(), columns=["Variable", "VIF"])
        return vdf.sort_values("VIF", ascending=False)

    def fit(self, X, y=None):
        df = X.copy().dropna()

        self.vif_cols_ = self.check_cols if self.check_cols else df.columns.tolist()
        X_vif = df[self.vif_cols_].copy()

        self.drop_cols_ = []
        i = 0

        while True:
            if X_vif.shape[1] == 0:
                break

            vdf = self._compute_vifs(X_vif)
            max_vif = vdf.iloc[0]["VIF"]
            max_col = vdf.iloc[0]["Variable"]

            if max_vif <= self.threshold:
                # print(
                #     "모든 변수의 VIF가 임계값 이하가 되어 종료합니다. 제거된 변수 {0}개.".format(
                #         i
                #     )
                # )
                break

            X_vif = X_vif.drop(columns=[max_col])
            self.drop_cols_.append(max_col)
            #print(f"제거된 변수: {max_col} (VIF={X_vif:.2f})")
            i += 1

        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols_, errors="ignore")


# --------------------------------------------------------
# 회귀 성능 평가 지표 함수
# --------------------------------------------------------
def get_scores(
    estimator,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray
) -> DataFrame:
    """
    회귀 성능 평가 지표 함수

    Args:
        estimator: 학습된 사이킷런 회귀 모델
        x_test: 테스트용 설명변수 데이터 (DataFrame)
        y_test: 실제 목표변수 값 (DataFrame 또는 ndarray)

    Returns:
        DataFrame: 회귀 성능 평가 지표 (R2, MAE, MSE, RMSE, MAPE, MPE)
    """
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    y_pred = estimator.predict(x_test)

    score_df = DataFrame(
        {
            "결정계수(R2)": r2_score(y_test, y_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_test, y_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_test, y_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_test, y_pred)),
            "평균 절대 백분오차 비율(MAPE)": mean_absolute_percentage_error(
                y_test, y_pred
            ),
            "평균 비율 오차(MPE)": np.mean((y_test - y_pred) / y_test * 100),
        },
        index=[classname],
    )

    return score_df


# --------------------------------------------------------
# 학습곡선기반 과적합 판별 함수
# --------------------------------------------------------
def learning_cv(
    estimator,
    x,
    y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
) -> DataFrame:
    """학습곡선 기반 과적합 판별 함수

    Args:
        estimator: 사이킷런 Estimator (파이프라인 권장)
        x: 설명변수 (DataFrame 또는 ndarray)
        y: 목표변수 (Series 또는 ndarray)
        scoring: 평가 지표 (기본값: neg_root_mean_squared_error)
        cv: 교차검증 폴드 수 (기본값: 5)
        train_sizes: 학습곡선 학습 데이터 비율 (기본값: np.linspace(0.1, 1.0, 10))
        n_jobs: 병렬 처리 개수 (기본값: -1, 모든 CPU 사용)

    Returns:
        DataFrame: 과적합 판별 결과 표
    """

    train_sizes, train_scores, cv_scores = learning_curve(  # type: ignore
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=52,
    )

    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # neg RMSE → RMSE
    train_rmse = -train_scores
    cv_rmse = -cv_scores

    # 평균 / 표준편차
    train_mean = train_rmse.mean(axis=1)
    cv_mean = cv_rmse.mean(axis=1)
    cv_std = cv_rmse.std(axis=1)

    # 마지막 지점 기준 정량 판정
    final_train = train_mean[-1]
    final_cv = cv_mean[-1]
    final_std = cv_std[-1]
    gap_ratio = final_train / final_cv
    var_ratio = final_std / final_cv

    # -----------------
    # 과소적합 기준선 (some_threshold)
    # -----------------
    # 기준모형 RMSE (평균 예측)
    y_mean = y.mean()
    rmse_naive = np.sqrt(np.mean((y - y_mean) ** 2))

    # 분산 기반
    std_y = y.std()

    # 최소 설명력(R²) 기반
    min_r2 = 0.10
    rmse_r2 = np.sqrt((1 - min_r2) * np.var(y))

    # 최종 threshold (가장 관대한 기준)
    # -> 원래 some_threshold는 도메인 지식 수준에서 이 모델은 최소 어느 정도의 성능은 내야 한다는 기준을 설정하는 것
    some_threshold = min(rmse_naive, std_y, rmse_r2)

    # -----------------
    # 판정 로직
    # -----------------
    if gap_ratio >= 0.95 and final_cv > some_threshold:
        status = "⚠️ 과소적합 (bias 큼)"
    elif gap_ratio <= 0.8:
        status = "⚠️ 과대적합 (variance 큼)"
    elif gap_ratio <= 0.95 and var_ratio <= 0.10:
        status = "✅ 일반화 양호"
    elif var_ratio > 0.15:
        status = "⚠️ 데이터 부족 / 분산 큼"
    else:
        status = "⚠️ 판단 유보"

    # -----------------
    # 정량 결과 표
    # -----------------
    result_df = DataFrame(
        {
            "Train RMSE": [final_train],
            "CV RMSE 평균": [final_cv],
            "CV RMSE 표준편차": [final_std],
            "Train/CV 비율": [gap_ratio],
            "CV 변동성 비율": [var_ratio],
            "판정 결과": [status],
        },
        index=[classname],
    )

    # -----------------
    # 학습곡선 시각화
    # -----------------
    fig, ax = create_figure()

    sb.lineplot(
        x=train_sizes,
        y=train_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="Train RMSE",
    )
    sb.lineplot(
        x=train_sizes,
        y=cv_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="CV RMSE",
    )

    ax.set_xlabel("RMSE", fontsize=8, labelpad=5)   # type : ignore
    ax.set_ylabel("학습곡선 (Learning Curve)", fontsize=8, labelpad=5)  # type : ignore
    ax.grid(True, alpha=0.3) # type : ignore

    finalize_plot(ax)

    return result_df


def get_score_cv(
    estimator,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    x_origin: DataFrame,
    y_origin: DataFrame | np.ndarray,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
) -> DataFrame:
    """
    회귀 성능 평가 지표 함수

    Args:
        estimator: 학습된 사이킷런 회귀 모델
        x_test: 테스트용 설명변수 데이터 (DataFrame)
        y_test: 실제 목표변수 값 (DataFrame 또는 ndarray)
        x_origin: 학습곡선용 전체 설명변수 데이터 (DataFrame, learning_curve=True일 때 필요)
        y_origin: 학습곡선용 전체 목표변수 값 (DataFrame 또는 ndarray, learning_curve=True일 때 필요)
        scoring: 학습곡선 평가 지표 (기본값: neg_root_mean_squared_error)
        cv: 학습곡선 교차검증 폴드 수 (기본값: 5)
        train_sizes: 학습곡선 학습 데이터 비율 (기본값: np.linspace(0.1, 1.0, 10))
        n_jobs: 학습곡선 병렬 처리 개수 (기본값: -1, 모든 CPU 사용)

    Returns:
        DataFrame: 회귀 성능 평가 지표 + 과적합 판정 여부
    """

    score_df = get_scores(estimator, x_test, y_test)
    cv_df = learning_cv(
        estimator,
        x_origin,
        y_origin,
        scoring=scoring,
        cv=cv,
        train_sizes=train_sizes,
        n_jobs=n_jobs,
    )

    return merge(score_df, cv_df, left_index=True, right_index=True)