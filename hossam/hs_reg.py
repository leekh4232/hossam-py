import seaborn as sb
import numpy as np
import shap

from pandas import DataFrame, Series, merge, concat
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# 성능 평가 지표 모듈
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from .hs_plot import create_figure, finalize_plot, barplot, config


# --------------------------------------------------------
# 회귀 성능 평가 지표 함수
# --------------------------------------------------------
def get_scores(
    estimator, x_test: DataFrame, y_test: DataFrame | np.ndarray
) -> DataFrame:
    """
    회귀 성능 평가 지표 함수
    수업에서 사용된 hs_get_scores 함수와 동일.

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
    """학습곡선 기반 과적합 판별 함수.
    수업에서 사용된 hs_learning_cv 함수와 동일.

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

    ax.set_xlabel("학습 데이터 비율", fontsize=8, labelpad=5)  # type : ignore
    ax.set_ylabel("RMSE", fontsize=8, labelpad=5)  # type : ignore
    ax.grid(True, alpha=0.3)  # type : ignore

    finalize_plot(ax, title=f"{classname} 학습곡선 (Learning Curve)")

    return result_df

# --------------------------------------------------------
# 회귀 성능 평가 + 과적합 판별 통합 함수
# --------------------------------------------------------
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
    회귀 성능 평가 지표 함수.
    수업에서 사용된 hs_get_score_cv 함수와 동일.

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

    if type(estimator) != list:
        estimator = [estimator]

    result_df = DataFrame()

    for est in estimator:
        score_df = get_scores(est, x_test, y_test)
        cv_df = learning_cv(
            est,
            x_origin,
            y_origin,
            scoring=scoring,
            cv=cv,
            train_sizes=train_sizes,
            n_jobs=n_jobs,
        )

        result_df = concat(
            [result_df, merge(score_df, cv_df, left_index=True, right_index=True)]
        )

    return result_df

# --------------------------------------------------------
# 특징 중요도 분석 함수
# --------------------------------------------------------
def feature_importance(
    model,
    x_train: DataFrame,
    y_train: DataFrame | np.ndarray | Series,
    plot: bool = True,
) -> DataFrame:
    perm = permutation_importance(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="r2",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )
    """
    특징 중요도 분석 함수.    
    수업에서 사용된 hs_feature_importance 함수와 동일.

    Args:
        model: 학습된 사이킷런 회귀 모델
        x_train: 학습용 설명변수 데이터 (DataFrame)
        y_train: 학습용 목표변수 값 (DataFrame, Series 또는 ndarray)
        plot: 중요도 시각화 여부 (기본값: True)

    Returns:
        DataFrame: 특징 중요도 결과 표
    """

    # 결과 정리
    perm_df = DataFrame(
        {
            "importance_mean": perm.importances_mean,  # type: ignore
            "importance_std": perm.importances_std,  # type: ignore
        },
        index=x_train.columns,
    ).sort_values("importance_mean", ascending=False)

    perm_df["importance_cumsum"] = perm_df["importance_mean"].cumsum()

    # 시각화
    if plot:
        barplot(
            df=perm_df,
            xname="importance_mean",
            yname=perm_df.index,
            title="Permutation Importance",
            callback=lambda ax: ax.set_xlabel("Permutation Importance (mean)"),
        )

    return perm_df


# --------------------------------------------------------
# SHAP 값 기반 특징 중요도 분석 함수
# --------------------------------------------------------
def shap_analysis(
    model,
    x: DataFrame,
    plot: bool = True,
    width: int = config.width,
    height: int = config.height,
) -> tuple[DataFrame, np.ndarray]:
    """
    SHAP 값 기반 특징 중요도 분석 함수.    
    수업에서 사용된 hs_shap_analysis 함수와 동일.

    Args:
        model: 학습된 사이킷런 회귀 모델
        x: 설명변수 데이터 (DataFrame)
        plot: SHAP 요약 플롯 시각화 여부 (기본값: True
        width: 플롯 너비 (기본값: config.width)
        height: 플롯 높이 (기본값: config.height)

    Returns:
        tuple: 특징 중요도 요약 DataFrame 및 SHAP 값 배열
    """
    # 1. SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # 2. SHAP 값 계산: shape = [n_samples, n_features]
    shap_values = explainer.shap_values(x)

    # 3. DataFrame 변환
    shap_df = DataFrame(
        shap_values,
        columns=x.columns,
        index=x.index,
    )

    # 4. 요약 통계
    summary_df = DataFrame(
        {
            "feature": shap_df.columns,
            "mean_abs_shap": shap_df.abs().mean().values,
            "mean_shap": shap_df.mean().values,
            "std_shap": shap_df.std().values,
        }
    )

    # 5. 영향 방향 (보수적 표현)
    summary_df["direction"] = np.where(
        summary_df["mean_shap"] > 0,
        "양(+) 경향",
        np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"),
    )

    # 6. 변동성 지표
    summary_df["cv"] = summary_df["std_shap"] / (summary_df["mean_abs_shap"] + 1e-9)

    summary_df["variability"] = np.where(
        summary_df["cv"] < 1,
        "stable",  # 변동성 낮음 - 평균 대비 일관적 영향 의미
        "variable",  # 변동성 큼 - 상황 의존적 영향 의미
    )

    # 7. 중요도 기준 정렬
    summary_df = summary_df.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )

    # 8. 중요 변수 표시 (누적 80%)
    total_importance = summary_df["mean_abs_shap"].sum()
    summary_df["importance_ratio"] = summary_df["mean_abs_shap"] / total_importance
    summary_df["importance_cumsum"] = summary_df["importance_ratio"].cumsum()

    summary_df["is_important"] = np.where(
        summary_df["importance_cumsum"] <= 0.80,
        "core",  # 누적 80% 내 중요 변수 - 모델 핵심 결정 요인 의미 명확
        "secondary",  # 누적 80% 초과 변수 - 보조적/상황적 영향 요인 의미
    )

    # 9. 시각화
    if plot:
        shap.summary_plot(shap_values, x, show=False)

        fig = plt.gcf()
        fig.set_size_inches(width / config.dpi, height / config.dpi)
        ax = fig.get_axes()[0]

        plt.xlabel("SHAP value", fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)

        finalize_plot(ax, title="SHAP Summary Plot")

    return summary_df, shap_values


# --------------------------------------------------------
# SHAP 값 기반 특징 의존성 분석 함수
# --------------------------------------------------------

def shap_dependence_analysis(
    summary_df: DataFrame,
    shap_values: np.ndarray,
    x: DataFrame,
    include_secondary: bool = False,
    width: int = config.width,
    height: int = config.height,
):
    """
    SHAP 값 기반 특징 의존성 분석 함수
    수업에서 사용된 hs_shap_dependence_analysis 함수와 동일.

    Args:
        summary_df: shap_analysis 함수의 요약 DataFrame
        shap_values: shap_analysis 함수의 SHAP 값 배열
        x: 설명변수 데이터 (DataFrame)
        include_secondary: 상호작용 변수에 보조 변수 포함 여부 (기본값: False)
        width: 플롯 너비 (기본값: config.width)
        height: 플롯 높이 (기본값: config.height)   

    Returns:
        list: 생성된 특징 상호작용 쌍 목록
    """

    # 1. 주 대상 변수 (Core + Variable)
    main_features = summary_df[
        (summary_df["is_important"] == "core")
        & (summary_df["variability"] == "variable")
    ]["feature"].tolist()

    # 2. 상호작용 후보 변수
    interaction_features = summary_df[
        summary_df["is_important"] == "core"
    ]["feature"].tolist()

    if include_secondary and len(interaction_features) < 2:
        interaction_features.extend(
            summary_df[
                summary_df["is_important"] == "secondary"
            ]["feature"].tolist()
        )

    # 3. 변수 쌍 생성 (자기 자신 제외)
    pairs = []

    for f in main_features:
        for inter in interaction_features:
            # 자기 자신과의 조합은 제외
            if f != inter:
                pairs.append((f, inter))

    # paris를 역순으로 순회하며 중복 제거
    seen = set()
    for i in range(len(pairs) - 1, -1, -1):
        key = frozenset(pairs[i])
        if key in seen:
            del pairs[i]
        else:
            seen.add(key)

    # 중요도 순 정렬 (주 변수 기준)
    importance_rank = {}

    for i, row in summary_df.iterrows():
        importance_rank[row["feature"]] = i

    pairs = sorted(
        pairs,
        key=lambda k: importance_rank.get(k[0], 999)
    )

    # 4. dependence plot 일괄 생성
    for feature_name, interaction_name in pairs:
        shap.dependence_plot(
            feature_name,
            shap_values,
            x,
            interaction_index=interaction_name,
            show=False,
        )

        # SHAP figure 직접 제어
        fig = plt.gcf()
        fig.set_size_inches(width / config.dpi, height / config.dpi)
        ax = fig.get_axes()[0]

        plt.xlabel(feature_name, fontsize=10)
        plt.ylabel(f"SHAP value for {feature_name}", fontsize=10)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)

        finalize_plot(ax, title=f"SHAP Dependence Plot: {feature_name} vs {interaction_name}")

    return pairs
