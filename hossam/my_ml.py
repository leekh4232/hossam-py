import joblib
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pandas import DataFrame, concat
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.model_selection import learning_curve as sk_learning_curve, cross_validate, GroupKFold

from . import my_plot
from .my_vif_selector import VIFSelector
from .my_outlier_clipper import OutlierClipper
from .my_ml_const import (
    RANDOM_STATE,
    _VALID_NUMERIC_IMPUTE, _VALID_CATEGORICAL_IMPUTE, _VALID_OUTLIER,
    _METRIC_SPECS, _CV_SCORERS,
    _IMPORTANCE_TREE, _IMPORTANCE_LINEAR, _FI_SCALER_CLASSES,
)

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_squared_log_error, mean_absolute_percentage_error
)


def build_pipeline(model, x_train, y_train=None, x_test=None, y_test=None,
                   numeric_impute='median',
                   categorical_impute='most_frequent',
                   outlier='iqr',
                   scale=False,
                   vif_threshold=None,
                   pca_variance=None,
                   onehot=False,
                   drop_first=False,
                   verbose=True):
    """모델과 전처리 옵션을 받아 sklearn 파이프라인을 구성해 반환한다.

    전처리 순서 — 수치형: impute → outlier → scale → vif → pca,
    명목형: impute → onehot.

    Args:
        model: 생성된 sklearn 모델 인스턴스
        x_train: 훈련 피처 (DataFrame)
        y_train, x_test, y_test: 검증용 (선택). 주면 형태 일관성을 검사한다.
        numeric_impute: 수치형 결측치 전략 — 'mean'/'median'/'most_frequent'/'constant'/None.
        categorical_impute: 명목형 결측치 전략 — 'most_frequent'/'constant'/None.
        outlier: 이상치 클리핑 — 'iqr'(Q1-1.5IQR~Q3+1.5IQR)/'zscore'(±3σ)/None.
        scale: True 면 StandardScaler.
        vif_threshold: float 주면 VIFSelector(threshold) 적용 (예: 10.0).
        pca_variance: float(0~1) 주면 PCA(n_components) 적용 (예: 0.95).
        onehot: True 면 OneHotEncoder. False 면 명목형 원본 유지 (CatBoost 용).
        drop_first: True 면 OneHotEncoder(drop='first') — 더미 트랩 방지. onehot=False 면 무시.
        verbose: 구성 정보 출력 여부.

    권장 조합: 선형/SGD → scale·vif·drop_first / 거리·SVM → +pca_variance /
    트리·부스팅 → 기본값 / CatBoost → onehot=False.

    Returns:
        Pipeline: .pipeline_info(dict) 속성에 구성 정보가 부착된 파이프라인.

    Raises:
        TypeError: x_train/x_test 가 DataFrame 이 아닌 경우.
        ValueError: 데이터 불일치 또는 옵션 값이 유효하지 않은 경우.
    """
    # ============ 입력 검증 ============
    if not isinstance(x_train, DataFrame):
        raise TypeError(f"x_train must be pandas.DataFrame, got {type(x_train).__name__}")

    if x_train.empty:
        raise ValueError("x_train cannot be empty")

    if y_train is not None and len(x_train) != len(y_train):
        raise ValueError(f"x_train and y_train must have same length: {len(x_train)} vs {len(y_train)}")

    if x_test is not None:
        if not isinstance(x_test, DataFrame):
            raise TypeError(f"x_test must be pandas.DataFrame, got {type(x_test).__name__}")
        if x_train.columns.tolist() != x_test.columns.tolist():
            raise ValueError(
                f"x_train and x_test must have same columns.\n"
                f"  x_train columns: {x_train.columns.tolist()}\n"
                f"  x_test columns: {x_test.columns.tolist()}"
            )
        if y_test is not None and len(x_test) != len(y_test):
            raise ValueError(f"x_test and y_test must have same length: {len(x_test)} vs {len(y_test)}")

    # ============ 파라미터 검증 ============
    if numeric_impute not in _VALID_NUMERIC_IMPUTE:
        raise ValueError(
            f"numeric_impute must be one of {sorted(str(v) for v in _VALID_NUMERIC_IMPUTE)}, "
            f"got {numeric_impute!r}"
        )

    if categorical_impute not in _VALID_CATEGORICAL_IMPUTE:
        raise ValueError(
            f"categorical_impute must be one of {sorted(str(v) for v in _VALID_CATEGORICAL_IMPUTE)}, "
            f"got {categorical_impute!r}"
        )

    if outlier not in _VALID_OUTLIER:
        raise ValueError(
            f"outlier must be one of {sorted(str(v) for v in _VALID_OUTLIER)}, "
            f"got {outlier!r}"
        )

    if vif_threshold is not None and vif_threshold <= 0:
        raise ValueError(f"vif_threshold must be positive, got {vif_threshold}")

    if pca_variance is not None and not 0.0 < pca_variance <= 1.0:
        raise ValueError(f"pca_variance must be in range (0.0, 1.0], got {pca_variance}")

    if drop_first:
        onehot = True  # drop_first=True 면 onehot=True 이어야 의미가 있음

    # ============ 훈련 데이터에서 피처 타입 자동 판단 ============
    numeric_features = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("x_train must have at least one numeric or categorical feature")

    # ============ 수치형 전처리 파이프라인 구성 ============
    numeric_steps = []
    if numeric_impute is not None:
        numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_impute)))
    if outlier is not None:
        numeric_steps.append(('outlier_clipper', OutlierClipper(method=outlier)))
    if scale:
        numeric_steps.append(('scaler', StandardScaler()))
    if vif_threshold is not None:
        numeric_steps.append(('vif_selector', VIFSelector(threshold=vif_threshold)))
    if pca_variance is not None:
        numeric_steps.append(('pca', PCA(n_components=pca_variance, random_state=RANDOM_STATE)))

    numeric_pipe = Pipeline(numeric_steps) if numeric_steps else 'passthrough'

    # ============ 명목형 전처리 파이프라인 구성 ============
    categorical_steps = []
    if categorical_impute is not None:
        categorical_steps.append(('imputer', SimpleImputer(strategy=categorical_impute)))

    if onehot:
        categorical_steps.append(('onehot', OneHotEncoder(
            drop='first' if drop_first else None,
            handle_unknown='ignore',
            sparse_output=False,
        )))
    categorical_pipe = Pipeline(categorical_steps) if categorical_steps else 'passthrough'

    # ============ ColumnTransformer + 최종 파이프라인 ============
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', categorical_pipe, categorical_features),
    ], remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)

    preprocessor.set_output(transform='pandas')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    # ============ 메타데이터 부착 ============
    model_name = type(model).__name__
    pipeline.pipeline_info = {
        'model_class': model_name,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'num_numeric_features': len(numeric_features),
        'num_categorical_features': len(categorical_features),
        'total_features': len(numeric_features) + len(categorical_features),
        'numeric_impute': numeric_impute,
        'categorical_impute': categorical_impute,
        'outlier': outlier,
        'scale': scale,
        'vif_threshold': vif_threshold,
        'pca_variance': pca_variance,
        'onehot': onehot,
        'drop_first': drop_first,
    }

    # ============ 파이프라인 정보 출력 ============
    if verbose:
        print("\n" + "="*70)
        print(f"◆ Pipeline Configuration: {model_name}")
        print("="*70)
        print(f"\n▲ Features:")
        print(f"   - Numeric    : {len(numeric_features)} - {numeric_features}")
        print(f"   - Categorical: {len(categorical_features)} - {categorical_features}")

        print(f"\n◉ Numeric Preprocessing:")
        if numeric_impute is not None:
            print(f"   - Imputation       : SimpleImputer(strategy='{numeric_impute}')")
        if outlier is not None:
            print(f"   - Outlier clipping : OutlierClipper(method='{outlier}')")
        if scale:
            print(f"   - Scaling          : StandardScaler")
        if vif_threshold is not None:
            print(f"   - VIF selection    : threshold={vif_threshold}")
        if pca_variance is not None:
            print(f"   - PCA              : variance={pca_variance}")
        if not numeric_steps:
            print(f"   (passthrough)")

        print(f"\n◉ Categorical Preprocessing:")
        if categorical_impute is not None:
            print(f"   - Imputation       : SimpleImputer(strategy='{categorical_impute}')")
        if onehot:
            onehot_mode = "drop='first' (prevent dummy trap)" if drop_first else "drop=None (keep all dummies)"
            print(f"   - OneHotEncoding   : {onehot_mode}")
        else:
            print(f"   - OneHotEncoding   : DISABLED (passthrough — for CatBoost etc.)")
        if not categorical_steps:
            print(f"   (passthrough)")
        print("="*70 + "\n")

    return pipeline


# --------------------------------------------------------
# 모델 저장 / 로드 함수
# --------------------------------------------------------
def save_model(model, save_path):
    """학습된 모델을 joblib 으로 직렬화해 저장한다. 상위 디렉토리는 자동 생성.

    Args:
        model: 저장할 sklearn 모델/파이프라인
        save_path (str | Path): 저장 경로 (.pkl 권장)

    Returns:
        Path: 저장된 파일의 절대 경로.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    return save_path.resolve()


def load_model(load_path):
    """저장된 모델 파일을 로드해서 반환한다.

    Args:
        load_path (str | Path): 모델 파일 경로

    Returns:
        저장 시점의 모델 객체.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    return joblib.load(load_path)


# --------------------------------------------------------
# 회귀 성능 평가 지표 함수
# --------------------------------------------------------
def reg_score(
    estimator, x_test: DataFrame, y_test: DataFrame | np.ndarray
) -> DataFrame:
    """회귀 모델의 성능 지표(R2/MAE/MSE/RMSE/RMSLE/MAPE/MPE)를 계산한다.

    Args:
        estimator: 학습된 사이킷런 회귀 모델
        x_test: 테스트 설명변수 (DataFrame)
        y_test: 실제 목표변수 (DataFrame 또는 ndarray)

    Returns:
        DataFrame: 모델명을 인덱스로 하는 지표 1행. 음수/0 으로 계산 불가한
            지표(RMSLE, MAPE/MPE)는 NaN.
    """
    # 모델 클래스명 추출
    if hasattr(estimator, "named_steps"):
        classname = estimator.named_steps["model"].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # 예측값 계산
    y_pred = estimator.predict(x_test)

    # y_test를 1D 배열로 변환
    if isinstance(y_test, DataFrame):
        y_test_array = y_test.values.ravel()
    else:
        y_test_array = np.asarray(y_test).ravel()

    # 기본 성능 지표
    r2 = r2_score(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    mse = mean_squared_error(y_test_array, y_pred)
    rmse = np.sqrt(mse)

    # MSLE/RMSLE (음수값 체크)
    if np.any(y_test_array < 0) or np.any(y_pred < 0):
        msle = np.nan
        rmsle = np.nan
    else:
        msle = mean_squared_log_error(y_test_array, y_pred)
        rmsle = np.sqrt(msle)

    # MAPE/MPE (0값 체크)
    if np.any(y_test_array == 0):
        mape = np.nan
        mpe = np.nan
    else:
        # sklearn 의 MAPE 는 비율(ratio)을 반환하므로 백분율로 변환 (MPE 와 단위 일치)
        mape = mean_absolute_percentage_error(y_test_array, y_pred) * 100
        mpe = np.mean((y_test_array - y_pred) / y_test_array) * 100

    # 결과 DataFrame 생성
    score_dict = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "RMSLE": rmsle,
        "MAPE": mape,
        "MPE": mpe,
    }

    score_df = DataFrame(score_dict, index=[classname])
    score_df.index.name = 'Model'  # 인덱스 이름 설정

    return score_df

def _unwrap_estimator(estimator):
    """GridSearchCV / RandomizedSearchCV 등 search 객체면 best_estimator_ 로 풀어준다.

    Returns:
        (effective_estimator, search_info|None): search 객체면 best_estimator_ 와
            {'search_class','best_params','best_score','scoring'} dict, 아니면 (원본, None).
    """
    if hasattr(estimator, 'best_estimator_'):
        info = {
            'search_class': type(estimator).__name__,
            'best_params': getattr(estimator, 'best_params_', None),
            'best_score': getattr(estimator, 'best_score_', None),
            'scoring': getattr(estimator, 'scoring', None),
        }
        return estimator.best_estimator_, info
    return estimator, None


def _cv_scores(estimator, x, y, metrics, cv, fit_params, groups=None):
    """metrics 각각의 K-Fold out-of-fold 점수를 cross_validate 한 번으로 계산한다.

    R2 는 안정성 게이트용으로 항상 포함. 음수 타깃 등으로 scorer 가 실패하면
    해당 지표 fold 는 NaN(error_score)으로 채워진다. groups 가 주어지면 cv(GroupKFold
    등)에 전달되어 같은 그룹이 train/val 폴드에 섞이지 않게 분할한다.

    Returns:
        dict: {지표명: {'folds': ndarray, 'mean': float, 'std': float}}
    """
    wanted = {m for m in metrics if m in _CV_SCORERS} | {'R2'}
    scoring = {m: _CV_SCORERS[m][0] for m in wanted}
    out = cross_validate(estimator, x, y, cv=cv, scoring=scoring, groups=groups,
                         n_jobs=-1, params=fit_params, error_score=np.nan)
    res = {}
    for m in wanted:
        folds = _CV_SCORERS[m][1](out[f'test_{m}'])  # 부호·단위 보정
        res[m] = {'folds': folds, 'mean': float(np.mean(folds)), 'std': float(np.std(folds))}
    return res


def _metric_best(series, better):
    """방향(better)에 맞는 최적값을 반환."""
    if better == 'lower':
        return series.min(skipna=True)
    if better == 'higher':
        return series.max(skipna=True)
    if better == 'closer_to_zero':
        return series.abs().min(skipna=True)
    raise ValueError(f"Unknown 'better': {better!r}")


def _metric_close_mask(series, best, better, tolerance):
    """주 지표 1등 대비 tolerance 이내인 행 마스크."""
    if better == 'lower':
        return series <= best * (1 + tolerance)
    if better == 'higher':
        return series >= best * (1 - tolerance)
    if better == 'closer_to_zero':
        return series.abs() <= best * (1 + tolerance)
    raise ValueError(f"Unknown 'better': {better!r}")


def _metric_is_flaw(value, best, flaw_type, threshold):
    """근소 격차 그룹 내 단일 보조 지표 결정적 결함 여부."""
    if np.isnan(best):
        return False  # 그룹 전체 미측정 → 비교 불가
    if np.isnan(value):
        return True   # 본인만 측정 불가 → 결함
    if flaw_type == 'rel_excess':
        return value > best * (1 + threshold)
    if flaw_type == 'abs_drop':
        return value < best - threshold
    if flaw_type == 'abs_excess':
        return abs(value) > best + threshold
    raise ValueError(f"Unknown 'flaw_type': {flaw_type!r}")


# --------------------------------------------------------
# 여러 회귀 모델의 지표를 한 번에 계산하고 4단계 전략으로 'Rank' 를 매긴 비교 테이블을 반환
# --------------------------------------------------------
def reg_score_table(
    estimator: list | dict,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    primary: str = 'RMSE',
    aux: list[str] | None = ['MAE', 'R2'],
    verbose: bool = True,
) -> DataFrame:
    """여러 회귀 모델의 지표를 한 번에 계산하고 4단계 전략으로 'Rank' 를 매긴 비교 테이블을 반환한다.

    순위 전략: ① 주 지표로 정렬 → ② 1등의 5% 이내를 '근소 격차 그룹'으로 묶음
    → ③ 그룹 내 보조 지표 결정적 결함 카운트 → ④ (결함수, 주 지표)로 재정렬.

    Args:
        estimator: 모델의 리스트 [m1, m2, ...] 또는 딕셔너리 {'name': m, ...}
        x_test: 테스트 설명변수 (DataFrame)
        y_test: 실제 목표변수 (DataFrame 또는 ndarray)
        primary: 순위 주 지표 (기본 'RMSE'). _METRIC_SPECS 의 키.
        aux: 결함 판정용 보조 지표 리스트 (기본 ['MAE', 'R2']). _METRIC_SPECS 의 키.
        verbose: True 면 판정 과정을 출력.

    Returns:
        DataFrame: 맨 앞 'Rank'·'Group', 맨 끝 '{primary}_Gap'(1등 대비 격차, 양수=더 나쁨)
            컬럼이 붙은 Rank 순 테이블.

    Raises:
        TypeError: estimator 가 리스트/딕셔너리가 아닌 경우.
        ValueError: primary/aux 지표명이 _METRIC_SPECS 에 없는 경우.
    """
    # ============ 파라미터 검증 ============
    if isinstance(aux, str):
        aux = [aux]  # 단일 보조 지표 문자열도 허용
    if primary not in _METRIC_SPECS:
        raise ValueError(
            f"Unknown primary metric: {primary!r}. "
            f"Supported: {sorted(_METRIC_SPECS)}"
        )
    for m in aux:
        if m not in _METRIC_SPECS:
            raise ValueError(
                f"Unknown auxiliary metric: {m!r}. "
                f"Supported: {sorted(_METRIC_SPECS)}"
            )

    # ============ 각 모델 점수 계산 ============
    if isinstance(estimator, dict):
        models = estimator.items()
    elif isinstance(estimator, list):
        models = [(f"Model {i+1}", m) for i, m in enumerate(estimator)]
    else:
        raise TypeError("estimator must be a list or dict of models")

    score_tables = []
    for name, model in models:
        score_df = reg_score(model, x_test, y_test)
        score_df.reset_index(inplace=True)
        score_df.index = [name]
        score_tables.append(score_df)

    final_score_table = DataFrame()
    for df in score_tables:
        final_score_table = concat([final_score_table, df])

    final_score_table.index.name = "name"

    # ============ 4단계 순위 산정 전략 ============
    p_spec = _METRIC_SPECS[primary]
    p_better = p_spec['better']

    if verbose:
        print("\n" + "=" * 70)
        print(f"◆ Score Table Ranking : primary={primary!r}, aux={aux}")
        print("=" * 70)

    # step1: 주 지표 기준 정렬 (방향에 맞게)
    if p_better == 'closer_to_zero':
        order_idx = final_score_table[primary].abs().sort_values(kind='mergesort').index
        sorted_table = final_score_table.loc[order_idx]
        primary_ascending = True  # |value| 기준 오름차순 (재정렬에서 사용)
    else:
        primary_ascending = (p_better == 'lower')
        sorted_table = final_score_table.sort_values(
            primary, ascending=primary_ascending, kind='mergesort'
        )

    if verbose:
        direction_label = {
            'lower': '낮을수록 좋음 (ASC)',
            'higher': '높을수록 좋음 (DESC)',
            'closer_to_zero': '0에 가까울수록 좋음 (|x| ASC)',
        }[p_better]
        print(f"\n▲ step1: 주 지표({primary}) 기준 정렬 — {direction_label}")
        for i, (name, val) in enumerate(sorted_table[primary].items(), 1):
            print(f"   {i:>2}. {name:<14} {primary:<6}= {val:>16.3f}")

    # step2: 1등과 5% 이내인 모델을 '근소 격차 그룹'으로 묶음
    best_primary = _metric_best(sorted_table[primary], p_better)
    close_mask = _metric_close_mask(sorted_table[primary], best_primary, p_better, 0.05)
    close_group = sorted_table[close_mask].copy()
    outside_group = sorted_table[~close_mask]

    if verbose:
        if p_better == 'lower':
            band_str = f"{primary} ≤ {best_primary * 1.05:.3f}"
        elif p_better == 'higher':
            band_str = f"{primary} ≥ {best_primary * 0.95:.3f}"
        else:
            band_str = f"|{primary}| ≤ {abs(best_primary) * 1.05:.3f}"
        print(f"\n▲ step2: 근소 격차 그룹 묶기 (1등의 5% 이내)")
        print(f"   - 1등 {primary:<6} : {best_primary:.3f}")
        print(f"   - 허용 범위    : {band_str}")
        print(f"   - 근소 격차 그룹 ({len(close_group)}) : {list(close_group.index)}")
        print(f"   - 그룹 외부     ({len(outside_group)}) : {list(outside_group.index)}")

    # 그룹에 1등만 있으면 step4 직행 (1등 압도적) — 추가 처리 불필요
    if len(close_group) > 1:
        # step3: 근소 격차 그룹 내부에서 보조 지표 결정적 결함 카운트
        aux_bests = {m: _metric_best(close_group[m], _METRIC_SPECS[m]['better']) for m in aux}

        # 모델별 결함을 카운트하면서, 어떤 보조 지표에서 결함이 발생했는지 라벨 보존
        flaw_details = {}
        for idx in close_group.index:
            row = close_group.loc[idx]
            triggered = []
            for m in aux:
                spec = _METRIC_SPECS[m]
                if _metric_is_flaw(row[m], aux_bests[m], spec['flaw_type'], spec['threshold']):
                    triggered.append(m)
            flaw_details[idx] = triggered
        close_group['_flaws'] = [len(flaw_details[idx]) for idx in close_group.index]

        if verbose:
            print(f"\n▲ step3: 보조 지표 결정적 결함 점검 (근소 격차 그룹 내부)")
            print(f"   - 그룹 1등 점수 / 결함 임계치:")
            for m in aux:
                spec = _METRIC_SPECS[m]
                b = aux_bests[m]
                if spec['flaw_type'] == 'rel_excess':
                    rule = f"row > {b * (1 + spec['threshold']):.3f}  (1등 × {1 + spec['threshold']:.2f})"
                elif spec['flaw_type'] == 'abs_drop':
                    rule = f"row < {b - spec['threshold']:.3f}  (1등 − {spec['threshold']:.2f})"
                else:  # abs_excess
                    rule = f"|row| > {b + spec['threshold']:.3f}  (1등 + {spec['threshold']:.2f})"
                print(f"       · {m:<6} best={b:>10.3f}   결함조건: {rule}")
            print(f"   - 모델별 결함:")
            for idx in close_group.index:
                triggered = flaw_details[idx]
                tag = "(결함 없음)" if not triggered else f"→ {triggered}"
                print(f"       · {idx:<14} {len(triggered)}개 {tag}")

        # step4: (결함수, 주 지표) 오름차순 재정렬 — 결함 적은 모델이 위로,
        #        동률이면 주 지표가 좋은 모델 우선.
        if p_better == 'closer_to_zero':
            close_group['_primary_key'] = close_group[primary].abs()
            close_group = close_group.sort_values(
                ['_flaws', '_primary_key'], ascending=[True, True], kind='mergesort'
            ).drop(columns=['_flaws', '_primary_key'])
        else:
            close_group = close_group.sort_values(
                ['_flaws', primary],
                ascending=[True, primary_ascending],
                kind='mergesort',
            ).drop(columns=['_flaws'])
    elif verbose:
        print(f"\n▲ step3: 스킵 — 근소 격차 그룹에 1등만 존재 (1등 압도적, step4 직행)")

    # 근소 격차 그룹(재정렬) + 외부 그룹(주 지표 순) 결합 후 Rank 부여
    final_score_table = concat([close_group, outside_group])
    final_score_table.insert(0, 'Rank', range(1, len(final_score_table) + 1))

    # Rank 다음에 그룹 구분 컬럼 표시 (근소 격차 그룹 / 그룹 외부)
    close_set = set(close_group.index)
    group_labels = [
        'Contender' if name in close_set else 'Outside'
        for name in final_score_table.index
    ]
    final_score_table.insert(1, 'Group', group_labels)

    # ============ 맨 끝 컬럼: 주 지표가 1등 대비 몇 % 나쁜지 (양수=더 나쁨) ============
    gap_col = f'{primary}_Gap'
    if p_better == 'closer_to_zero':
        ref = abs(final_score_table[primary].iloc[0])
        diff = final_score_table[primary].abs() - ref
    else:
        ref = final_score_table[primary].iloc[0]
        # '높을수록 좋음' 이면 1등보다 낮을수록 나쁜 것이므로 부호를 뒤집어 양수화
        sign = 1.0 if p_better == 'lower' else -1.0
        diff = sign * (final_score_table[primary] - ref)

    if ref == 0:
        # 기준값이 0 이면 비율 계산이 불가 → NaN 처리
        final_score_table[gap_col] = np.where(diff == 0, 0.0, np.nan)
    else:
        final_score_table[gap_col] = (diff / abs(ref)).round(3)

    if verbose:
        print(f"\n▲ step4: 최종 Rank")
        for rank, name in zip(final_score_table['Rank'], final_score_table.index):
            tag = "[그룹 내]" if name in close_set else "[그룹 외 · 주 지표 순]"
            print(f"   {rank:>2}. {name:<14} {tag}")
        print("=" * 70 + "\n")

    return final_score_table


# --------------------------------------------------------
# 회귀 과적합 진단 함수 (Train / CV / Test)
# --------------------------------------------------------
def reg_overfit(
    estimator,
    x_train: DataFrame,
    y_train: DataFrame | np.ndarray,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    metrics: list[str] | str | None = ['RMSE', 'MAE', 'R2'],
    threshold: float = 0.15,
    underfit_threshold: float = 0.3,
    cv_score: bool = True,
    cv: int = 5,
    groups=None,
    cv_stability: bool = False,
    cv_std_threshold: float = 0.05,
    fit_params: dict | None = None,
    learning_curve: bool = True,
    width: int = 1280,
    height: int = 640,
    grid: bool = True,
    save_path: str | None = None,
    verbose: bool = True,
) -> DataFrame:
    """Train / CV / Test 성능을 한 표로 보여주고 과적합을 진단한다.

    estimator 가 GridSearchCV·RandomizedSearchCV 등 search 객체면 best_estimator_
    를 자동으로 풀어 쓰고(best_params_ 도 함께 출력), 아니면 그대로 사용한다.

    세 점수의 출처:
        - Train : x_train 재예측 (in-sample)
        - CV    : x_train 을 K-Fold 로 out-of-fold 평가 (effective 모델 재학습)
        - Test  : x_test holdout 예측 (최종 일반화 참고용, 1회만 쓸 것)

    각 지표 행의 'Overfit' 컬럼은 일반화 / 과대적합 / 과소적합 셋 중 하나로 판정한다:
        - 과소적합(Underfit · 高편향): train 성능 자체가 낮음. 격차가 아닌 절대 성능
            문제이므로 스케일 무관한 train R2 로 본다(train R2 < underfit_threshold).
            모델 수준 문제라 해당 시 전 지표 행에 우선 적용된다.
        - 과대적합(Overfit · 高분산): Train↔일반화 격차가 큼. 학술적 정석대로 Train↔CV
            로 보며(단일 holdout 보다 안정적, test 는 반복 진단 시 정보 누수), CV 가
            없으면(cv_score=False) Train↔Test 로 폴백. 각 지표 방향에 맞춰 '일반화
            추정치가 train 보다 나쁜 정도'를 양수 gap 으로 만들고, R2(상한 1) 류는 절대
            격차(점수차)를, 오차 류는 상대 격차(gap/max(|train|,|gen|))를 Gap% 로 삼아
            Gap% ≥ threshold 이면 과대적합으로 본다.
        - 일반화(Good fit): 위 둘 다 아님 (격차 작고 train 성능 양호).
    모델 수준 최종 진단(result.attrs['diagnosis'])도 동일하게 셋 중 하나이며,
    과소적합 > 과대적합 > 일반화 우선순위로 정한다.

    Args:
        estimator: 학습된 회귀 모델/파이프라인 또는 (refit 된) search 객체
        x_train, y_train: 훈련 데이터
        x_test, y_test: holdout(test) 데이터
        metrics: 표시·진단할 지표. 리스트 또는 단일 문자열(예: 'R2') 모두 허용
            (기본 ['RMSE','MAE','R2']). 'closer_to_zero' 방향(MPE)은 해석이 모호하여 불가.
        threshold: 과대적합 경계. Gap% ≥ threshold 면 과대적합 (기본 0.15).
            ※ 확립된 학술 기준이 아닌 보수적 경험칙이며, 도메인에 맞게 조정해야 한다.
            절대 임계치보다 '학습곡선 추세·절대 CV 성능'이 더 신뢰할 만하다.
            기본값 0.15 는 사회·도시 데이터(주택가격·수요·상권 등 중간~높은 노이즈)
            기준이다. 권장값: 정형·저노이즈 0.05~0.10 / 사회·도시 0.15(기본값) /
            시계열·금융·의료 등 매우 노이즈 큰 도메인 0.20~0.30.
        underfit_threshold: train R2 가 이 값 미만이면 과소적합(高편향) 신호 (기본 0.3).
            ※ threshold 와 마찬가지로 미출처 경험칙이다. 도메인 불가피오차(Bayes error)가
            큰 곳에선 정상 모델도 R2 가 낮아 오판할 수 있으니 조정할 것.
            기본값 0.3 은 사회·도시 데이터 기준. 권장값: 정형·저노이즈 0.5 /
            사회·도시 0.3(기본값) / 매우 노이즈 큰 도메인 0.2.
        cv_score: True 면 CV 컬럼 계산 + Train↔CV 로 진단 (기본 True, K-Fold 재학습).
            False 면 CV 컬럼 NaN, 진단은 Train↔Test 로 폴백.
        cv: K-Fold 폴드 수(int) 또는 sklearn CV splitter. CV 컬럼·안정성·학습곡선 공통.
            groups 를 주면서 int 를 넘기면 내부적으로 GroupKFold(n_splits=cv) 로 승격된다.
        groups: 그룹 라벨 배열(길이 n_samples). 주면 같은 그룹이 train/val 폴드에
            섞이지 않게 분할한다(예: 동일 지역·가구·매장의 누수 방지). cross_validate·
            learning_curve 의 CV 에 전달된다. None 이면 일반 KFold.
        cv_stability: True 면 R2 fold std 로 CV 안정성(신뢰도 게이트)도 함께 출력.
        cv_std_threshold: R2 fold std 가 이 값을 넘으면 '불안정' (기본 0.05, 경험칙).
        fit_params: 내부 재학습(cross_validate/learning_curve)에 넘길 fit 파라미터
            dict. CatBoost 의 cat_features 등. 예: {'model__cat_features': [...]}.
        learning_curve: True 면 학습곡선을 my_plot 으로 출력 (기본 True).
            주 지표 metrics[0] 기준으로 그린다(진단 표와 기준 일치).
        width, height, grid, save_path: 학습곡선 그래프 옵션.
        verbose: 진단 과정 출력 여부.

    Returns:
        DataFrame: index=Metric, 컬럼=[Train, CV, Test, Gap(Train↔기준),
            Gap%(Train↔기준), Overfit]. Gap 컬럼명의 '기준'은 'CV' 또는 'Test'.
            Overfit 컬럼은 '일반화'|'과대적합'|'과소적합'.
            result.attrs['gap_basis'] 에 진단 기준('CV'|'Test'),
            result.attrs['diagnosis'] 에 최종 진단('일반화'|'과대적합'|'과소적합')과
            result.attrs['underfit']/['overfit'] bool 저장.
            search 객체면 result.attrs['search'] 에 best_params 등 저장.
            cv_stability=True 면 result.attrs['cv_stability'] 에 안정성 결과 저장.

    Raises:
        ValueError: metrics 가 _METRIC_SPECS 에 없거나 'closer_to_zero' 방향인 경우,
            또는 threshold 가 양수가 아닌 경우.
    """
    # ============ 파라미터 검증 ============
    if isinstance(metrics, str):
        metrics = [metrics]  # 단일 지표 문자열도 허용
    if not metrics:
        raise ValueError("metrics 는 최소 1개 이상의 지표를 포함해야 합니다.")
    for m in metrics:
        if m not in _METRIC_SPECS:
            raise ValueError(
                f"Unknown metric: {m!r}. Supported: {sorted(_METRIC_SPECS)}"
            )
        if _METRIC_SPECS[m]['better'] == 'closer_to_zero':
            raise ValueError(
                f"metric {m!r} 은(는) 'closer_to_zero' 방향이라 train/test gap "
                f"해석이 모호합니다. 과적합 진단에는 단조 지표(R2/RMSE/MAE 등)를 사용하세요."
            )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if groups is not None and len(groups) != len(x_train):
        raise ValueError(
            f"groups 길이가 x_train 과 다릅니다: {len(groups)} vs {len(x_train)}"
        )

    # groups 가 있고 cv 가 정수면 GroupKFold 로 승격 (같은 그룹의 폴드 간 누수 방지)
    cv_splitter = GroupKFold(n_splits=cv) if (groups is not None and isinstance(cv, int)) else cv

    # ============ search 객체면 best_estimator_ 로 언랩 ============
    base_est, search_info = _unwrap_estimator(estimator)

    # ============ Train / Test 점수 (reg_score 재사용) ============
    train_scores = reg_score(base_est, x_train, y_train)
    test_scores = reg_score(base_est, x_test, y_test)  # x_test = holdout
    classname = train_scores.index[0]

    # ============ CV 점수 (out-of-fold) — 진단의 일반화 기준 ============
    y_cv = y_train.values.ravel() if isinstance(y_train, DataFrame) else np.asarray(y_train).ravel()
    cv_res = (_cv_scores(base_est, x_train, y_cv, metrics, cv_splitter, fit_params, groups=groups)
              if (cv_score or cv_stability) else None)

    # 진단 기준: CV 있으면 Train↔CV(학술 정석), 없으면 Train↔Test 폴백
    gap_basis = 'CV' if (cv_score and cv_res is not None) else 'Test'

    # ============ 과소적합(高편향) 선판정 — 모델 수준 ============
    #   과소적합은 격차가 아닌 '절대 성능' 문제라 스케일 무관한 train R2 로 본다
    #   (RMSE 등은 절대 임계치를 못 잡음). metrics 에 R2 가 없어도 train_scores 에는
    #   항상 들어 있어 사용 가능. 과소적합이면 격차 분석은 무의미하므로 전 지표에 우선 적용.
    train_r2 = train_scores['R2'].iloc[0]
    gen_r2 = (cv_res['R2']['mean'] if (gap_basis == 'CV' and cv_res is not None)
              else test_scores['R2'].iloc[0])
    underfit = (not np.isnan(train_r2)) and (train_r2 < underfit_threshold)

    # gap 컬럼명: 어떤 점수 간 격차인지 드러나게 (Train↔CV 또는 Train↔Test)
    gap_col = f'Gap(Train↔{gap_basis})'
    gap_pct_col = f'Gap%(Train↔{gap_basis})'

    # ============ 지표별 gap + Overfit 라벨(일반화/과대적합/과소적합) ============
    rows = {}
    for m in metrics:
        tr = train_scores[m].iloc[0]
        te = test_scores[m].iloc[0]
        cvv = cv_res[m]['mean'] if (cv_res is not None and m in cv_res) else np.nan
        better = _METRIC_SPECS[m]['better']

        # 일반화 추정치: CV 기준이고 값이 유효하면 CV, 아니면 Test 로 폴백
        gen = cvv if (gap_basis == 'CV' and not np.isnan(cvv)) else te

        # 방향에 맞춰 '일반화 추정치가 train 보다 나쁜 정도'를 양수 gap 으로
        gap = (tr - gen) if better == 'higher' else (gen - tr)

        # Gap% : 격차를 스케일 보정.
        #   - R2 류(higher·상한 1): 이미 정규화된 지표 → 절대 격차(점수차)를 그대로 쓴다.
        #     상대화하면 R2≈0 부근에서 분모가 0 에 수렴해 격차가 폭발하는 허위 과적합 발생.
        #   - 오차 류(lower·무한대): 상대 격차 = gap / max(|train|,|gen|). 분모 max 는
        #     train 오차가 0 에 수렴할 때(완벽 암기) 0-나눗셈을 막는다.
        denom = 1.0 if better == 'higher' else max(abs(tr), abs(gen))
        gap_pct = np.nan if (np.isnan(tr) or np.isnan(gen) or denom == 0) else gap / denom

        # Overfit 라벨: 과소적합(모델 수준) 우선 → 격차 임계 초과면 과대적합 → 그 외 일반화
        if underfit:
            label = '과소적합'
        elif np.isnan(gap_pct):
            label = 'N/A'
        elif gap_pct >= threshold:
            label = '과대적합'
        else:
            label = '일반화'

        rows[m] = {
            'Train': tr,
            'CV': cvv,
            'Test': te,
            gap_col: gap,
            gap_pct_col: round(gap_pct * 100, 2) if not np.isnan(gap_pct) else np.nan,
            'Overfit': label,
        }

    result = DataFrame.from_dict(rows, orient='index',
                                 columns=['Train', 'CV', 'Test', gap_col, gap_pct_col, 'Overfit'])
    result.index.name = 'Metric'
    result.attrs['gap_basis'] = gap_basis
    if search_info is not None:
        result.attrs['search'] = search_info

    # ============ 모델 수준 최종 진단 (일반화/과대적합/과소적합) ============
    overfit_any = any(rows[m]['Overfit'] == '과대적합' for m in metrics)
    if underfit:
        diagnosis = '과소적합'          # 高편향 — train 조차 못 맞춤 (격차보다 우선)
    elif overfit_any:
        diagnosis = '과대적합'          # 高분산 — train↔일반화 격차 큼
    else:
        diagnosis = '일반화'            # 격차 작고 train 성능 양호
    result.attrs['diagnosis'] = diagnosis
    result.attrs['underfit'] = bool(underfit)
    result.attrs['overfit'] = bool(overfit_any)

    # ============ 학습곡선 시각화 (my_plot 재사용) ============
    # PPT 흐름(학습곡선 → 수치 진단)에 맞춰 그래프를 verbose 텍스트보다 먼저 출력
    if learning_curve:
        # 주 지표(metrics[0]) 기준으로 — 진단 표와 기준 일치. MPE 는 검증에서 막혀
        # 남은 지표는 모두 _CV_SCORERS 에 있어 항상 매핑 가능.
        lc_metric = metrics[0]
        lc_scoring, lc_transform = _CV_SCORERS[lc_metric]

        # 훈련 크기별 train/CV 점수 — base_est 는 내부에서 clone 후 재학습됨.
        # neg_* scorer 부호·단위는 lc_transform 으로 보정해 실제 지표값으로 표시.
        sizes, train_sc, cv_sc = sk_learning_curve(
            base_est, x_train, y_cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv_splitter, scoring=lc_scoring, n_jobs=-1,
            groups=groups, params=fit_params,
        )
        train_sc, cv_sc = lc_transform(train_sc), lc_transform(cv_sc)
        train_mean, train_std = train_sc.mean(axis=1), train_sc.std(axis=1)
        cv_mean, cv_std = cv_sc.mean(axis=1), cv_sc.std(axis=1)

        # init/show 로 그림 틀 구성
        fig, ax = my_plot.init(
            width=width, height=height, grid=grid,
            title=f'Learning Curve: {classname}',
            xlabel='Training samples', ylabel=f'{lc_metric} score',
        )
        # train 곡선 + 분산 밴드
        ax.plot(sizes, train_mean, marker='o', color='tab:blue', label='Train')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color='tab:blue')
        # CV 곡선 + 분산 밴드 (두 곡선 간격이 클수록 과적합)
        ax.plot(sizes, cv_mean, marker='s', color='tab:orange', label='Validation (CV)')
        ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std,
                        alpha=0.15, color='tab:orange')
        ax.legend(fontsize=13)
        my_plot.show(save_path=save_path)

    # ============ 진단 출력 ============
    if verbose:
        label_tag = {'일반화': '', '과대적합': ' ⚠', '과소적합': ' ⚑', 'N/A': ''}
        gen_name = gap_basis  # 'CV' 또는 'Test'
        basis_label = f'Train↔{gen_name}' + (' (정석)' if gap_basis == 'CV' else ' (폴백)')
        print("\n" + "=" * 78)
        print(f"◆ Fit Diagnosis: {classname}  "
              f"(threshold={threshold:.0%}, Gap 기준={basis_label})")
        print("  ※ threshold·underfit 기준은 학술 표준이 아닌 보수적 경험칙입니다. "
              "도메인에 맞게 조정하고, 절대값보다 학습곡선 추세를 함께 보세요.")
        if groups is not None:
            print(f"   ▷ CV 분할: {type(cv_splitter).__name__} (groups 기반 — 그룹 간 누수 방지)")
        if search_info is not None:
            print(f"   ▷ {search_info['search_class']} → best_estimator_ 사용  "
                  f"(best_score={search_info['best_score']})")
            print(f"     best_params: {search_info['best_params']}")
        print("=" * 78)
        for m in metrics:
            r = rows[m]
            cv_str = 'N/A' if np.isnan(r['CV']) else f"{r['CV']:>12.4f}"
            pct = r[gap_pct_col]
            pct_str = 'N/A' if np.isnan(pct) else f"{pct:+.2f}%"
            lab = r['Overfit']
            print(f"   - {m:<6}  "
                  f"Train={r['Train']:>12.4f}  {gen_name}={cv_str}  Test={r['Test']:>12.4f}  "
                  f"Gap(Train↔{gen_name})%={pct_str:>9}  [{lab}]{label_tag[lab]}")
        diag_msg = {
            '일반화':   '✔ 일반화 (Good fit) — 격차 작고 train 성능 양호',
            '과대적합': '⚠ 과대적합 (Overfit · 高분산) — train↔일반화 격차 큼',
            '과소적합': '⚑ 과소적합 (Underfit · 高편향) — train 성능 자체가 낮음',
        }[diagnosis]
        r2_str = 'N/A' if np.isnan(train_r2) else f"{train_r2:.4f}"
        gen_str = 'N/A' if np.isnan(gen_r2) else f"{gen_r2:.4f}"
        print(f"\n   ▶ 진단: {diag_msg}")
        print(f"     · train R2={r2_str} (과소적합 기준 < {underfit_threshold}) · "
              f"{gen_name} R2={gen_str}")
        print("=" * 78 + "\n")

    # ============ CV 안정성 (opt-in · R2 fold std 재사용) ============
    if cv_stability and cv_res is not None and 'R2' in cv_res:
        fold_scores = cv_res['R2']['folds']
        cv_mean = cv_res['R2']['mean']
        cv_std = cv_res['R2']['std']  # R2 는 음수 가능 → 변동계수 대신 절대 std
        stable = bool(cv_std <= cv_std_threshold)

        result.attrs['cv_stability'] = {
            'scoring': 'r2', 'cv': cv,
            'fold_scores': fold_scores,
            'mean': cv_mean, 'std': cv_std,
            'stable': stable, 'std_threshold': cv_std_threshold,
        }

        if verbose:
            # cv 가 정수면 'K-Fold', splitter 객체면 클래스명으로 표기
            cv_desc = f"{cv}-Fold" if isinstance(cv, int) else type(cv_splitter).__name__
            print("=" * 78)
            print(f"◇ CV Stability (신뢰도 게이트): {classname}  [{cv_desc} · r2]")
            print("=" * 78)
            print(f"   - fold scores : {np.round(fold_scores, 4).tolist()}")
            print(f"   - mean ± std  : {cv_mean:.4f} ± {cv_std:.4f}")
            state = '안정' if stable else '불안정'
            note = ('Fold 간 일관성 ↑ → 판정 신뢰 가능'
                    if stable else '분할에 민감 → 판정 신뢰도 ↓ · 데이터/CV 점검')
            print(f"   ▶ 판정: {state} (std {cv_std:.4f} "
                  f"{'≤' if stable else '>'} {cv_std_threshold}) — {note}")
            print("   ※ 과적합의 직접 증거가 아니라 판정의 신뢰도 게이트")
            print("=" * 78 + "\n")

    return result


# --------------------------------------------------------
# 변수 중요도(Feature Importance) 기반 변수 선택 함수
# --------------------------------------------------------
def _fi_resolve_tree_importance(model, model_class, importance_type):
    """트리/부스팅 모델에서 (중요도 배열, 사용된 기준 라벨) 을 반환한다.

    importance_type='auto'(권장·기본) 이면 라이브러리별 권장 기준을 쓴다:
        - XGBoost  : gain (분할 이득 — split 횟수보다 신뢰도 ↑). 모델이 'weight'
          등으로 생성됐어도 feature_importances_ 의 기준을 잠시 gain 으로 바꿔 읽는다.
        - LightGBM : gain. 기본값 'split'(사용 횟수)은 고카디널리티 변수를 과대
          평가하므로, 학습된 booster_ 에서 gain 을 직접 추출한다.
        - sklearn 트리(DecisionTree/RandomForest): MDI(불순도 감소) — 유일한 옵션.
        - CatBoost : PredictionValuesChange (기본) — 별도 권장 대체 없음.
    importance_type='native' 이면 모델 생성 시 설정된 feature_importances_ 를 그대로
    쓴다. 권장 기준 추출이 어떤 이유로 실패하면 native 로 폴백한다.
    """
    use_native = (importance_type == 'native')

    # XGBoost — 학습된 booster 에서 gain 직접 추출 (모델 속성 비변형, LightGBM 과 일관).
    #   get_score 는 split 에 쓰인 변수만 dict 로 주므로 booster.feature_names 순서
    #   (= preprocessor.get_feature_names_out 순서)대로 0 채움 정렬한다. 회귀·분류 공통.
    if model_class in ('XGBRegressor', 'XGBClassifier') and not use_native:
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type='gain')
            names = booster.feature_names
            if names is None:  # 이름 없이 학습된 경우 f0, f1 … 로 폴백
                names = [f'f{i}' for i in range(model.n_features_in_)]
            imp = np.array([score.get(f, 0.0) for f in names], dtype=float)
            return imp, 'gain (xgboost·권장)'
        except Exception:
            pass

    # LightGBM — 학습된 booster 에서 gain 직접 추출 (생성 시 split 이어도 무관). 회귀·분류 공통.
    if model_class in ('LGBMRegressor', 'LGBMClassifier') and not use_native:
        try:
            imp = np.asarray(
                model.booster_.feature_importance(importance_type='gain'),
                dtype=float).ravel()
            return imp, 'gain (lightgbm·권장)'
        except Exception:
            pass

    # 그 외 / native / 폴백 — 모델 기본 feature_importances_
    imp = np.asarray(model.feature_importances_, dtype=float).ravel()
    label = {
        'DecisionTreeRegressor':  'MDI(불순도 감소)',
        'RandomForestRegressor':  'MDI(불순도 감소)',
        'DecisionTreeClassifier': 'MDI(불순도 감소)',
        'RandomForestClassifier': 'MDI(불순도 감소)',
        'CatBoostRegressor':      'PredictionValuesChange',
        'CatBoostClassifier':     'PredictionValuesChange',
        'XGBRegressor':           f"native '{getattr(model, 'importance_type', None) or 'gain'}'",
        'XGBClassifier':          f"native '{getattr(model, 'importance_type', None) or 'gain'}'",
        'LGBMRegressor':          f"native '{getattr(model, 'importance_type', 'split')}'",
        'LGBMClassifier':         f"native '{getattr(model, 'importance_type', 'split')}'",
    }.get(model_class, 'feature_importances_')
    return imp, label


def _fi_ohe_dummy_to_origin(ohe):
    """학습된 OneHotEncoder 의 더미 출력명 → 원본 컬럼명 매핑 dict 를 만든다.

    get_feature_names_out 의 출력은 입력 컬럼별로 묶여 있고(categories_ 순서), 각
    입력 컬럼의 더미 개수는 카테고리 수에서 drop 된 만큼을 뺀 값이다. 이 개수로
    출력명을 청크 단위로 끊어 원본 컬럼에 귀속시킨다.

    drop_idx_ 는 drop=None 이면 None, drop='first'/'if_binary' 면 object 배열로
    컬럼별 (드롭 인덱스|None) 을 담는다. object 배열 원소는 파이썬 객체라
    `drop_idx_[i] is not None` 비교가 정확하다('if_binary' 의 부분 드롭도 OK).
    build_pipeline 은 'first'/None 만 쓰고 infrequent 옵션은 쓰지 않으므로 위 청크
    가정이 항상 성립하지만, 혹시라도 가정이 깨져 출력명을 다 소비하지 못하면
    (infrequent 등) 잘못 귀속하느니 빈 매핑을 반환해 집계를 건너뛰게 한다.
    """
    in_cols = list(ohe.feature_names_in_)
    out_names = list(ohe.get_feature_names_out(in_cols))
    drop_idx = getattr(ohe, 'drop_idx_', None)  # None 또는 컬럼별 (드롭 인덱스|None)

    mapping = {}
    pos = 0
    for i, col in enumerate(in_cols):
        n_out = len(ohe.categories_[i])
        if drop_idx is not None and drop_idx[i] is not None:
            n_out -= 1  # 해당 컬럼에서 카테고리 하나 제거됨
        for _ in range(n_out):
            if pos >= len(out_names):
                return {}  # 청크 가정 붕괴 — 안전하게 집계 비활성
            mapping[out_names[pos]] = col
            pos += 1
    if pos != len(out_names):
        return {}  # 출력명을 다 소비 못함(예상 밖 구조) — 안전하게 집계 비활성
    return mapping


def _fi_build_name_map(base_est):
    """변환 후 피처명 → 원본 입력 컬럼명 매핑 dict 를 만든다 (원핫 더미 합산용).

    원핫 더미만 원본 컬럼으로 되돌리고, 그 외(수치형·passthrough)는 이름이 그대로
    유지되므로 매핑에 넣지 않는다(호출부에서 없으면 자기 자신으로 처리). PCA 출력
    (pca0, pca1 …)은 원본으로 복원 불가하므로 매핑되지 않고 그대로 남는다.
    """
    name_map = {}
    if not isinstance(base_est, Pipeline):
        return name_map
    pre = base_est.named_steps.get('preprocessor')
    if pre is None or not hasattr(pre, 'transformers_'):
        return name_map
    for _tname, trans, _cols in pre.transformers_:
        ohe = trans.named_steps.get('onehot') if isinstance(trans, Pipeline) else (
            trans if isinstance(trans, OneHotEncoder) else None)
        if ohe is not None and hasattr(ohe, 'categories_'):
            name_map.update(_fi_ohe_dummy_to_origin(ohe))
    return name_map


def _fi_has_scaler_in_pipeline(base_est):
    """전처리에 스케일러가 있는지 best-effort 로 탐지한다.

    Returns:
        True  : 스케일러 발견 (|coef_| 비교 전제 충족)
        False : 파이프라인이지만 스케일러 없음 (선형 모델이면 경고 대상)
        None  : 파이프라인이 아니라 판단 불가 (단독 모델 등)
    """
    if not isinstance(base_est, Pipeline):
        return None
    pre = base_est.named_steps.get('preprocessor')
    if pre is None or not hasattr(pre, 'transformers_'):
        return None
    for _name, trans, _cols in pre.transformers_:
        steps = list(trans.named_steps.values()) if isinstance(trans, Pipeline) else [trans]
        if any(type(s).__name__ in _FI_SCALER_CLASSES for s in steps):
            return True
    return False


def _fi_extract_model_and_features(estimator):
    """파이프라인/서치 객체를 풀어 (최종 모델, 변수명, 원본명 매핑, base_est, search_info).

    build_pipeline 로 만든 Pipeline([('preprocessor', ...), ('model', ...)]) 구조를
    가정한다. GridSearchCV 등 search 객체면 _unwrap_estimator 로 best_estimator_ 를
    먼저 푼다. 변수명은 모델 직전까지의 변환 결과(preprocessor.get_feature_names_out)
    기준 — 즉 모델이 '실제로 학습한' 피처 이름이다(원핫·PCA 변환 후 이름 포함).

    Returns:
        (model, feature_names, name_map, base_est, search_info):
            model         : 중요도를 가진 최종 추정기
            feature_names : np.ndarray[str] — 모델이 학습한 피처명
            name_map      : dict[변환후명 → 원본컬럼명] (원핫 더미 합산용)
            base_est      : search 언랩 후 추정기 (스케일러 탐지 등에 사용)
            search_info   : search 객체였으면 dict, 아니면 None
    """
    base_est, search_info = _unwrap_estimator(estimator)

    if isinstance(base_est, Pipeline):
        model = base_est.named_steps.get('model', base_est[-1])
        # 모델 직전 단계까지의 출력 피처명 = 모델이 학습한 피처
        try:
            feature_names = np.asarray(base_est[:-1].get_feature_names_out())
        except Exception:
            feature_names = None
    else:
        # 파이프라인이 아닌 단독 모델도 방어적으로 허용
        model = base_est
        feature_names = (np.asarray(model.feature_names_in_)
                         if hasattr(model, 'feature_names_in_') else None)

    name_map = _fi_build_name_map(base_est)

    return model, feature_names, name_map, base_est, search_info


def feature_importance(
    estimator,
    cum_ratio: float = 0.9,
    top_n: int | None = None,
    aggregate_dummies: bool = True,
    importance_type: str = 'auto',
    plot: bool = True,
    max_display: int | None = 30,
    width: int = 1280,
    height: int = 640,
    grid: bool = True,
    save_path: str | None = None,
    verbose: bool = True,
) -> DataFrame:
    """학습된 회귀/분류 모델에서 변수 중요도를 도출해 상위 변수만 추려 반환한다.

    회귀·분류 공용. 중요도 산출 메커니즘(트리=feature_importances_, 선형=|coef_|)이
    두 과제에서 동일하므로 한 함수로 처리한다. 과적합 완화 목적의 변수 채택용.
    모델 유형에 따라 산출 방식이 다르며, importance_type='auto'(기본) 이면
    라이브러리별 '권장 기준'을 자동 적용한다:
        - 트리·부스팅(DecisionTree/RandomForest/XGBoost/LightGBM/CatBoost,
          회귀·분류 모두):
          모델이 제공하는 feature_importances_ 를 사용하되 —
            · XGBoost  → gain (모델이 'weight' 로 생성됐어도 gain 으로 읽음)
            · LightGBM → gain (기본값 'split' 은 고카디널리티 변수 과대평가 →
              학습된 booster_ 에서 gain 직접 추출)
            · sklearn 트리 → MDI(불순도 감소, 유일 옵션)
            · CatBoost → PredictionValuesChange(기본)
          ※ 중요도 '기준'은 라이브러리마다 다르므로(gain/MDI 등) 모델 간 절대 비교가
            아니라 동일 모델 내 상대 순위로만 해석할 것.
        - 선형(회귀: LinearRegression/Ridge/Lasso/ElasticNet/SGD,
          분류: LogisticRegression/RidgeClassifier/SGDClassifier/LinearSVC): |coef_|.
          ※ 계수 크기 비교가 공정하려면 변수가 스케일링(StandardScaler)되어
            있어야 한다. build_pipeline(scale=True) 권장.
          ※ 다중클래스 분류는 coef_ 가 (클래스, 변수) 2D → 클래스축 |coef_| 평균.
        - 거리 기반(KNN)·비선형 커널 SVR/SVC 등 중요도를 정의할 수 없는 모델: 예외 발생.

    estimator 가 Pipeline 이면 'model' 스텝의 최종 모델을 쓰고, 변수명은 모델 직전
    까지의 변환 결과 기준이다. GridSearchCV·RandomizedSearchCV 등 search 객체면
    best_estimator_ 를 자동으로 풀어 쓴다.

    원핫 인코딩된 명목형 변수는 기본적으로(aggregate_dummies=True) 더미들의 중요도를
    합산해 '원본 컬럼' 단위로 되돌린다 — 재학습 시 원본 컬럼 단위로 변수를 고를 수
    있게 하기 위함. 따라서 cat_y/cat_z/cat_x 가 아니라 cat 하나로 집계된다.
    (PCA 출력 pca0/pca1 … 은 원본 복원이 불가능해 그대로 남는다.)

    채택 규칙: (집계 후) 중요도를 합 1 로 정규화 → 내림차순 정렬 → 누적 비율이
    cum_ratio 에 처음 도달하는 지점까지의 변수를 채택(경계 변수 포함). top_n 을 주면
    cum_ratio 대신 상위 top_n 개를 채택한다.

    Args:
        estimator: 학습된 회귀/분류 모델·파이프라인 또는 (refit 된) search 객체.
        cum_ratio: 채택할 누적 중요도 비율 (0, 1]. 예: 0.9 면 전체 중요도의 90%를
            설명하는 상위 변수까지 채택 (기본 0.9). top_n 이 주어지면 무시된다.
        top_n: 양의 정수를 주면 누적 비율 대신 상위 top_n 개 변수를 채택.
        aggregate_dummies: True(기본) 면 원핫 더미 중요도를 원본 명목형 컬럼으로
            합산해 원본 컬럼 단위로 집계·채택한다. False 면 더미 단위 그대로 본다.
        importance_type: 'auto'(기본) 면 라이브러리별 권장 기준(부스팅=gain 등) 자동
            적용. 'native' 면 모델 생성 시 설정된 feature_importances_ 를 그대로 사용.
            선형 모델에는 영향 없음(항상 |coef_|).
        max_display: 그래프에 표시할 최대 변수 개수 (기본 30). 변수가 이보다 많으면
            상위 그 수만 그리고 나머지는 생략한다(채택 로직은 전체 기준 그대로 —
            그래프 표시만 제한). None/0 이면 전부 표시.
        plot: True 면 변수 중요도 막대그래프 출력 (채택/탈락 색 구분).
        width, height, grid, save_path: 그래프 옵션.
        verbose: 채택 결과·사용된 중요도 기준을 출력할지 여부 (기본 True).
            선형 모델인데 스케일러가 없으면(|coef_| 비교 전제 위반) 경고도 출력한다.

    Returns:
        DataFrame: 전체 변수 표(중요도 내림차순). index=변수명,
            컬럼=[Importance, Ratio, CumRatio, 채택여부] — 마지막 '채택여부' 는
            '채택'/'탈락' 문자열. 채택된 변수명만 필요하면 attrs['selected_features'] 사용.
            result.attrs['all'] 에 동일한 전체 변수 표,
            result.attrs['selected_features'] 에 채택된 변수명 리스트,
            result.attrs['importance_metric'] 에 실제 사용된 기준 라벨
            (예: 'gain (lightgbm·권장)', 'MDI(불순도 감소)', '|coef_|'),
            result.attrs['model_class'] 에 모델 클래스명,
            result.attrs['scaled'] 에 선형 모델의 스케일러 유무(True/False/None=판단불가),
            search 객체였으면 result.attrs['search'] 에 best_params 등 저장.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, top_n·max_display 가 양의 정수가
            아니거나, importance_type 이 {'auto','native'} 밖이거나,
            변수명/중요도 길이가 맞지 않거나, 중요도 총합이 0 인 경우.
        TypeError: 변수 중요도를 도출할 수 없는 모델 유형인 경우
            (KNN·비선형 커널 SVR/SVC 등 feature_importances_·coef_ 둘 다 없음).
    """
    # ─────────────────────────────────────────────────────────────
    # 처리 절차 (PPT '코드 동작 흐름' 5단계와 1:1 대응)
    #   STEP 01 입력 수신   → 파라미터 검증
    #   STEP 02 언랩        → search/Pipeline 분해 (모델·변수명·더미매핑 추출)
    #   STEP 03 중요도 산출 → ★ 모델 유형 분기 (A 트리·부스팅 / B 선형 / C 산출불가)
    #   STEP 04 집계·정렬   → 더미 합산 → 합1 정규화 → 내림차순 정렬 → 누적 비율
    #   STEP 05 채택·반환   → top_n 또는 cum_ratio 절단 → 표·메타·그래프 반환
    # ─────────────────────────────────────────────────────────────

    # ============ STEP 01 · 입력 수신 (파라미터 검증) ============
    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio must be in range (0.0, 1.0], got {cum_ratio}")
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError(f"top_n must be a positive integer or None, got {top_n!r}")
    if max_display is not None and (not isinstance(max_display, int) or max_display < 0):
        raise ValueError(f"max_display must be a non-negative integer or None, got {max_display!r}")
    if importance_type not in ('auto', 'native'):
        raise ValueError(
            f"importance_type must be 'auto' or 'native', got {importance_type!r}")

    # ============ STEP 02 · 언랩 (search/Pipeline 분해 → 모델·변수명·더미매핑 추출) ============
    model, feature_names, name_map, base_est, search_info = _fi_extract_model_and_features(estimator)
    model_class = type(model).__name__
    scaled = None  # 선형 모델일 때만 의미: 스케일러 유무 (True/False/None)

    # ============ STEP 03 · 중요도 산출 ★ 모델 유형 분기 (auto = 라이브러리별 권장 기준) ============
    # CASE A — 트리·부스팅: feature_importances_ 사용
    #   XGBoost/LightGBM → gain(booster 직접 호출) · sklearn 트리 → MDI · CatBoost → PredictionValuesChange
    if hasattr(model, 'feature_importances_'):
        importances, importance_metric = _fi_resolve_tree_importance(
            model, model_class, importance_type)
    # CASE B — 선형: |coef_| (2D 이면 클래스/출력축 평균) + 스케일러 탐지로 신뢰도 점검
    elif hasattr(model, 'coef_'):
        # 다중 출력(2D)이면 출력축 평균, 단일 타깃이면 그대로. 부호 무관 → 절대값.
        coef = np.abs(np.asarray(model.coef_, dtype=float))
        importances = coef.mean(axis=0) if coef.ndim == 2 and coef.shape[0] > 1 else coef.ravel()
        importance_metric = '|coef_|'
        # |coef_| 는 동일 스케일 전제 — 선형 모델인데 스케일러 없으면 결과 신뢰 불가
        if model_class in _IMPORTANCE_LINEAR:
            scaled = _fi_has_scaler_in_pipeline(base_est)
    # CASE C — 산출 불가: 두 속성 모두 없음 → TypeError 로 흐름 중단 (대안: Permutation/SHAP)
    else:
        if model_class in ('KNeighborsRegressor', 'KNeighborsClassifier'):
            hint = '거리 기반 모델이라 변수별 기여도를 분리할 수 없습니다'
        elif model_class in ('SVR', 'SVC'):
            hint = '비선형 커널은 coef_ 가 없습니다 (linear 커널만 도출 가능)'
        else:
            hint = 'feature_importances_·coef_ 속성이 모두 없습니다'
        raise TypeError(
            f"'{model_class}' 모델은 변수 중요도를 도출할 수 없습니다 — {hint}. "
            f"중요도 산출 가능: 트리/부스팅{sorted(_IMPORTANCE_TREE)} 또는 "
            f"선형{sorted(_IMPORTANCE_LINEAR)} 계열."
        )

    # ── STEP 03 후처리: 변수명 정합성 — 변환 변수명을 못 얻었으면 임시 이름(x0,x1…) 생성 ──
    names_synthesized = feature_names is None
    if names_synthesized:
        feature_names = np.array([f'x{i}' for i in range(len(importances))])
    if len(feature_names) != len(importances):
        raise ValueError(
            f"변수명 수({len(feature_names)})와 중요도 수({len(importances)})가 "
            f"일치하지 않습니다. 파이프라인 구조를 확인하세요."
        )

    # ============ STEP 04 · 집계·정렬 ============
    # ① 원핫 더미 → 원본 컬럼 단위 중요도 합산 (aggregate_dummies=True 기본 동작)
    n_raw = len(feature_names)
    if aggregate_dummies and name_map:
        # 변환후명을 원본명으로 치환(매핑 없으면 자기 자신) 후 원본 단위로 합산.
        # 첫 등장 순서를 보존해 결정적으로 집계한다.
        agg = {}
        for fname, imp in zip(feature_names, importances):
            origin = name_map.get(fname, fname)
            agg[origin] = agg.get(origin, 0.0) + imp
        feature_names = np.array(list(agg.keys()))
        importances = np.array(list(agg.values()), dtype=float)
    n_aggregated = len(feature_names)  # 집계 후 변수 수 (더미 합산 시 n_raw 보다 작음)

    # ② 합계 검증 — 전 계수 0(Lasso 등)·미학습이면 비율 계산 불가
    total = importances.sum()
    if total <= 0:
        raise ValueError(
            f"중요도 총합이 0 입니다 ({model_class}). 모델이 학습되지 않았거나 "
            f"(Lasso/ElasticNet 등) 모든 계수가 0 으로 규제되었을 수 있습니다."
        )

    # ③ 합 1 정규화 → ④ 내림차순 정렬 → ⑤ 누적 비율 계산
    ratio = importances / total
    order = np.argsort(ratio)[::-1]              # 내림차순
    sorted_names = feature_names[order]
    sorted_ratio = ratio[order]
    sorted_imp = importances[order]
    cum = np.cumsum(sorted_ratio)
    n = len(sorted_ratio)

    # ============ STEP 05 · 채택·반환 ============
    # 절단 개수 결정 — 우선순위: top_n 이 지정되면 top_n 우선, 미지정 시 cum_ratio
    if top_n is not None:
        k = min(top_n, n)
        select_mode = f'top_n={top_n}'
    else:
        # 누적 비율이 cum_ratio 에 처음 도달하는 지점까지(경계 변수 포함)
        k = int(np.searchsorted(cum, cum_ratio)) + 1
        k = max(1, min(k, n))
        select_mode = f'cum_ratio={cum_ratio:.0%}'

    selected_mask = np.arange(n) < k

    # 반환 ① 결과 표(DataFrame) — 전체 변수 + 마지막 '채택여부' 컬럼(채택/탈락) + 메타데이터(attrs)
    full = DataFrame({
        'Importance': sorted_imp,
        'Ratio': sorted_ratio,
        'CumRatio': cum,
        '채택여부': np.where(selected_mask, '채택', '탈락'),
    }, index=sorted_names)
    full.index.name = 'Feature'

    # 전체 변수를 채택/탈락 표시와 함께 반환 (채택 변수만 필요하면 attrs['selected_features']).
    # attrs['all'] 은 하위호환용 별도 객체로 둔다(result 자기참조 시 attrs deepcopy 무한재귀 방지).
    result = full.copy()
    result.attrs['all'] = full
    result.attrs['selected_features'] = sorted_names[selected_mask].tolist()
    result.attrs['importance_metric'] = importance_metric
    result.attrs['model_class'] = model_class
    result.attrs['aggregated_dummies'] = bool(aggregate_dummies and n_aggregated < n_raw)
    result.attrs['scaled'] = scaled
    if search_info is not None:
        result.attrs['search'] = search_info

    # 그래프 표시 개수 제한 (채택 로직과 무관 — 그래프 가독성용)
    n_show = n if not max_display else min(n, max_display)
    truncated = n_show < n

    # 반환 ② 시각화 — 변수 중요도 막대그래프(채택=파랑/탈락=회색), plot=True 일 때만
    if plot:
        title = f'Feature Importance: {model_class} ({importance_metric})'
        if truncated:
            title += f'  — 상위 {n_show}/{n}'
        fig, ax = my_plot.init(
            width=width, height=height, grid=grid,
            title=title,
            xlabel='Importance ratio', ylabel='Feature',
        )
        d_names = sorted_names[:n_show]
        d_ratio = sorted_ratio[:n_show]
        colors = ['tab:blue' if s else 'lightgray' for s in selected_mask[:n_show]]
        # barh 는 아래→위로 그려지므로 역순으로 넣어 중요도 큰 변수가 위로 오게
        ax.barh(d_names[::-1], d_ratio[::-1], color=colors[::-1])
        ax.legend(handles=[
            Patch(color='tab:blue', label='채택'),
            Patch(color='lightgray', label='탈락'),
        ], fontsize=12)
        my_plot.show(save_path=save_path)

    # 반환 ③ 콘솔 요약 출력 — 채택 결과·신뢰도 경고, verbose=True 일 때만
    if verbose:
        kept_ratio = float(cum[k - 1])
        print("\n" + "=" * 78)
        print(f"◆ Feature Importance Selection: {model_class}  "
              f"(중요도 기준={importance_metric}, 채택 기준={select_mode})")
        print("=" * 78)
        # 신뢰도 경고 — 선형 모델 미스케일(|coef_| 전제 위반) / 변수명 합성
        if scaled is False:
            print(f"   ⚠ {model_class} 는 |coef_| 기준이라 변수가 동일 스케일이어야 "
                  f"공정합니다.\n     파이프라인에 스케일러가 없습니다 → "
                  f"build_pipeline(scale=True) 로 재학습 권장 (현재 결과는 신뢰 불가).")
        if names_synthesized:
            print(f"   ⚠ 변환 후 변수명을 얻지 못해 임시 이름(x0, x1 …)을 사용합니다. "
                  f"원본 변수 식별이 어렵다면 파이프라인 구조를 확인하세요.")
        if search_info is not None:
            print(f"   ▷ {search_info['search_class']} → best_estimator_ 사용")
        if result.attrs['aggregated_dummies']:
            print(f"   ▷ 원핫 더미 {n_raw}개 → 원본 컬럼 {n_aggregated}개로 중요도 합산")
        print(f"   전체 변수 {n}개 중 {k}개 채택 "
              f"(누적 중요도 {kept_ratio:.2%} 설명)")
        print("=" * 78)
        print(f"   ▶ 채택된 변수: {sorted_names[selected_mask].tolist()}")
        print("=" * 78 + "\n")

    return result




# ==========================================================================
# SHAP 분석 (회귀·분류 공용) — feature_importance 와 같은 "공유 도구" 레이어
# ==========================================================================
# 설계 메모:
#   feature_importance 가 모델 내부 지표(트리 gain/MDI, 선형 |coef_|)로 '전역' 중요도를
#   뽑는다면, SHAP 은 각 예측을 변수 기여도로 분해해 '개별 예측' 단위까지 설명한다.
#   거리기반(KNN)·비선형 커널(SVR/SVC)처럼 feature_importance 가 거부하는 모델도
#   KernelExplainer(model-agnostic)로 설명 가능하다는 점이 핵심 차별점.
#
#   3개 함수로 SHAP 표준 시각화 4종을 모두 커버:
#     shap_analysis()        → SHAP 값 계산 + 요약표(summary_df) 반환
#                              + Summary Plot: ① Bar(영향력 순위)·② Beeswarm(방향·분포)
#     shap_dependence_plot() → ③ Dependence(변수-기여 관계·상호작용). 의미있는 변수쌍 자동.
#     shap_waterfall_plot()  → ④ Waterfall(개별 사례 세로 누적 분해)
#   개별 사례 시각화는 (Force 와 동일 목적이나) 최신 권장 표현인 Waterfall 로 통일했다.
#   뒤 2개 함수는 shap_analysis 가 반환한 DataFrame 의 .attrs(원시 SHAP 배열·모델공간
#   데이터·base value)를 그대로 받아 재계산 없이 그린다.
#
#   설계 시 추가로 고려한 점(요청 외 보강):
#     ① 파이프라인 transform: SHAP 은 모델이 '실제 학습한' 피처공간에서 계산해야 의미가
#        있으므로, 전처리(스케일·원핫·PCA)를 통과시킨 뒤 변환 후 피처명으로 설명한다.
#     ② 행 샘플링(max_samples): SHAP(특히 Kernel)은 느려서 대용량이면 재현가능 샘플링.
#     ③ 배경표본(background_samples): Linear/Kernel explainer 가 요구하는 기준분포를
#        kmeans 요약으로 축약(속도·안정).
#     ④ 분류 다중클래스: shap 값이 (n,f,c) 또는 클래스 리스트로 나오므로 class_index 로
#        설명 대상 클래스를 고르고, base value 도 같은 클래스로 정렬.
#     ⑤ KNN·SVM 폴백: TreeExplainer/LinearExplainer 불가 → KernelExplainer + 경고.
#     ⑥ 희소행렬(원핫 결과)→밀집 변환, base value 저장(Force Plot 필수) 등.


# 어떤 explainer 를 쓸지 결정하는 모델 그룹은 feature_importance 와 동일 분류를 재사용한다
# (_IMPORTANCE_TREE = 트리·부스팅 → TreeExplainer, _IMPORTANCE_LINEAR = 선형 → LinearExplainer).


def _shap_import():
    """shap 를 지연 import 한다 (무거운 선택적 의존성이라 모듈 로드시 강제하지 않음)."""
    try:
        import shap  # noqa: WPS433  (지연 import 의도)
        return shap
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError(
            "SHAP 분석에는 shap 패키지가 필요합니다. `pip install --upgrade shap` 후 사용하세요."
        ) from e


def _shap_prepare(estimator, x):
    """파이프라인/서치 객체를 풀어 (최종모델, 모델공간 입력 DataFrame, base_est, search_info).

    SHAP 은 모델이 실제 학습한 피처공간에서 계산해야 하므로, 전처리 단계를 통과시킨
    변환 결과를 '변환 후 피처명' 으로 DataFrame 화해 돌려준다. 파이프라인이 아니면
    입력을 그대로 쓴다.

    중요 — dtype 보존: build_pipeline 의 전처리는 set_output(transform='pandas') 라
    transform 결과가 DataFrame(범주형은 object dtype 유지)으로 나온다. 이를 float 로
    강제 변환하면 CatBoost 네이티브 범주형(cat_features)이 0.0 으로 깨져
    TreeExplainer 가 catboost.Pool 생성 시 거부한다. 따라서 DataFrame 이면 dtype 을
    그대로 보존하고, 수치 변환은 ndarray 폴백 경로에서 '가능할 때만' 한다.
    """
    base_est, search_info = _unwrap_estimator(estimator)

    if isinstance(base_est, Pipeline):
        model = base_est.named_steps.get('model', base_est[-1])
        pre = base_est[:-1]  # 모델 직전까지의 전처리 파이프라인
        x_trans = pre.transform(x)
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = None
    else:
        model = base_est
        x_trans = x
        feat_names = (list(model.feature_names_in_)
                      if hasattr(model, 'feature_names_in_')
                      else (list(x.columns) if isinstance(x, DataFrame) else None))

    index = x.index if isinstance(x, DataFrame) else None

    if isinstance(x_trans, DataFrame):
        # 전처리가 pandas 로 출력 — dtype(범주형 포함)·컬럼명·인덱스 그대로 보존
        x_df = x_trans.copy()
        if feat_names is not None and len(feat_names) == x_df.shape[1]:
            x_df.columns = feat_names
    else:
        # ndarray/희소행렬 폴백 — 밀집화 후 가능하면 수치(float), 불가하면 원형 유지
        if hasattr(x_trans, 'toarray'):
            x_trans = x_trans.toarray()
        arr = np.asarray(x_trans)
        try:
            arr = arr.astype(float)  # 전부 수치면 float, 범주 섞이면 그대로(object)
        except (ValueError, TypeError):
            pass
        if feat_names is None or len(feat_names) != arr.shape[1]:
            feat_names = [f'x{i}' for i in range(arr.shape[1])]
        x_df = DataFrame(arr, columns=feat_names, index=index)

    return model, x_df, base_est, search_info


def _shap_make_explainer(shap, model, model_class, background):
    """모델 유형별로 가장 적합한 SHAP explainer 를 생성하고 (explainer, 종류라벨) 반환.

    트리·부스팅 → TreeExplainer(정확·고속), 선형 → LinearExplainer(원시 배경분포 사용),
    그 외(KNN·SVM 등) → KernelExplainer(model-agnostic·느림, predict_proba 우선).
    background 는 모델공간 원시 표본 DataFrame. TreeExplainer 는 무시하고,
    LinearExplainer 는 그대로, KernelExplainer 는 kmeans 로 한 번 더 축약해 쓴다.
    """
    if model_class in _IMPORTANCE_TREE:
        return shap.TreeExplainer(model), 'TreeExplainer'
    if model_class in _IMPORTANCE_LINEAR:
        # LinearExplainer 는 kmeans 요약(DenseData)을 받지 못하므로 원시 표본을 넘긴다.
        return shap.LinearExplainer(model, background), 'LinearExplainer'

    # 폴백: 모델 무관 KernelExplainer. 분류는 확률, 회귀는 예측값을 설명 대상으로.
    if hasattr(model, 'predict_proba'):
        f = model.predict_proba
    elif hasattr(model, 'decision_function'):
        f = model.decision_function
    else:
        f = model.predict
    # Kernel 은 O(배경수 × 표본수) 라 kmeans 로 배경을 더 축약해 속도를 확보.
    bg = shap.kmeans(background, min(50, len(background)))
    return shap.KernelExplainer(f, bg), 'KernelExplainer'


def _shap_select_class(values, expected_value, task, class_index, n_classes):
    """다양한 형태의 SHAP 출력에서 설명 대상 1개 클래스 슬라이스를 골라낸다.

    values 는 회귀면 (n,f), 분류면 (n,f,c) 배열이나 클래스별 리스트로 나온다.
    분류일 때 class_index 클래스의 (n,f) 와 그 base value 를 반환한다.

    Returns:
        (shap_2d[n,f], base_value(float), used_class_index|None)
    """
    if isinstance(values, list):
        # 클래스별 (n,f) 리스트 — 분류 (구버전 shap 형태)
        idx = class_index if task == 'classification' else 0
        arr = np.asarray(values[idx])
        ev = expected_value[idx] if np.ndim(expected_value) else expected_value
        return arr, float(np.ravel(ev)[0]), (idx if task == 'classification' else None)

    arr = np.asarray(values)
    if arr.ndim == 3:                    # (n, f, c)
        idx = class_index if task == 'classification' else arr.shape[2] - 1
        ev_arr = np.atleast_1d(expected_value)
        ev = ev_arr[idx] if idx < ev_arr.shape[0] else ev_arr[-1]
        return arr[:, :, idx], float(ev), (idx if task == 'classification' else None)

    # (n, f) — 회귀 또는 이진(양성클래스 단일출력) explainer
    ev = float(np.ravel(expected_value)[0]) if np.ndim(expected_value) else float(expected_value)
    return arr, ev, (class_index if task == 'classification' else None)


def _shap_summary_height(n_disp):
    """표시 변수 수에 따라 Summary Plot 높이(px)를 동적 결정 (변수당 ~38px)."""
    return int(min(1600, max(360, 130 + 38 * n_disp)))


def _shap_bar_plot(summary, model_class, cls_tag, max_display, cum_ratio,
                   width, height, grid, save_path):
    """전역 중요도 막대(mean|SHAP|) + 누적 비율 텍스트 + 채택/탈락 색 구분.

    summary 는 mean_abs_shap 내림차순 정렬 + ratio/cum_ratio 포함(shap_analysis 작성).
    막대는 mean|SHAP| 크기, 오른쪽 텍스트는 그 변수까지의 누적 비율 — 정렬이 내림차순
    이므로 '상위 k개가 전체 영향력의 N% 를 설명' 을 한 줄로 읽을 수 있다.

    cum_ratio(= shap_dependence_plot 의 선정 기준) 에 누적 비율이 처음 도달하는 지점까지의
    변수를 '채택'(파랑·dependence 에 쓰임), 그 뒤는 '탈락'(회색)으로 색을 구분한다 —
    feature_importance 막대그래프와 동일한 개념.
    """
    n_all = len(summary)
    # 채택 개수 k — cum_ratio 에 누적 비율이 처음 도달하는 지점까지(경계 포함).
    #   shap_dependence_plot(_shap_rank_features)의 selection 과 동일한 규칙.
    cum_all = summary['cum_ratio'].values
    k = int(np.searchsorted(cum_all, cum_ratio)) + 1
    k = max(1, min(k, n_all))

    n_show = min(n_all, max_display) if max_display else n_all
    s = summary.head(n_show)
    names = list(s.index)
    vals = s['mean_abs_shap'].values
    cum = s['cum_ratio'].values
    selected = np.arange(n_show) < k  # 표시 막대의 채택 여부

    h = height if height is not None else _shap_summary_height(n_show)
    title = f"SHAP Bar (mean|SHAP| · 누적 {cum_ratio:.0%} 채택): {model_class}{cls_tag}"
    if n_show < n_all:
        title += f"  — 상위 {n_show}/{n_all}"
    fig, ax = my_plot.init(width=width, height=h, grid=grid, title=title,
                           xlabel='mean|SHAP|', ylabel='Feature')

    # barh 는 아래→위로 그려지므로 역순으로 넣어 중요도 큰 변수가 위로 오게
    y, v, c = names[::-1], vals[::-1], cum[::-1]
    colors = ['tab:blue' if sel else 'lightgray' for sel in selected][::-1]
    ax.barh(y, v, color=colors)
    vmax = float(vals.max()) if n_show else 1.0
    ax.set_xlim(0, vmax * 1.20)  # 오른쪽 텍스트 들어갈 여백
    for yi, (vv, cc) in enumerate(zip(v, c)):
        ax.text(vv + vmax * 0.01, yi, f'{cc:.0%}', va='center', ha='left',
                fontsize=10, color='tab:red')
    ax.legend(handles=[
        Patch(color='tab:blue', label=f'채택 (dependence 표시, 누적≤{cum_ratio:.0%})'),
        Patch(color='lightgray', label='탈락'),
    ], fontsize=11)
    my_plot.show(save_path=save_path)


def _shap_pareto_plot(summary, model_class, cls_tag, max_display,
                      width, height, grid, save_path):
    """전역 중요도 Pareto 차트 — 막대=mean|SHAP| 비율(ratio), 선=내림차순 누적 비율.

    summary 는 shap_analysis 가 만든 요약표(mean_abs_shap 내림차순, ratio/cum_ratio 포함).
    막대(좌축)와 누적선(우축, twinx)을 겹쳐 '상위 몇 개가 영향력의 N% 를 설명하는지' 를
    한눈에 보여준다. 변수 수에 따라 가로폭을 동적으로 넓힌다.
    """
    n_all = len(summary)
    n_show = min(n_all, max_display) if max_display else n_all
    s = summary.head(n_show)
    names = list(s.index)
    ratio = s['ratio'].values
    cum = s['cum_ratio'].values

    # 세로 막대 + 회전 라벨 — 변수 수에 비례해 가로폭 확대(과도하지 않게 상한)
    pw = int(min(3000, max(width, 70 * n_show + 200)))
    ph = height if height is not None else 560
    fig, (axL, axR) = my_plot.init(
        width=pw, height=ph, grid=grid, twinx=True,
        title=f"SHAP Cumulative (Pareto): {model_class}{cls_tag}")

    xs = np.arange(n_show)
    axL.bar(xs, ratio, color='tab:blue', alpha=0.85)
    axL.set_xticks(xs)
    axL.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    axL.set_ylabel('mean|SHAP| 비율', fontsize=14)

    axR.plot(xs, cum, color='tab:red', marker='o', linewidth=2, label='누적 비율')
    axR.set_ylim(0, 1.05)
    axR.set_ylabel('누적 비율', fontsize=14)
    axR.grid(False)  # 우축 그리드는 끄고 좌축 그리드만 사용(가독성)
    for xi, cv in zip(xs, cum):  # 누적 % 텍스트
        axR.annotate(f'{cv:.0%}', (xi, cv), textcoords='offset points',
                     xytext=(0, 7), ha='center', fontsize=9, color='tab:red')

    my_plot.show(save_path=save_path)


def shap_analysis(
    estimator,
    x: DataFrame,
    max_samples: int | None = 200,
    background_samples: int = 100,
    class_index: int | None = None,
    beeswarm: bool = True,
    bar: bool = True,
    cumulative: bool = False,
    cum_ratio: float = 0.9,
    max_display: int = 20,
    width: int = 1280,
    height: int | None = None,
    grid: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
    random_state: int = RANDOM_STATE,
) -> DataFrame:
    """학습된 회귀/분류 모델을 SHAP 으로 분석해 변수 기여도 요약표를 반환한다.

    회귀·분류 공용. 모델이 Linear/Ridge/Lasso/ElasticNet/SGD/KNN/SVM/DecisionTree/
    RandomForest/XGBoost/LightGBM/CatBoost 중 무엇이든, 하이퍼파라미터 튜닝 여부·
    Pipeline 구성 여부와 무관하게 동작한다. 모델 유형에 따라 explainer 를 자동 선택:
        - 트리·부스팅 → TreeExplainer (정확·고속)
        - 선형        → LinearExplainer (배경분포 사용)
        - KNN·SVM 등  → KernelExplainer (model-agnostic·느림, 경고 출력)

    estimator 가 Pipeline 이면 전처리(스케일·원핫·PCA)를 통과시킨 '모델 학습 피처공간'
    에서 SHAP 을 계산하고 변환 후 피처명으로 설명한다. GridSearchCV 등 search 객체면
    best_estimator_ 를 자동으로 푼다.

    Args:
        estimator: 학습된 회귀/분류 모델·파이프라인 또는 (refit 된) search 객체.
        x: 설명에 사용할 입력(원본 피처). 보통 x_train 또는 x_test.
        max_samples: SHAP 을 계산할 행 수 상한. 데이터가 많으면 재현가능 샘플링으로
            속도를 확보한다(기본 200). None 이면 전체 사용.
        background_samples: Linear/Kernel explainer 의 기준분포 표본 수. kmeans 로
            축약한다(기본 100). TreeExplainer 에는 사용되지 않음.
        class_index: (분류 전용) 설명할 클래스 인덱스. None 이면 마지막 클래스
            (이진이면 양성=1)를 설명한다. 회귀에서는 무시.
        beeswarm: True 면 ② Beeswarm(dot) Summary Plot 출력 — 변수별 SHAP 값의 방향·
            분포를 점으로 표시 (기본 True).
        bar: True 면 ① Bar 출력 — mean|SHAP| 영향력 순위 막대 + 막대 오른쪽에 내림차순
            누적 비율(cum_ratio) 텍스트 ('상위 k개가 영향력의 N% 설명'). cum_ratio 에
            도달하는 상위 변수는 '채택'(파랑), 나머지는 '탈락'(회색)으로 색 구분 —
            feature_importance 막대그래프와 동일 개념 (기본 False).
        cumulative: True 면 ③ Pareto 차트 출력 — 막대=mean|SHAP| 비율(ratio), 선=내림차순
            누적 비율(cum_ratio). '상위 몇 개가 영향력의 N% 를 설명하는지' 를 보여준다
            (기본 False). beeswarm·bar·cumulative 는 독립 토글이라 켠 것만(Bar→Beeswarm→
            Pareto 순) 그리고, 셋 다 False 면 그래프를 그리지 않는다.
        cum_ratio: bar 막대의 채택/탈락 색 구분 기준 (0, 1]. 누적 mean|SHAP| 비율이
            이 값에 도달하는 상위 변수까지 '채택'(기본 0.9). shap_dependence_plot 의
            cum_ratio 와 같은 값·의미 — bar 가 'dependence 에 쓰일 변수' 를 미리 보여준다.
        max_display: 그래프에 표시할 최대 변수 수 (기본 20).
        width: 그래프 가로(px). height: 세로(px). None 이면 변수 수에 따라 동적 결정.
        grid, save_path, verbose: 그래프·콘솔 옵션. 여러 그래프를 저장하면 파일명에
            종류 꼬리표(_bar/_beeswarm/_cumulative)를 붙여 덮어쓰기를 막는다.
        random_state: 행/배경 샘플링 재현용 시드.

    Returns:
        DataFrame: 변수별 SHAP 요약표(mean_abs_shap 내림차순). 컬럼=
            [mean_abs_shap, ratio(합1 정규화), cum_ratio(내림차순 누적),
             mean_shap, std_shap, direction(증가/감소/중립),
             cv, stability(안정적/비선형·불안정)]. index=Feature.
            .attrs 에 후속 플롯용 원시 데이터 저장:
              ['shap_values'] (n,f) ndarray — 설명 대상 클래스 기여도
              ['shap_df']     동일 값을 담은 DataFrame
              ['data']        모델공간 입력 DataFrame (shap_values 와 행 정렬)
              ['expected_value'] base value (Force Plot 기준선)
              ['feature_names'], ['explainer_type'],
              ['model_class'], ['task']('regression'|'classification'),
              ['class_index'], ['class_names'].
            search 객체였으면 .attrs['search'] 에 best_params 등.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 class_index 가 클래스 범위 밖.
        ImportError: shap 패키지가 없을 때.
    """
    # ============ STEP 01 · 입력 검증 ============
    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio must be in range (0.0, 1.0], got {cum_ratio}")
    shap = _shap_import()

    # ============ STEP 02 · 언랩 + 모델공간 입력 구성 ============
    model, x_df, base_est, search_info = _shap_prepare(estimator, x)
    model_class = type(model).__name__
    task = 'classification' if is_classifier(model) else 'regression'
    class_names = list(getattr(model, 'classes_', [])) or None
    n_classes = len(class_names) if class_names else 0

    # 분류 설명 클래스 결정 (기본: 마지막 = 이진이면 양성)
    if task == 'classification' and n_classes:
        if class_index is None:
            class_index = n_classes - 1
        elif not 0 <= class_index < n_classes:
            raise ValueError(
                f"class_index 는 0..{n_classes - 1} 범위여야 합니다 (classes_={class_names}).")

    # ============ STEP 03 · 행 샘플링(속도) + 배경표본 ============
    rng = np.random.RandomState(random_state)
    if max_samples is not None and len(x_df) > max_samples:
        pick = rng.choice(len(x_df), size=max_samples, replace=False)
        pick.sort()
        x_explain = x_df.iloc[pick]
    else:
        x_explain = x_df

    # 배경분포(원시 표본) — Linear/Kernel explainer 의 기준분포. 재현가능 샘플링.
    if len(x_df) > background_samples:
        bg_pick = rng.choice(len(x_df), size=background_samples, replace=False)
        bg_pick.sort()
        background = x_df.iloc[bg_pick]
    else:
        background = x_df

    # ============ STEP 04 · explainer 생성 + SHAP 값 계산 ============
    explainer, explainer_type = _shap_make_explainer(shap, model, model_class, background)
    if explainer_type == 'KernelExplainer':
        raw = explainer.shap_values(x_explain, silent=not verbose)
    else:
        raw = explainer.shap_values(x_explain)
    expected_value = getattr(explainer, 'expected_value', 0.0)

    shap_2d, base_value, used_class = _shap_select_class(
        raw, expected_value, task, class_index, n_classes)
    shap_2d = np.asarray(shap_2d, dtype=float)

    # ============ STEP 05 · 요약표 작성 (mean_abs / 방향 / 안정성) ============
    feat_names = list(x_explain.columns)
    shap_df = DataFrame(shap_2d, columns=feat_names, index=x_explain.index)

    mean_abs = shap_df.abs().mean().values
    mean_s = shap_df.mean().values
    std_s = shap_df.std().values
    direction = np.sign(mean_s)
    summary = DataFrame({
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_s,
        'std_shap': std_s,
        'direction': np.where(direction > 0, '증가', np.where(direction < 0, '감소', '중립')),
        'cv': std_s / (mean_abs + 1e-9),
    }, index=feat_names)
    summary['stability'] = np.where(summary['cv'] < 1, '안정적', '비선형/불안정')
    summary.index.name = 'Feature'
    summary = summary.sort_values('mean_abs_shap', ascending=False)

    # mean|SHAP| 를 합 1 로 정규화한 ratio 와 내림차순 누적 비율 cum_ratio (전역 중요도 분포)
    total_abs = float(summary['mean_abs_shap'].sum())
    summary['ratio'] = (summary['mean_abs_shap'] / total_abs) if total_abs > 0 else 0.0
    summary['cum_ratio'] = summary['ratio'].cumsum()
    # 컬럼 순서: mean_abs_shap → ratio → cum_ratio → 나머지
    summary = summary[['mean_abs_shap', 'ratio', 'cum_ratio',
                       'mean_shap', 'std_shap', 'direction', 'cv', 'stability']]

    # ── attrs: 후속 Dependence/Force Plot 이 재계산 없이 쓰는 원시 데이터 ──
    summary.attrs['shap_values'] = shap_2d
    summary.attrs['shap_df'] = shap_df
    summary.attrs['data'] = x_explain
    summary.attrs['expected_value'] = base_value
    summary.attrs['feature_names'] = feat_names
    # explainer 객체 자체는 attrs 에 넣지 않는다 — KernelExplainer 등은 deepcopy 불가
    # (cython 메모리뷰)라 summary.head()/.round() 같은 슬라이스가 attrs deepcopy 중 깨진다.
    summary.attrs['explainer_type'] = explainer_type
    summary.attrs['model_class'] = model_class
    summary.attrs['task'] = task
    summary.attrs['class_index'] = used_class
    summary.attrs['class_names'] = class_names
    if search_info is not None:
        summary.attrs['search'] = search_info

    # ============ STEP 06 · 시각화 (bar·beeswarm·cumulative 독립 토글) ============
    n_plots = int(bar) + int(beeswarm) + int(cumulative)
    cls_tag = (f" · class={class_names[used_class]}"
               if task == 'classification' and class_names else "")

    def _save_path(tag):
        # 여러 장 저장 시 덮어쓰기 방지로 파일명에 종류 꼬리표를 붙인다.
        if not save_path or n_plots <= 1:
            return save_path
        p = Path(save_path)
        return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))

    # ① Bar — 커스텀 막대(mean|SHAP|) + 누적 비율 텍스트 + 채택/탈락 색 구분
    if bar:
        _shap_bar_plot(summary, model_class, cls_tag, max_display, cum_ratio,
                       width, height, grid, _save_path('bar'))

    # ② Beeswarm — shap.summary_plot(dot)
    if beeswarm:
        n_disp = min(len(feat_names), max_display) if max_display else len(feat_names)
        h = height if height is not None else _shap_summary_height(n_disp)
        my_plot.init(width=width, height=h, grid=grid,
                     title=f"SHAP Summary (Beeswarm): {model_class}{cls_tag}")
        shap.summary_plot(shap_2d, x_explain, plot_type='dot',
                          max_display=max_display, show=False, plot_size=None)
        plt.xlabel('SHAP value')
        my_plot.show(save_path=_save_path('beeswarm'))

    # ③ Cumulative — Pareto 차트 (막대=mean|SHAP| 비율, 선=내림차순 누적 비율)
    if cumulative:
        _shap_pareto_plot(summary, model_class, cls_tag, max_display,
                          width, height, grid, _save_path('cumulative'))

    # ============ STEP 07 · 콘솔 요약 ============
    if verbose:
        print("\n" + "=" * 78)
        print(f"◆ SHAP Analysis: {model_class}  ({explainer_type}, {task})")
        print("=" * 78)
        if explainer_type == 'KernelExplainer':
            print(f"   ⚠ {model_class} 는 전용 explainer 가 없어 KernelExplainer(model-agnostic)"
                  f" 를 사용합니다.\n     계산이 느릴 수 있어 max_samples({max_samples})·"
                  f"background_samples({background_samples}) 로 표본을 제한했습니다.")
        if task == 'classification' and class_names:
            print(f"   ▷ 설명 대상 클래스: {class_names[used_class]} (index={used_class})")
        if search_info is not None:
            print(f"   ▷ {search_info['search_class']} → best_estimator_ 사용")
        print(f"   설명 표본 {len(x_explain)}행 × 변수 {len(feat_names)}개 · "
              f"base value={base_value:.4f}")
        print("=" * 78)
        top = summary.head(5)
        for f, row in top.iterrows():
            print(f"   {f:<22} mean|SHAP|={row['mean_abs_shap']:.4f}  "
                  f"방향={row['direction']}  {row['stability']}")
        print("=" * 78 + "\n")

    return summary


def _shap_select_main_features(shap_values, cum_ratio, top_k):
    """주변수(Dependence Plot 의 主축)를 mean|SHAP| 기준으로 자동 선정한다.

    feature_importance 의 채택 규칙과 동일한 원리: 변수별 mean|SHAP| 를 합 1 로 정규화
    → 내림차순 정렬 → 누적 비율이 cum_ratio 에 처음 도달하는 지점까지 채택(경계 포함).
    top_k 가 주어지면(None 아님) cum_ratio 대신 상위 top_k 개를 채택한다.

    Returns:
        (order: ndarray[int] 선정된 변수 인덱스(중요도 순), select_mode: str 설명문)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order_all = np.argsort(mean_abs)[::-1]
    n = len(order_all)

    if top_k is not None:
        k = max(1, min(int(top_k), n))
        return order_all[:k], f'top_k={top_k}'

    total = mean_abs.sum()
    if total <= 0:                       # 전 변수 기여 0 (미학습 등) — 최상위 1개만
        return order_all[:1], 'cum_ratio(폴백: 기여 0)'
    cum = np.cumsum(mean_abs[order_all] / total)
    k = int(np.searchsorted(cum, cum_ratio)) + 1
    k = max(1, min(k, n))
    return order_all[:k], f'cum_ratio={cum_ratio:.0%}'


def _shap_auto_pairs(shap_values, data, cum_ratio, top_k):
    """자동 선정한 주변수 각각에 대해 상호작용이 가장 강한 짝을 계산한다.

    주변수는 _shap_select_main_features(누적 mean|SHAP| 비율 또는 top_k)로 고르고,
    shap.utils.approximate_interactions 로 각 주변수와 상호작용이 큰 변수를 찾는다.

    Returns:
        (pairs: list[dict], select_mode: str)
        pairs = [{'feature', 'interaction_feature'}, ...] (중요도 순)
    """
    shap = _shap_import()
    cols = list(data.columns)
    order, select_mode = _shap_select_main_features(shap_values, cum_ratio, top_k)

    pairs = []
    for fi in order:
        inter = shap.utils.approximate_interactions(int(fi), shap_values, data)
        partner = cols[int(inter[0])] if len(inter) else cols[int(fi)]
        pairs.append({'feature': cols[int(fi)], 'interaction_feature': partner})
    return pairs, select_mode


def shap_dependence_plot(
    result: DataFrame,
    features: list[str] | None = None,
    cum_ratio: float = 0.9,
    top_k: int | None = None,
    interaction_index='auto',
    plot: bool = True,
    width: int = 1280,
    height: int = 640,
    grid: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """shap_analysis 결과로 Dependence Plot 을 그린다 (의미있는 변수쌍 자동 계산).

    Dependence Plot 은 한 변수의 값(x축) 대비 그 변수의 SHAP 값(y축)을, 상호작용이
    강한 다른 변수의 색으로 표시해 비선형·상호작용 패턴을 드러낸다.

    features 를 주지 않으면 주변수를 자동 선정한다 — feature_importance 의 채택 규칙과
    동일한 원리로, 변수별 mean|SHAP| 를 합 1 로 정규화 후 내림차순 누적 비율이
    cum_ratio 에 처음 도달하는 지점까지 채택한다(경계 포함). 즉 '영향력의 90% 를
    설명하는 핵심 변수' 만 그린다. 각 주변수마다 상호작용이 가장 큰 변수를 짝으로 잡는다.

    Args:
        result: shap_analysis() 가 반환한 DataFrame (.attrs 사용).
        features: 주변수 이름 리스트. None 이면 cum_ratio/top_k 로 자동 선정.
        cum_ratio: 자동 선정 시 채택할 누적 mean|SHAP| 비율 (0, 1]. 예: 0.9 면 전체
            영향력의 90% 를 설명하는 상위 변수까지 (기본 0.9). top_k 지정 시 무시.
        top_k: 양의 정수를 주면 cum_ratio 대신 mean|SHAP| 상위 top_k 개를 주변수로 채택.
        interaction_index: 색으로 쓸 상호작용 변수. 'auto'(기본) 면 변수별로 자동
            결정. 특정 변수명을 주면 모든 플롯에 고정.
        plot: True 면 Dependence Plot 출력 (기본 True). False 면 변수쌍만 계산해 반환.
            변수쌍마다 개별 플롯을 하나씩 따로 그린다(서브플롯 격자 아님 — rows=cols=1).
        width/height: 개별 플롯 1개의 px 크기.
        grid, save_path, verbose: 그래프·콘솔 옵션. save_path 가 있고 플롯이 여러 개면
            파일명에 주변수명을 꼬리표로 붙여 덮어쓰기를 막는다.

    Returns:
        list[dict]: 사용된 변수쌍 [{'feature','interaction_feature'}, ...].

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 top_k 가 양의 정수가 아닐 때.
    """
    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio must be in range (0.0, 1.0], got {cum_ratio}")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
    # features 는 변수명 리스트여야 한다. DataFrame(x_train) 등을 두 번째 인자로 넘기면
    # features 가 채워져 cum_ratio/top_k 자동선정이 통째로 무시되므로 명확히 막는다.
    if features is not None and not isinstance(features, (list, tuple)):
        raise TypeError(
            "features 는 변수명 리스트여야 합니다 (예: ['area', 'rooms']). "
            "shap_dependence_plot(result, ...) 은 데이터를 result.attrs 에서 읽으므로 "
            "x_train 같은 DataFrame 을 두 번째 인자로 넘기면 안 됩니다 — "
            "cum_ratio/top_k 로 자동선정하려면 features 는 비워 두세요. "
            f"받은 타입: {type(features).__name__}")

    shap = _shap_import()
    shap_values = result.attrs['shap_values']
    data = result.attrs['data']

    # 변수쌍 결정 — 자동(누적 mean|SHAP| 또는 top_k) 또는 사용자가 준 주변수
    if features is None:
        pairs, select_mode = _shap_auto_pairs(shap_values, data, cum_ratio, top_k)
    else:
        select_mode = f'features={features}'
        pairs = []
        for f in features:
            if f not in data.columns:
                raise ValueError(f"'{f}' 는 모델공간 변수에 없습니다. 가능: {list(data.columns)}")
            if interaction_index == 'auto':
                inter = shap.utils.approximate_interactions(
                    data.columns.get_loc(f), shap_values, data)
                partner = data.columns[int(inter[0])] if len(inter) else f
            else:
                partner = interaction_index
            pairs.append({'feature': f, 'interaction_feature': partner})

    if verbose:
        print(f"◆ SHAP Dependence — 변수쌍(주변수 → 상호작용변수) · 선정기준={select_mode}:")
        for p in pairs:
            print(f"   {p['feature']}  ×  {p['interaction_feature']}")

    if plot and pairs:
        # 서브플롯 격자가 아니라, 변수쌍마다 개별 플롯(rows=cols=1)을 하나씩 반복 출력.
        multi = len(pairs) > 1
        for p in pairs:
            inter = p['interaction_feature'] if interaction_index == 'auto' else interaction_index
            fig, ax = my_plot.init(
                width=width, height=height, grid=grid,
                title=f"SHAP Dependence: {p['feature']}  ×  {p['interaction_feature']}")
            shap.dependence_plot(p['feature'], shap_values, data,
                                 interaction_index=inter, ax=ax, show=False)
            sp = save_path
            if sp and multi:  # 여러 장 저장 — 주변수명 꼬리표로 덮어쓰기 방지
                path = Path(sp)
                sp = str(path.with_name(f"{path.stem}_{p['feature']}{path.suffix}"))
            my_plot.show(save_path=sp)

    return pairs


def _shap_representative_indices(shap_values, base_value, n):
    """예측값(=base + ΣSHAP) 분포의 균등 분위수에서 대표 관측치 n개를 선정한다.

    SHAP 가산성에 의해 행별 예측 = base_value + 그 행 SHAP 합. 이 예측값을 오름차순
    정렬해 0~100% 를 n 등분한 위치의 실제 행을 고른다(n=5 → 0/25/50/75/100% =
    최저·하위·중앙·상위·최고). 결정적이라 재현 가능하고, 출력 범위 전체를 대표한다.

    Returns:
        list[dict]: [{'index'(위치), 'pred'(예측값), 'quantile'(0~1)}, ...] 예측 오름차순.
    """
    pred = np.asarray(base_value, dtype=float) + np.asarray(shap_values, dtype=float).sum(axis=1)
    order = np.argsort(pred)
    N = len(order)
    n = max(1, min(int(n), N))
    pos = np.unique(np.round(np.linspace(0, N - 1, n)).astype(int))  # 분위 위치(중복 제거)
    return [{'index': int(order[p]),
             'pred': float(pred[order[p]]),
             'quantile': float(p / (N - 1)) if N > 1 else 0.0} for p in pos]


def shap_waterfall_plot(
    result: DataFrame,
    index: int | list | None = None,
    n: int = 5,
    max_display: int = 12,
    width: int = 1600,
    height: int | None = None,
    save_path: str | None = None,
    verbose: bool = False,
):
    """shap_analysis 결과로 관측치별 Waterfall Plot 을 그린다 (개별 사례 기여 분해).

    base value(평균 예측)에서 시작해 각 변수의 SHAP 기여를 위/아래 누적 막대로 쌓아
    최종 예측에 도달하는 과정을 세로로 보여준다(기여 큰 변수부터).

    index 를 주지 않으면(None) 대표 관측치를 자동 선정한다 — 예측값(=base + ΣSHAP)
    분포의 균등 분위수에서 n개(기본 5: 최저·하위·중앙·상위·최고)를 뽑아 각각 그린다.
    출력 범위 전체를 대표하며 결정적(재현 가능)이다.

    Args:
        result: shap_analysis() 가 반환한 DataFrame (.attrs 사용).
        index: 설명할 관측치의 위치(0-based, result.attrs['data'] 행 순서 기준 — 라벨
            인덱스 아님). None(기본)이면 예측값 분위수로 대표 n개 자동 선정. 정수면 그
            1개, 정수 리스트면 그 여러 개를 그린다.
        n: index=None 일 때 자동 선정할 대표 관측치 수 (기본 5). index 지정 시 무시.
        max_display: 표시할 최대 변수 수 (나머지는 'others' 로 합산, 기본 12).
        width: 그래프 가로(px). height: 세로(px). None 이면 변수 수에 따라 동적.
        save_path, verbose: 옵션. 여러 장 저장 시 파일명에 관측치 위치 꼬리표를 붙인다.

    Returns:
        설명에 사용된 관측치 행. 1개면 Series, 여러 개면 list[Series].

    Raises:
        ValueError: index 가 표본 행 범위를 벗어난 경우.
    """
    shap = _shap_import()
    shap_values = result.attrs['shap_values']
    data = result.attrs['data']
    base_value = result.attrs['expected_value']
    feat_names = result.attrs['feature_names']
    N = len(data)

    # ── 대상 관측치 결정: 자동(분위수) / 단일 정수 / 정수 리스트 ──
    if index is None:
        picks = _shap_representative_indices(shap_values, base_value, n)
        mode = f'auto: 예측값 분위수 {len(picks)}개'
    elif isinstance(index, (list, tuple, np.ndarray)):
        picks = [{'index': int(i), 'pred': None, 'quantile': None} for i in index]
        mode = f'지정 {len(picks)}개'
    else:
        picks = [{'index': int(index), 'pred': None, 'quantile': None}]
        mode = '지정 1개'

    for p in picks:  # 범위 검증
        if not -N <= p['index'] < N:
            raise ValueError(f"index 는 -{N}..{N - 1} 범위여야 합니다 (표본 {N}행).")

    cls = result.attrs.get('class_names')
    ci = result.attrs.get('class_index')
    cls_tag = f" · class={cls[ci]}" if cls and ci is not None else ""
    if verbose:
        print(f"◆ SHAP Waterfall — 선정={mode}{cls_tag}, base value={base_value:.4f}")

    n_disp = min(len(feat_names), max_display)
    h = height if height is not None else int(min(1200, max(320, 90 + 40 * n_disp)))
    multi = len(picks) > 1
    rows = []
    for p in picks:
        i = p['index']
        row = data.iloc[i]
        # 제목·로그용 예측/분위 꼬리표 (자동 선정일 때만 값이 있음)
        if p['pred'] is not None:
            info = f" — pred≈{p['pred']:.3g} (분위 {p['quantile']:.0%})"
        else:
            info = ""
        if verbose:
            print(f"   · 위치={i} (라벨={data.index[i]}){info}")

        expl = shap.Explanation(
            values=np.asarray(shap_values[i], dtype=float),
            base_values=float(base_value),
            data=row.values,
            feature_names=list(feat_names),
        )
        my_plot.init(width=width, height=h, grid=False,
                     title=f"SHAP Waterfall: obs#{i}{info}{cls_tag}")
        shap.plots.waterfall(expl, max_display=max_display, show=False)

        sp = save_path
        if sp and multi:  # 여러 장 — 관측치 위치 꼬리표로 덮어쓰기 방지
            path = Path(sp)
            sp = str(path.with_name(f"{path.stem}_obs{i}{path.suffix}"))
        my_plot.show(save_path=sp)
        rows.append(row)

    #return rows if multi else rows[0]




# ==========================================================================
# 분류(Classification) 전용 함수 — 뼈대 (TODO: 구현 예정)
# ==========================================================================
# 설계 메모 (작업 재개 시 참고):
#   - 회귀(reg_*) 와 1:1 대칭. 전처리(build_pipeline)·모델저장(save/load)·
#     변수중요도(feature_importance)·과적합 골격은 회귀와 '공유'하고, 여기서는
#     평가지표 레이어만 분류용으로 새로 구현한다.
#   - 분류 지표 스펙은 my_ml_const.py 의 _CLF_METRIC_SPECS / _CLF_CV_SCORERS
#     (현재 TODO 주석으로 초안만 있음) 를 채워서 가져다 쓴다.
#   - 분류 CV 는 클래스 비율 보존을 위해 StratifiedKFold 가 기본 (groups 면
#     StratifiedGroupKFold). 회귀의 KFold/GroupKFold 와 대비.
#   - 대부분의 분류 지표는 higher-is-better (LogLoss 만 lower). 따라서 과적합 gap
#     부호 처리는 회귀의 _METRIC_SPECS better 방향 로직을 그대로 재사용 가능.
#   - 이진/다중클래스 분기: average 인자('macro'|'micro'|'weighted'), ROC-AUC·
#     LogLoss 는 predict_proba/decision_function 필요 → 없으면 NaN 처리.
#
# 공유 헬퍼 재사용 메모 (clf_overfit 구현 시 회귀와 그대로 공유할 것):
#   - _unwrap_estimator : reg_overfit + feature_importance(내부) 가 이미 공유 중.
#     clf_overfit 도 그대로 호출 → search 객체(GridSearchCV 등) 언랩 공통 처리.
#     여러 그룹이 쓰는 범용 헬퍼라 그룹 접두사 없이 일반 이름으로 유지한다.
#   - _cv_scores : 현재 reg_overfit 전용이나, clf_overfit 도 동일하게 out-of-fold
#     점수 계산에 재사용할 'overfit 계열 공유' 헬퍼다. 그래서 _of_ 등 그룹 접두사를
#     붙이지 않고 일반 이름으로 둔다. (단 분류는 scorer 를 _CLF_CV_SCORERS 로,
#     CV 를 Stratified 계열로 넘겨야 함 — _cv_scores 시그니처는 그대로 활용 가능.)


def clf_score(
    estimator,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    average: str = 'macro',
) -> DataFrame:
    """[TODO] 분류 모델의 성능 지표를 계산한다 (reg_score 의 분류판).

    구현 계획:
        - 지표: Accuracy, Precision, Recall, F1, ROC-AUC, LogLoss.
        - y_pred = estimator.predict(x_test). ROC-AUC·LogLoss 는 predict_proba
          (없으면 decision_function, 그것도 없으면 NaN) 사용.
        - 이진 vs 다중클래스 자동 판별 → average 인자로 Precision/Recall/F1 집계
          ('macro'|'micro'|'weighted'). 이진은 pos_label 처리.
        - 반환: 모델명을 인덱스로 하는 지표 1행 DataFrame (reg_score 와 동일 형식).

    Args:
        estimator: 학습된 분류 모델/파이프라인.
        x_test, y_test: 테스트 설명변수·실제 라벨.
        average: 다중클래스 평균 방식 ('macro'|'micro'|'weighted').
    """
    raise NotImplementedError("clf_score 는 아직 구현되지 않았습니다 (분류 수업 단계에서 구현 예정).")


def clf_score_table(
    estimator: list | dict,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    primary: str = 'F1',
    aux: list[str] | None = ['Accuracy', 'ROC-AUC'],
    average: str = 'macro',
    verbose: bool = True,
) -> DataFrame:
    """[TODO] 여러 분류 모델을 한 번에 평가·순위화한 비교 테이블 (reg_score_table 의 분류판).

    구현 계획:
        - reg_score_table 의 4단계 순위 전략(주 지표 정렬 → 근소 격차 그룹 →
          보조 지표 결정적 결함 카운트 → 재정렬)을 그대로 재사용.
        - 회귀의 _METRIC_SPECS 자리에 분류용 _CLF_METRIC_SPECS 를 쓰고, 내부 점수
          계산은 clf_score 를 재사용한다.
        - 기본 주 지표는 F1 (불균형에 강건). 보조 지표는 Accuracy·ROC-AUC.

    Args:
        estimator: 모델 리스트 또는 {이름: 모델} dict.
        x_test, y_test: 테스트 데이터.
        primary: 순위 기준 주 지표 (기본 'F1').
        aux: 보조 지표 리스트.
        average: 다중클래스 평균 방식.
        verbose: 순위 산정 과정 출력 여부.
    """
    raise NotImplementedError("clf_score_table 은 아직 구현되지 않았습니다 (분류 수업 단계에서 구현 예정).")


def clf_overfit(
    estimator,
    x_train: DataFrame,
    y_train: DataFrame | np.ndarray,
    x_test: DataFrame,
    y_test: DataFrame | np.ndarray,
    metrics: list[str] | str | None = ['F1', 'Accuracy'],
    threshold: float = 0.15,
    underfit_threshold: float = 0.6,
    cv_score: bool = True,
    cv: int = 5,
    groups=None,
    cv_stability: bool = False,
    cv_std_threshold: float = 0.05,
    average: str = 'macro',
    fit_params: dict | None = None,
    learning_curve: bool = True,
    width: int = 1280,
    height: int = 640,
    grid: bool = True,
    save_path: str | None = None,
    verbose: bool = True,
) -> DataFrame:
    """[TODO] Train/CV/Test 성능으로 분류 모델의 과적합을 진단한다 (reg_overfit 의 분류판).

    구현 계획:
        - reg_overfit 의 진단 골격(Train↔CV gap, 과소/과대/일반화 3분류, 학습곡선)을
          그대로 재사용. 차이점만 분류용으로 교체:
            · scorer: _CLF_CV_SCORERS (회귀의 _CV_SCORERS 대응).
            · 과소적합 게이트: 회귀의 train R2 대신 train Accuracy(또는 F1) <
              underfit_threshold (기본 0.6 — R2 0.3 과 다른 스케일이라 별도 기본값).
            · CV 분할: StratifiedKFold(int 면 승격), groups 면 StratifiedGroupKFold.
            · 대부분 higher-is-better 라 gap 부호 로직은 회귀와 동일하게 재사용.
              LogLoss 만 lower-is-better → _CLF_METRIC_SPECS better 로 처리.
        - 반환·attrs(diagnosis/underfit/overfit/gap_basis 등) 구조는 reg_overfit 과 동일.

    Args:
        estimator: 학습된 분류 모델/파이프라인 또는 (refit 된) search 객체.
        x_train, y_train, x_test, y_test: 학습·holdout 데이터.
        metrics: 표시·진단할 분류 지표 (기본 ['F1','Accuracy']).
        threshold: 과대적합 경계 (Gap% ≥ threshold → 과대적합).
        underfit_threshold: train Accuracy 가 이 값 미만이면 과소적합 (기본 0.6).
        cv_score, cv, groups, cv_stability, cv_std_threshold: CV 관련 옵션
            (reg_overfit 과 동일하되 CV 는 Stratified 계열).
        average: 다중클래스 평균 방식.
        fit_params, learning_curve, width, height, grid, save_path, verbose:
            reg_overfit 과 동일.
    """
    raise NotImplementedError("clf_overfit 은 아직 구현되지 않았습니다 (분류 수업 단계에서 구현 예정).")