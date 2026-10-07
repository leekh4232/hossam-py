# ---- 기본 참조 ----------------------------------------------------
import joblib                                   # 모델 저장
import numpy as np                              # 배열 연산
from pathlib import Path                        # 경로 처리
from pandas import concat, DataFrame, Series    # 데이터프레임 처리
from IPython.display import display             # 출력 기능

from . import RANDOM_STATE                      # 재현성을 위한 랜덤시드
from . import my_prep                           # 스케일러 목록(SCALERS)
from . import my_plot                           # 시각화 참조
from .my_vif_selector import VIFSelector        # 다중공선성 제거
from .my_outlier_clipper import OutlierClipper  # 이상치 경계값 대체

# ---- 머신러닝 파이프라인 구축 관련 참조 --------------------------------
from sklearn.pipeline import Pipeline               # 전처리 + 모델 연결
from sklearn.impute import SimpleImputer            # 결측치 대체
from sklearn.preprocessing import OneHotEncoder     # 더미변수 인코딩
from sklearn.compose import ColumnTransformer       # 연속형·명목형 분기 처리
from sklearn.decomposition import PCA               # 차원 축소
from sklearn.preprocessing import LabelEncoder      # 종속변수 라벨 인코딩(cls_baseline)
from sklearn.preprocessing import label_binarize    # 다중분류 지표의 일대다(OvR) 변환

# ---- 성능 지표 계산 -------------------------------------------------
# 회귀 지표
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_squared_log_error, mean_absolute_percentage_error
)

# 분류 지표
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, multilabel_confusion_matrix
)

# ---- 머신러닝 모델 참조 참조 -----------------------------------------
# xgboost·lightgbm·catboost 는 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 분류 모델 (cls_baseline)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# --------------------------------------------------------
# 학습된 모델을 joblib 으로 직렬화해 저장
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

    # 절대경로로 변환하여 리턴
    return save_path.resolve()


# --------------------------------------------------------
# 저장된 모델 파일을 로드해서 반환
# --------------------------------------------------------
def load_model(load_path):
    """저장된 모델 파일을 로드해서 반환한다.

    Args:
        load_path (str | Path): 모델 파일 경로

    Returns:
        저장 시점의 모델 객체.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
    """
    # 경로 문자열을 경로 객체로 변환
    load_path = Path(load_path)

    # 파일이 없으면 예외 발생
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    # joblib 로 모델 로드후 반환
    return joblib.load(load_path)


# --------------------------------------------------------
# 전처리 파이프라인 + 모델 학습
# --------------------------------------------------------
def fit_pipeline(model, x_train, y_train, nominal_cols=None, *,
                 # --- 1) 결측치 대체 ---
                 impute=False,                        # 결측치 대체 수행 여부
                 numeric_impute='median',             # 연속형 대체 전략
                 categorical_impute='most_frequent',  # 명목형 대체 전략
                 # --- 2) 이상치 대체 (경계값 클리핑, 행 삭제 없음) ---
                 outlier=False,                       # 이상치 대체 수행 여부
                 outlier_method='iqr',                # 이상치 판단 방식 (iqr / zscore)
                 # --- 3) 다중공선성 제거 (VIF) ---
                 vif=False,                           # 다중공선성 제거 수행 여부
                 vif_threshold=10.0,                  # VIF 임계값
                 # --- 4) 정규화 ---
                 scale=False,                         # 정규화 수행 여부
                 scale_method='standard',             # 사용할 스케일러 이름 (standard / minmax / robust / maxabs)
                 # --- 5) 차원 축소 (PCA) ---
                 pca=False,                           # 차원 축소 수행 여부
                 pca_variance=0.95,                   # 유지할 누적 설명분산 비율
                 # --- 6) 더미변수 인코딩 ---
                 encode=True,                         # 더미변수 인코딩 수행 여부
                 drop_first=False,                    # 첫 번째 더미 제거 여부 (더미 트랩 방지)
                 # --- 기타 ---
                 name=None,                           # 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다
                 save_path=None,                      # 학습이 끝난 모델의 저장 경로 (.pkl)
                 verbose=True,                        # 단계별 전처리 내역 출력 여부
                 **fit_params):                       # 모델의 fit 에 그대로 넘길 인자 (예: model__cat_features)
    """전처리 단계를 쌓아 모델까지 연결한 뒤, 훈련 데이터로 학습해서 반환한다.

    Args:
        model: 파이프라인 끝에 연결할 사이킷런 모델.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        nominal_cols (list): 명목형 컬럼명. None 이면 자동 선택 (기본값: None).
        impute (bool): 결측치 대체 여부 (기본값: False).
        numeric_impute (str): 연속형 대체 전략 — mean/median/most_frequent/constant (기본값: 'median').
        categorical_impute (str): 명목형 대체 전략 — most_frequent/constant (기본값: 'most_frequent').
        outlier (bool): 이상치를 경계값으로 대체할지 여부 (기본값: False).
        outlier_method (str): 이상치 판단 방식 — iqr/zscore (기본값: 'iqr').
        vif (bool): 다중공선성 제거 여부 (기본값: False).
        vif_threshold (float): VIF 임계값 (기본값: 10.0).
        scale (bool): 정규화 여부 (기본값: False).
        scale_method (str): 스케일러 이름 — standard/minmax/robust/maxabs (기본값: 'standard').
        pca (bool): 차원 축소 여부 (기본값: False).
        pca_variance (float): 유지할 누적 설명분산 비율 (기본값: 0.95).
        encode (bool): 더미변수 인코딩 여부. False 면 명목형을 원본 그대로 넘긴다 (기본값: True).
        drop_first (bool): 첫 번째 더미 제거 여부 (기본값: False).
        name (str): 모델을 구분할 이름 (기본값: None).
        save_path (str): 학습된 모델의 저장 경로(.pkl) (기본값: None).
        verbose (bool): 전처리 내역 출력 여부 (기본값: True).
        **fit_params: 모델의 fit 에 넘길 인자. `단계명__인자명` 형식 (예: model__cat_features).

    Returns:
        Pipeline: 학습된 파이프라인. 전처리 구성을 담은 `pipeline_info_` 와 이름 `name_` 속성이 붙는다.

    Raises:
        KeyError: nominal_cols 에 x_train 이 갖고 있지 않은 컬럼이 있는 경우.
        ValueError: 스케일러 이름이 유효하지 않은 경우.
    """
    # --- 1) 명목형 컬럼 확정 ---
    # 지정이 없으면 category/object 타입을 자동으로 선택한다
    if nominal_cols is None:
        nominal_cols = list(x_train.select_dtypes(include=['category', 'object']).columns)
    else:
        missing = []
        for c in nominal_cols:
            if c not in x_train.columns:
                missing.append(c)

        if missing:
            raise KeyError(f'x_train 에 존재하지 않는 컬럼입니다: {missing}')

        nominal_cols = list(nominal_cols)

    # --- 2) 연속형 컬럼 확정 ---
    # 수치형 중에서 명목형으로 지정된 것을 뺀 나머지.
    # 이상치대체·정규화·다중공선성·차원축소의 대상이 된다
    continuous = []
    for c in x_train.select_dtypes(include='number').columns:
        if c not in nominal_cols:
            continuous.append(c)

    # --- 3) 대상 요약 출력 ---
    if name:
        model_name = name
    else:
        model_name = model.__class__.__name__.lower()
        model_name = model_name.removesuffix("regressor")
        model_name = model_name.removesuffix("regression")
        model_name = model_name.removesuffix("classifier")

    if verbose:
        print(f'대상: {x_train.shape[0]}행 x {x_train.shape[1]}열 | 모델: {model_name}')
        print(f'명목형: {nominal_cols}')
        print(f'연속형: {continuous}')

        # impute 를 끈 채 결측치가 남아 있으면 대부분의 모델이 학습 도중 멈춘다.
        # 다만 XGBoost·LightGBM 처럼 결측치를 자체 처리하는 모델도 있으므로 안내만 한다
        if not impute:
            na_cols = x_train.columns[x_train.isna().any()].tolist()

            if na_cols:
                print(f'참고: 결측치가 있는 컬럼 {na_cols} | '
                      f'모델이 결측치를 직접 다루지 못하면 impute=True 로 설정하세요.')

    # --- 4) 연속형 전처리 단계 구성 ---
    numeric_steps = []      # 연속형 변수의 처리 순서를 저장할 리스트
    numeric_report = []     # 전처리 단계별 설명을 저장할 리스트

    if continuous and impute:
        # 처리 단계의 이름(imputer)과 수행할 클래스(SimpleImputer)를 지정한다.
        # SimpleImputer는 sklearn에서 제공하는 결측치 처리 클래스
        numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_impute)))
        numeric_report.append(f'결측치 대체({numeric_impute})')

    if continuous and outlier:
        # 이상치 대체 단계의 이름(outlier_clipper)과 수행할 클래스(OutlierClipper)를 지정한다.
        # OutlierClipper는 my_outlier_clipper.py에서 정의한 이상치 처리 클래스
        numeric_steps.append(('outlier_clipper', OutlierClipper(method=outlier_method)))
        numeric_report.append(f'이상치 대체({outlier_method})')

    if continuous and vif:
        # 다중공선성 제거 단계의 이름(vif_selector)과 수행할 클래스(VIFSelector)를 지정한다.
        # VIFSelector는 my_vif_selector.py에서 정의한 다중공선성 제거 클래스
        numeric_steps.append(('vif_selector', VIFSelector(threshold=vif_threshold)))
        numeric_report.append(f'다중공선성 제거(VIF >= {vif_threshold})')

    if continuous and scale:
        # 스케일러 이름은 my_prep.scaling() 과 같은 표기를 받는다 ('StandardScaler' -> 'standard')
        scale_name = scale_method.lower().replace('scaler', '').strip()

        # 오타를 냈을 때 KeyError 대신 사용 가능한 이름을 알려준다
        if scale_name not in my_prep.SCALERS:
            raise ValueError(f"지원하지 않는 스케일러입니다: '{scale_method}' "
                             f"(사용 가능: {list(my_prep.SCALERS.keys())})")

        # 연속형 전처리 단계에 스케일러를 추가한다.
        # my_prep.SCALERS[scale_name]()는 해당 스케일러 클래스의 인스턴스를 생성한다.
        # 예: my_prep.SCALERS['standard']()는 StandardScaler()를 반환한다.
        numeric_steps.append(('scaler', my_prep.SCALERS[scale_name]()))
        numeric_report.append(f'정규화({scale_name})')

    if continuous and pca:
        # 차원 축소 단계의 이름(pca)과 수행할 클래스(PCA)를 지정한다.
        # PCA는 sklearn에서 제공하는 차원 축소 클래스
        numeric_steps.append(('pca', PCA(n_components=pca_variance, random_state=RANDOM_STATE)))
        numeric_report.append(f'차원 축소(설명분산 {pca_variance})')

    # --- 5) 명목형 전처리 단계 구성 ---
    categorical_steps = []
    categorical_report = []

    if nominal_cols and impute:
        categorical_steps.append(('imputer', SimpleImputer(strategy=categorical_impute)))
        categorical_report.append(f'결측치 대체({categorical_impute})')

    if nominal_cols and encode:
        categorical_steps.append(('onehot', OneHotEncoder(
            # 학습 알고리즘 유형에 따라 drop_first가 선택적으로 수행되어야 한다.
            drop='first' if drop_first else None,
            # handle_unknown='ignore' 는 훈련 때 없던 범주가 검증 데이터에 나타나도
            # 예외 대신 전부 0인 더미로 처리해, 예측이 중간에 끊기지 않게 한다
            handle_unknown='ignore',
            sparse_output=False,
        )))
        categorical_report.append(f'더미변수 인코딩(drop_first={drop_first})')

    # --- 6) 전처리 단계 출력 ---
    if verbose:
        print('\n전처리 단계')

        for label, columns, report in (('연속형', continuous, numeric_report),
                                       ('명목형', nominal_cols, categorical_report)):
            if not columns:      text = '(대상 없음)'
            elif not report:     text = '(변환 없음)'
            else:                text = ' → '.join(report)

            print(f'  {label}: {text}')

    # --- 7) 파이프라인 조립 ---
    # 컬럼 종류별로 다른 전처리를 적용해야 하므로 ColumnTransformer 로 갈래를 나눈다.
    # set_output(transform='pandas') 를 해야 VIFSelector 가 컬럼명을 보고 변수를 고를 수 있다.
    # 갈래가 두 개뿐이라 병렬로 나눠 봐야 프로세스를 띄우는 비용이 더 크고, GridSearchCV 안에서
    # 돌 때는 병렬이 겹쳐 코어를 과점유하므로 순차 처리한다(n_jobs 미지정)
    preprocessor = ColumnTransformer([
        ('num', Pipeline(numeric_steps) if numeric_steps else 'passthrough', continuous),
        ('cat', Pipeline(categorical_steps) if categorical_steps else 'passthrough', nominal_cols),
    ], remainder='passthrough', verbose_feature_names_out=False)

    preprocessor.set_output(transform='pandas')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    # --- 8) 모델 학습 ---
    pipeline.fit(x_train, y_train, **fit_params)

    if verbose:
        print(f'\n모델 학습 완료: {model_name}')

    # --- 9) 보고에 필요한 정보를 결과 객체에 붙여 반환 ---
    pipeline.pipeline_info_ = {
        'model_class': model_name,
        'nominal_cols': nominal_cols,
        'continuous_cols': continuous,
        'impute': impute,
        'numeric_impute': numeric_impute,
        'categorical_impute': categorical_impute,
        'outlier': outlier,
        'outlier_method': outlier_method,
        'vif': vif,
        'vif_threshold': vif_threshold,
        'scale': scale,
        'scale_method': scale_method,
        'pca': pca,
        'pca_variance': pca_variance,
        'encode': encode,
        'drop_first': drop_first,
    }

    # 모델을 구분할 이름 (성능 비교표의 인덱스로 쓴다)
    pipeline.name_ = model_name

    # --- 10) 학습된 모델 저장 (선택) ---
    if save_path:
        save_model(pipeline, save_path)

        if verbose:
            print(f'모델 저장: {save_path}')

    return pipeline



# --------------------------------------------------------
# 파이프라인·탐색 객체를 풀어 모델과 전처리 단계를 분리
# --------------------------------------------------------
def _unwrap_estimator(estimator):
    """학습 결과물을 풀어 (모델, 전처리 단계, 원본 추정기, best_params) 를 반환한다.

    feature_importance 와 SHAP 은 둘 다 '최종 모델' 과 '모델 직전까지의 전처리' 를
    따로 필요로 한다. 어느 쪽이든 GridSearchCV 로 감싼 파이프라인, 맨 파이프라인,
    단독 모델을 모두 받을 수 있어야 해서 이 풀어내기를 한곳에 모아 둔다.

    Args:
        estimator: 학습된 모델·파이프라인 또는 GridSearchCV 등 탐색 객체

    Returns:
        tuple: (model, pre, base_est, search_params). pre 는 모델 직전까지를 잘라낸
            전처리 파이프라인이며, 단독 모델이면 None 이다.
    """
    # 탐색 객체면 최적 모델을 꺼낸다
    base_est = getattr(estimator, 'best_estimator_', estimator)
    search_params = getattr(estimator, 'best_params_', None)

    if isinstance(base_est, Pipeline):
        # fit_pipeline 은 마지막 단계 이름을 'model' 로 붙이지만, 직접 만든
        # 파이프라인은 이름이 다를 수 있어 맨 뒤 단계로 대신한다
        model = base_est.named_steps.get('model', base_est[-1])
        pre = base_est[:-1]                     # 모델 직전까지의 전처리 단계
    else:
        # 파이프라인이 아닌 단독 모델도 허용 — 전처리 단계가 없다
        model = base_est
        pre = None

    return model, pre, base_est, search_params


# --------------------------------------------------------
# 회귀 모델의 성능 지표 계산
# --------------------------------------------------------
def reg_score(estimator, x_test, y_test):
    """회귀 모델의 성능 지표(R2/MAE/MSE/RMSE/RMSLE/MAPE/MPE)를 계산한다.

    Args:
        estimator: 학습이 완료된 사이킷런 회귀 모델 또는 파이프라인. GridSearchCV 같은
            하이퍼파라미터 탐색 객체를 주면 내부의 best_estimator_ 로 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.

    Returns:
        DataFrame: 모델 클래스명을 인덱스로 하는 지표 1행. 음수·0 으로 계산이 불가능하면 NaN.
    """
    # --- 1) 평가할 모델을 확정하고 예측을 수행한다 ---
    # 탐색 객체면 최적 모델을, 파이프라인이면 마지막 단계에서 모델명을 꺼낸다.
    # 탐색 객체의 클래스명('GridSearchCV')으로는 어떤 모델의 점수인지 알 수 없기 때문이다.
    # 예측 자체는 전처리가 붙은 파이프라인 전체(base_est)로 해야 한다.
    model, _, base_est, _ = _unwrap_estimator(estimator)

    # 모델 클래스명을 추출한다.
    classname = type(model).__name__

    # 검증 데이터로 예측 수행
    y_pred = base_est.predict(x_test)

    # DataFrame·Series·ndarray 를 1차원 배열로 통일
    y_test_array = np.asarray(y_test).ravel()

    # --- 2) 성능 지표 계산 ---
    # 기본 지표: 어떤 데이터에서든 항상 계산된다
    r2 = r2_score(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    mse = mean_squared_error(y_test_array, y_pred)
    rmse = np.sqrt(mse)

    # RMSLE: 로그를 취하므로 음수가 하나라도 있으면 계산 불가
    if np.any(y_test_array < 0) or np.any(y_pred < 0):
        rmsle = np.nan
    else:
        rmsle = np.sqrt(mean_squared_log_error(y_test_array, y_pred))

    # MAPE·MPE: 실제값으로 나누므로 0 이 하나라도 있으면 계산 불가.
    # 두 지표 모두 비율(0.05 = 5%)로 돌려준다. R2 와 단위를 맞추기 위함이며,
    # 백분율이 필요하면 사용하는 쪽에서 100 을 곱한다
    if np.any(y_test_array == 0):
        mape = np.nan
        mpe = np.nan
    else:
        mape = mean_absolute_percentage_error(y_test_array, y_pred)
        mpe = np.mean((y_test_array - y_pred) / y_test_array)

    # --- 3) 계산한 지표를 모델명 1행짜리 표로 정리해 반환 ---
    score_df = DataFrame({
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'MAPE': mape,
        'MPE': mpe,
    }, index=[classname])
    score_df.index.name = 'Model'

    return score_df


# --------------------------------------------------------
# 모델별 점수표에 4단계 전략으로 Rank 를 매긴다
# --------------------------------------------------------
def _rank_score_table(final_score_table, metric_specs, primary, aux, verbose=True,
                      plot=True, title=None, width=1280, height=640, save_path=None):
    """모델별 점수표를 받아 4단계 전략으로 순위를 매긴 비교표와 성능 비교 그래프를 만든다.

    주 지표 하나만으로는 소수점 차이로 1등이 갈리므로, ① 주 지표로 정렬 → ② 1등의 5%
    이내를 '근소 격차 그룹' 으로 묶기 → ③ 그룹 내부의 보조 지표 결함 개수 세기 →
    ④ (결함 수, 주 지표) 순으로 그룹 내부 재정렬, 의 순서로 순위를 매긴다.

    Args:
        final_score_table (DataFrame): index=모델 이름, 컬럼=지표인 점수표
        metric_specs (dict): 지표별 방향·결함 판정 방식
        primary (str): 순위를 가르는 주 지표
        aux (list): 결함 판정에 쓸 보조 지표
        verbose (bool): 판정 과정 출력 여부
        plot (bool): 성능 비교 그래프 출력 여부
        title (str): 그래프 제목. 뒤에 '(지표 기준)' 이 붙는다
        width (int): 그래프 가로 크기(픽셀)
        height (int): 그래프 세로 크기(픽셀)
        save_path (str): 그래프 이미지 저장 경로

    Returns:
        DataFrame: 성능 비교표
    """
    # --- 1) 지표별 key 를 만들어 방향을 통일한다 ---
    # 지표마다 좋은 방향이 달라(낮을수록·높을수록·0에 가까울수록) 그대로 두면 정렬·그룹·
    # 결함·격차 계산마다 비교식이 세 갈래로 갈린다. 여기서 한 번만 방향을 맞춰 두면
    # 이후 단계는 전부 "key 가 작을수록 좋다, 1등과의 격차 = key - 1등 key" 로 읽힌다.
    
    # 인덱스만 갖는 빈 DataFrame 을 만들어, 지표별 key 를 채워 넣는다
    keys = DataFrame(index=final_score_table.index)

    for m in [primary] + aux:                       # `주 지표 + 보조 지표`만큼 반복
        better = metric_specs[m]['better']          # 낮을수록 좋은지, 높을수록 좋은지, 0에 가까울수록 좋은지

        if better == 'lower':                       # 낮을수록 좋은 지표
            keys[m] = final_score_table[m]          # 그대로 쓴다
        elif better == 'higher':                    # 높을수록 좋은 지표
            keys[m] = -final_score_table[m]         # 부호를 뒤집어 작을수록 좋게 만든다
        else:                                       # 0에 가까울수록 좋은 지표
            keys[m] = final_score_table[m].abs()    # 0 에서 벗어난 크기만 본다

    if verbose:
        print('\n' + '=' * 70)
        print(f'◆ Score Table Ranking : primary={primary}, aux={aux}')
        print('=' * 70)

    # --- 2) 주 지표의 key 가 작은 순으로 정렬한다 (NaN 은 맨 뒤) ---
    order = keys[primary].sort_values(kind='mergesort').index
    sorted_table = final_score_table.loc[order]
    keys = keys.loc[order]

    if verbose:
        direction_label = {
            'lower': '낮을수록 좋음',
            'higher': '높을수록 좋음',
            'closer_to_zero': '0에 가까울수록 좋음',
        }[metric_specs[primary]['better']]
        print(f'\n▲ step1: 주 지표({primary}) 기준 정렬 — {direction_label}')
        for i, (name, val) in enumerate(sorted_table[primary].items(), 1):
            print(f'   {i:>2}. {name:<14} {primary:<6}= {val:>16.3f}')

    # --- 3) 1등과의 격차가 5% 이내인 모델을 '근소 격차 그룹' 으로 묶는다 ---
    best_key = keys[primary].min()
    allow = abs(best_key) * 0.05                       # 허용 격차 = 1등 크기의 5%
    close_mask = keys[primary] <= best_key + allow     # NaN 은 비교가 False 라 그룹 외부

    close_group = sorted_table[close_mask]
    outside_group = sorted_table[~close_mask]

    if verbose:
        print(f'\n▲ step2: 근소 격차 그룹 묶기 (1등의 5% 이내)')
        print(f'   - 1등 {primary:<6} : {sorted_table[primary].iloc[0]:.3f}')
        print(f'   - 허용 격차    : 1등보다 {allow:.3f} 까지')
        print(f'   - 근소 격차 그룹 ({len(close_group)}) : {list(close_group.index)}')
        print(f'   - 그룹 외부     ({len(outside_group)}) : {list(outside_group.index)}')

    # --- 4) 그룹 안에서 보조 지표의 '결정적 결함' 개수를 센다 ---
    # 그룹 1등보다 허용 격차를 넘게 나쁘면 결함 1개. 계산되지 않은(NaN) 지표도 결함이다.
    # 그룹에 1등만 있으면 비교할 상대가 없으므로 건너뛴다.
    if len(close_group) > 1:
        # --- 4-1) 결함 여부표를 만든다 ---
        flaws = DataFrame(index=close_group.index)     # 모델 × 보조 지표 결함 여부표

        if verbose:
            print(f'\n▲ step3: 보조 지표 결정적 결함 점검 (근소 격차 그룹 내부)')

        # --- 4-2) 각 보조 지표별로 결함 여부를 계산한다 ---
        for m in aux:   # `aux` 에 지정된 보조 지표만큼 반복
            key = keys.loc[close_group.index, m]        # 근소 격차 그룹 내부의 m 지표 key
            best_key = key.min()                        # 근소 격차 그룹 내부의 m 지표 1등 key
            threshold = metric_specs[m]['threshold']    # 결함 판정 임계값

            # 허용 격차: 상한이 없는 지표(오차·DOR)는 1등 크기의 threshold 비율,
            #            상한이 정해진 지표(R2·정확도 등)는 threshold 그 자체
            if metric_specs[m]['flaw_type'] in ('rel_excess', 'rel_drop'):
                allow = abs(best_key) * threshold
            else:
                allow = threshold

            flaws[m] = (key > best_key + allow) | key.isna()

            if verbose:
                best = -best_key if metric_specs[m]['better'] == 'higher' else best_key
                print(f'       · {m:<6} best={best:>10.3f}   결함조건: 1등보다 {allow:.3f} 넘게 나쁘면')

        # --- 4-3) 각 모델별 결함 개수를 카운트하여 문자열 출력 ---
        if verbose:
            for name in close_group.index:
                hit = [m for m in aux if flaws.loc[name, m]]
                print(f'       · {name:<14} {len(hit)}개 {hit if hit else "(결함 없음)"}')

        # --- 4-4) (결함 수, 주 지표 key) 순으로 그룹 내부를 재정렬한다 ---
        # 결함이 적은 모델이 위로, 동률이면 주 지표가 좋은 모델이 위로 간다
        order = DataFrame({
            'flaws': flaws.sum(axis=1),
            'key': keys.loc[close_group.index, primary],
        }).sort_values(['flaws', 'key'], kind='mergesort').index
        close_group = close_group.loc[order]

    # --- 5) 재정렬한 그룹 + 그룹 외부를 이어 붙이고 Rank·Group 을 붙인다 ---
    # 근소격차그룹과 그룹 외부를 합쳐 최종 점수표를 만든다.
    final_score_table = concat([close_group, outside_group])

    # 최종 점수표에 Rank 컬럼을 추가 --> 1등부터 순위를 매긴다
    final_score_table.insert(0, 'Rank', range(1, len(final_score_table) + 1))

    # 최종 점수표에 Group 컬럼을 추가 --> 근소격차그룹은 'Contender', 그룹 외부는 'Outside'
    final_score_table.insert(1, 'Group', [
        'Contender' if name in close_group.index else 'Outside'
        for name in final_score_table.index
    ])

    # --- 6) 맨 끝 컬럼: 주 지표가 Rank 1 대비 몇 % 나쁜지 (양수일수록 나쁨) ---
    # 결함 때문에 밀린 모델은 주 지표가 Rank 1 보다 좋을 수 있어 음수가 나오기도 한다
    key = keys.loc[final_score_table.index, primary]
    diff = key - key.iloc[0]
    ref = abs(key.iloc[0])

    if ref == 0:
        # 기준값이 0 이면 비율을 계산할 수 없다
        final_score_table[f'{primary}_Gap'] = np.where(diff == 0, 0.0, np.nan)
    else:
        final_score_table[f'{primary}_Gap'] = (diff / ref).round(3)

    if verbose:
        print(f'\n▲ step4: 최종 Rank')
        for rank, name in zip(final_score_table['Rank'], final_score_table.index):
            tag = '[그룹 내]' if name in close_group.index else '[그룹 외 · 주 지표 순]'
            print(f'   {rank:>2}. {name:<14} {tag}')
        print('=' * 70 + '\n')

    # --- 7) 성능 비교 그래프 ---
    # RMSE·MAE 처럼 낮을수록 좋은 지표를 그대로 그리면 막대가 길수록 나쁜 모델이 되어
    # 그래프가 직관과 어긋난다. 역수를 취해 '클수록 좋은 값' 으로 방향을 통일한다.
    if plot:
        chart = final_score_table[['Group']].copy()     # 그래프용 임시표 (결과표에는 넣지 않는다)

        if metric_specs[primary]['better'] == 'higher':
            chart['Score'] = final_score_table[primary]             # 이미 높을수록 좋은 지표
            score_label = primary
        elif metric_specs[primary]['better'] == 'lower':
            chart['Score'] = 1 / final_score_table[primary]         # 역수로 방향을 뒤집는다
            score_label = f'1/{primary}'
        else:
            chart['Score'] = 1 / final_score_table[primary].abs()   # 부호를 떼고 크기만 역수로
            score_label = f'1/|{primary}|'

        my_plot.barplot(chart, y=chart.index, x='Score', hue='Group',
                        palette='tab10', width=width, height=len(chart) * 60 + 50,
                        title=f'{title}({score_label} 기준)', save_path=save_path)

    # --- 8) 결과표 리턴 ---
    return final_score_table


# --------------------------------------------------------
# 여러 회귀 모델의 지표를 한 번에 계산하고 4단계 전략으로 순위를 매긴 비교표 생성
# --------------------------------------------------------
def reg_compare_models(estimator, x_test, y_test, primary='RMSE',
                       aux=['MAE', 'R2'], verbose=True, plot=True,
                       title='모델 성능 비교', width=1280, height=640, save_path=None):
    """여러 회귀 모델의 지표를 계산하고 4단계 전략으로 'Rank' 를 매긴 비교표와 그래프를 만든다.

    Args:
        estimator (list | dict): 비교할 모델의 리스트 또는 {'이름': 모델} 딕셔너리.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'RMSE').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['MAE', 'R2']).
        verbose (bool): 판정 과정 출력 여부 (기본값: True).
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: '모델 성능 비교').
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).

    Returns:
        DataFrame: Rank 순 비교표
    """
    # --- 1) 지표 메타데이터 ---
    # better    : 'lower' | 'higher' | 'closer_to_zero'  — 어느 쪽이 좋은 값인가
    # flaw_type : 보조 지표의 결정적 결함 판정 방식
    #     rel_excess  :  값 > 1등 * (1 + threshold)   낮을수록 좋은 지표용
    #     abs_drop    :  값 < 1등 - threshold         높을수록 좋은 지표용
    #     abs_excess  : |값| > 1등 + threshold        0에 가까울수록 좋은 지표용
    # threshold : 1등 대비 결함으로 판정할 임계치
    metric_specs = {
        'R2':    {'better': 'higher',         'flaw_type': 'abs_drop',   'threshold': 0.05},
        'MAE':   {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MSE':   {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'RMSE':  {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'RMSLE': {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MAPE':  {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MPE':   {'better': 'closer_to_zero', 'flaw_type': 'abs_excess', 'threshold': 0.05},
    }

    # --- 2) 파라미터 검증 ---
    if isinstance(aux, str):
        aux = [aux]     # 보조 지표를 문자열 하나로 준 경우도 허용

    # `주 지표+보조 지표`가 위 metric_specs 에 있는 이름인지 확인한다
    for m in [primary] + aux:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    # --- 3) 모델 이름과 객체를 (이름, 모델) 튜플로 묶어 리스트로 만든다 ---
    # 딕셔너리면 키가 곧 이름이고, 리스트면 아래 루프에서 이름을 정한다
    if isinstance(estimator, dict):
        models = list(estimator.items())
    elif isinstance(estimator, list):
        models = [(None, m) for m in estimator]
    else:
        raise TypeError('estimator 는 모델의 리스트 또는 딕셔너리여야 합니다: '
                        f'{type(estimator).__name__}')

    # --- 4) 모델별 점수 결과표 만들기 ---
    score_tables = []
    for i, (name, model) in enumerate(models):
        # GridSearchCV·RandomizedSearchCV 등 탐색 객체면 최적 모델을 꺼내 쓴다.
        # 그대로 두면 모델명이 전부 'GridSearchCV' 로 찍혀 구분이 되지 않는다.
        best_model = getattr(model, 'best_estimator_', model)

        # 리스트로 받았으면 name_ 을 이름으로 쓴다. 탐색 객체는 내부 모델을 clone 해서
        # 재학습하므로 안쪽 name_ 은 사라진다. 탐색 객체 자신에 붙여둔 이름을 먼저 찾는다.
        if name is None:
            name = (getattr(model, 'name_', None)
                    or getattr(best_model, 'name_', None)
                    or f'Model {i + 1}')

        score_df = reg_score(best_model, x_test, y_test)    # 현재 모델의 성능평가표
        score_df.reset_index(inplace=True)   # 인덱스(=모델 클래스명)을 컬럼으로 내린다
        score_df.index = [name]              # 인덱스를 모델 이름으로 바꾼다 (리스트로 받았을 때 구분이 되도록)
        score_tables.append(score_df)        # 모델별 평가표를 score_tables 리스트에 저장

    final_score_table = concat(score_tables)    # score_tables의 개별 평가표를 하나로 병합
    final_score_table.index.name = 'name'       # 점수표의 인덱스에 이름 지정

    # --- 5) 4단계 전략으로 순위를 매긴 비교표와 그래프를 만든다 ---
    return _rank_score_table(final_score_table, metric_specs, primary, aux, verbose,
                             plot=plot, title=title, width=width, height=height,
                             save_path=save_path)


# --------------------------------------------------------
# 베이스라인 모델 11종을 한 번에 학습·저장하고 성능 순위표를 만든다
# --------------------------------------------------------
def reg_baseline(project_name, x_train, y_train, x_test, y_test, primary='RMSE', aux=['MAE', 'R2'], 
                 impute=False, outlier=False, pca=False,
                 plot=True, width=1280, height=640, save_path=None, verbose=True):
    """회귀 베이스라인 모델을 모두 학습·저장하고, 성능 순위표와 비교 그래프를 만든다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'RMSE').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['MAE', 'R2']).
        impute (bool): 결측치 대체 여부 (기본값: False).
        outlier (bool): 독립변수의 이상치를 경계값으로 대체(클리핑)할지 여부 (기본값: False).
        pca (bool): PCA 차원 축소 여부 (기본값: False).
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 학습 진행 상황 출력 여부 (기본값: True).

    Returns:
        DataFrame: Rank 순 비교표
    """
    # 부스팅 3종은 무겁고 별도 설치가 필요한 패키지라 모듈 로드 시가 아니라 함수 안에서 import 한다
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    # --- 1) 학습 결과물을 담을 작업 폴더 생성 ---
    workdir = Path(project_name) / f'baseline'
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')

    # 학습을 마친 모델을 {이름: 파이프라인} 으로 모아 마지막 비교표에 넘긴다
    models = {}

    # --- 2) 선형 계열 ---
    # 결측치·이상치·PCA 는 함수 인자(impute·outlier·pca)를 그대로 넘기고,
    # VIF·정규화·더미변수는 모델 계열에 맞춘 고정값을 적는다 (이하 모든 모델 동일)
    # 계수 해석과 규제의 전제가 되는 다중공선성을 VIF 로 제거하고, 더미 트랩도 함께 막는다.
    # LinearRegression 은 규제가 없어 스케일에 좌우되지 않으므로 정규화를 하지 않는다
    model = LinearRegression()
    model_name = model.__class__.__name__.lower().removesuffix('regression')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 1/11] : {model_name}')

    # Ridge·Lasso·ElasticNet 은 계수의 크기에 벌점을 매기므로 정규화가 반드시 필요하다
    model = Ridge(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 2/11] : {model_name}')

    model = Lasso(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 3/11] : {model_name}')

    model = ElasticNet(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 4/11] : {model_name}')

    # --- 3) 비선형 계열 ---
    # 거리·마진으로 학습하는 모델이라 변수의 단위가 다르면 큰 값의 변수가 거리를 독점한다
    model = KNeighborsRegressor(n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=True, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 5/11] : {model_name}')

    model = SVR()
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=True, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 6/11] : {model_name}')

    # --- 4) 트리 계열 ---
    # 분기 기준이 값의 대소 관계뿐이라 정규화가 필요 없고, 더미도 전부 남겨야
    # 각 범주가 독립적인 분기 후보가 된다 (drop_first 를 쓰지 않는 이유)
    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 7/11] : {model_name}')

    # --- 5) 앙상블 계열 ---
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 8/11] : {model_name}')

    model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 9/11] : {model_name}')

    model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [10/11] : {model_name}')

    # CatBoost 는 범주형을 자체 방식(Ordered Target Statistics)으로 처리하므로
    # 더미 인코딩을 끄고, 어떤 컬럼이 범주형인지만 fit 인자로 알려준다.
    # 범주형 판정은 fit_pipeline 의 명목형 자동 선택과 같은 기준(category·object)으로 한다
    model = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=True, scale=False, encode=False,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False,
        model__cat_features=list(x_train.select_dtypes(include=['category', 'object']).columns))

    if verbose:
        print(f'학습 완료 [11/11] : {model_name}')

    # --- 6) 모델간 성능 비교 + 성능 비교 그래프 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다.
    # 그래프는 순위를 매기는 쪽(_rank_score_table)이 함께 그린다
    score_table = reg_compare_models(models, x_test, y_test,
                                     primary=primary, aux=aux, verbose=False,
                                     plot=plot, width=width, height=height,
                                     save_path=save_path)

    return score_table




# --------------------------------------------------------
# 베이스라인 모델들을 모델별 하이퍼파라미터 그리드로 튜닝하고 성능 순위표를 만든다
# --------------------------------------------------------
def reg_tunes(project_name, models, x_train, y_train, x_test, y_test,
              primary='RMSE', aux=['MAE', 'R2'], practice=False,
              cv=5, n_jobs=-1, workdir="tuned", plot=True,
              width=1280, height=640, save_path=None, verbose=True):
    """넘겨받은 회귀 모델들을 GridSearchCV 로 튜닝·저장하고, 순위표와 비교 그래프를 만든다.

    그리드는 실무용(기본값)과 실습용 두 벌을 갖고 있다. 실무용은 탐색할 값을 넉넉히
    두어 조합이 많고, 실습용은 수업 시간에 탐색 과정을 직접 돌려볼 수 있도록 같은
    하이퍼파라미터를 유지한 채 값의 개수만 줄인 축소판이다.

    CatBoost 는 범주형을 자체 방식(Ordered Target Statistics)으로 처리하므로 더미
    인코딩 대신 범주형 컬럼명을 fit 인자로 넘긴다. 이 처리는 함수가 알아서 붙인다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        models (dict): 튜닝할 {'이름': 모델} 딕셔너리. 값은 fit_pipeline 이 만든
            파이프라인, 단독 모델, 이미 튜닝된 탐색 객체 중 무엇이든 된다.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 탐색 기준이자 순위를 가르는 주 지표 (기본값: 'RMSE').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['MAE', 'R2']).
        practice (bool): 실습용 축소 그리드 사용 여부. False 면 실무용 (기본값: False).
        cv (int): 교차검증 폴드 수 (기본값: 5).
        n_jobs (int): 탐색에 쓸 프로세스 수 (기본값: -1, 전체 코어).
        workdir (str): 모델별 튜닝 결과를 저장할 폴더명 (기본값: "tuned").
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 탐색 진행 상황·최적 조합 출력 여부 (기본값: True).

    Raises:
        ValueError: 잘못된 성능평가지표 이름이 전달된 경우
    """
    # 탐색 도구와 소요 시간 측정은 이 함수에서만 쓰므로 여기서 import 한다
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone
    from time import perf_counter

    # --- 1) 탐색 기준 지표를 사이킷런 scoring 문자열로 변환 ---
    # R2는 높을수록 좋은 지표, "neg_*"는 낮을수록 좋은 지표
    scoring_map = {
        'R2':    'r2',
        'MAE':   'neg_mean_absolute_error',
        'MSE':   'neg_mean_squared_error',
        'RMSE':  'neg_root_mean_squared_error',
        'RMSLE': 'neg_root_mean_squared_log_error',
        'MAPE':  'neg_mean_absolute_percentage_error',
    }

    # 주지표 이름이 탐색 기준으로 쓸 수 있는 지표인지 확인
    if primary not in scoring_map:
        raise ValueError(f'탐색 기준으로 쓸 수 없는 지표입니다: {primary} '
                         f'(가능한 값: {", ".join(scoring_map)})')

    scoring = scoring_map[primary]

    # --- 2) 모델별 탐색 범위(실무용) ---
    # {클래스이름: 튜닝 옵션} 형식의 딕셔너리로 정의한다.
    full_grids = {
        # LinearRegression 은 규제 항이 없어 성능을 조절할 하이퍼파라미터가 사실상 없다.
        # 아래 두 옵션이 가질 수 있는 값의 전부라, 이 그리드가 곧 전체 탐색 범위다
        'LinearRegression': {
            'model__fit_intercept': [True, False],   # 절편 사용 여부
            'model__positive': [True, False],        # 계수를 양수로 제한할지 여부
        },
        # alpha 가 클수록 계수를 0 쪽으로 강하게 눌러 분산을 줄인다
        'Ridge': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],       # 규제 강도(L2)
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr'],  # 최적화 알고리즘
        },
        # 기본값 alpha=1.0 은 타깃의 편차에 비해 강한 경우가 많아 계수가 전부 0 이 된다.
        # 작은 alpha 를 함께 넣어야 모델이 살아난다
        'Lasso': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],     # 규제 강도(L1)
            'model__max_iter': [1000, 5000, 10000],              # 좌표하강 반복 횟수
        },
        # l1_ratio 는 L1(Lasso)과 L2(Ridge)의 혼합 비율로,
        # 1 에 가까울수록 Lasso, 0 에 가까울수록 Ridge 처럼 동작한다
        'ElasticNet': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],     # 규제 강도
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.9],             # L1 규제의 비중
        },
        # 이웃 수가 적으면 과대적합, 많으면 과소적합 쪽으로 기운다
        'KNeighborsRegressor': {
            'model__n_neighbors': [3, 5, 10, 20, 30, 50],        # 참조할 이웃 수
            'model__weights': ['uniform', 'distance'],           # 거리에 따른 가중 방식
            'model__p': [1, 2],                                  # 거리 척도(1=맨해튼, 2=유클리드)
        },
        # 표본 수의 제곱에 비례해 학습 시간이 늘어나므로 이 그리드가 가장 오래 걸린다
        'SVR': {
            'model__C': [0.1, 1.0, 10.0],                        # 오차 허용에 대한 벌점
            'model__gamma': ['scale', 0.01, 0.1],                # RBF 커널의 영향 반경
            'model__epsilon': [0.05, 0.1, 0.2],                  # 오차 허용 범위
        },
        # 기본값(max_depth=None)은 잎이 하나가 될 때까지 분할해 훈련 R2 가 1.0 이 되는
        # 전형적인 과대적합 상태다. 깊이와 잎 크기로 가지치기를 건다
        'DecisionTreeRegressor': {
            'model__max_depth': [4, 6, 10, 14, 20, None],        # 트리의 최대 깊이
            'model__min_samples_leaf': [1, 5, 10, 20],           # 잎 노드의 최소 표본 수
        },
        # 트리를 n_estimators 개 만큼 학습하므로 조합 하나가 비싸다
        'RandomForestRegressor': {
            'model__n_estimators': [100, 300, 500],              # 숲을 이루는 트리 개수
            'model__max_depth': [10, 20, None],                  # 트리의 최대 깊이
            'model__min_samples_leaf': [1, 5],                   # 잎 노드의 최소 표본 수
        },
        # learning_rate 를 낮추면 n_estimators 를 늘려야 하므로 두 값은 짝지어 움직인다
        'XGBRegressor': {
            'model__n_estimators': [300, 600, 1000],             # 부스팅 라운드 수
            'model__max_depth': [3, 4, 6],                       # 트리 깊이
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
        # LightGBM 은 깊이 대신 잎 개수(num_leaves)로 복잡도를 조절한다
        'LGBMRegressor': {
            'model__n_estimators': [300, 600, 1000],             # 부스팅 라운드 수
            'model__num_leaves': [15, 31, 63],                   # 트리 하나가 가질 잎 개수
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
        # 조합 하나마다 CatBoost 를 cv 회 학습하므로 전체 학습 횟수가 가장 많다
        'CatBoostRegressor': {
            'model__iterations': [500, 1000, 2000],              # 부스팅 라운드 수
            'model__depth': [4, 6, 8, 10],                       # 트리 깊이
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
    }


    # --- 3) 모델별 탐색 범위(실습용) ---
    # 실무용과 같은 하이퍼파라미터를 쓰되, 값의 개수만 줄여 수업 시간 안에 탐색이 끝나도록 한 축소판
    practice_grids = {
        # 규제 항이 없어 이 두 옵션이 전부다. 줄일 것이 없으므로 실무용과 같다
        'LinearRegression': {
            'model__fit_intercept': [True, False],
            'model__positive': [True, False],
        },
        # 8개 조합. alpha 가 커질수록 계수가 눌리는 흐름을 보기 위해 자릿수를 남긴다
        'Ridge': {
            'model__alpha': [0.1, 1.0, 10.0, 100.0],
            'model__solver': ['auto', 'lsqr'],
        },
        # 8개 조합. alpha=1.0 은 계수를 전부 0 으로 만들어 버리므로 작은 값만 남긴다
        'Lasso': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1],
            'model__max_iter': [1000, 5000],
        },
        # 8개 조합. l1_ratio 는 Lasso 쪽·Ridge 쪽 양극단만 둬 차이를 드러낸다
        'ElasticNet': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1],
            'model__l1_ratio': [0.2, 0.8],
        },
        # 8개 조합. 거리 척도(p)는 빼고 이웃 수의 영향에 집중한다
        'KNeighborsRegressor': {
            'model__n_neighbors': [5, 10, 20, 30],
            'model__weights': ['uniform', 'distance'],
        },
        # 1개 조합. 표본 수의 제곱에 비례해 느려지는 모델이라 기본값 근처만 확인한다
        'SVR': {
            'model__C': [0.1, 1.0],
            'model__gamma': [0.01, 0.1],
            'model__epsilon': [0.1, 0.2],
        },
        # 8개 조합. max_depth=None 을 남겨 가지치기 전의 과대적합을 함께 본다
        'DecisionTreeRegressor': {
            'model__max_depth': [6, 10, 14, None],
            'model__min_samples_leaf': [1, 10],
        },
        # 4개 조합. 조합 하나가 트리 n_estimators 개라서 가장 크게 줄인다
        'RandomForestRegressor': {
            'model__n_estimators': [100, 300],
            'model__min_samples_leaf': [1, 5],
        },
        # 8개 조합. learning_rate 와 n_estimators 가 짝지어 움직이는 것만 확인한다
        'XGBRegressor': {
            'model__n_estimators': [300, 600],
            'model__max_depth': [4, 6],
            'model__learning_rate': [0.05, 0.1],
        },
        # 8개 조합. LightGBM 은 학습이 빨라 조합 수를 유지해도 부담이 적다
        'LGBMRegressor': {
            'model__n_estimators': [300, 600],
            'model__num_leaves': [31, 63],
            'model__learning_rate': [0.05, 0.1],
        },
        # 18개 조합. 값을 하나씩만 덜어내 세 하이퍼파라미터의 상호작용은 남긴다
        'CatBoostRegressor': {
            'model__iterations': [500, 1000],
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.03, 0.05, 0.1],
        },
    }

    # --- 4) 튜닝 조합 채택, 작업 폴더 생성, 결과 조합 저장용 자료구조 정의 ---
    # 하이퍼파라미터 조합 채택
    #  --> 실습용은 명시적으로 요청했을 때만 쓰고, 기본은 실무용 범위로 탐색한다
    param_grids = practice_grids if practice else full_grids

    # 작업폴더 구성
    workdir = Path(project_name) / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')
        print(f'탐색 범위: {"실습용(축소)" if practice else "실무용"}')

    # CatBoost 에 넘길 범주형 컬럼명. 판정 기준은 fit_pipeline 의 명목형 자동 선택과
    # 같은 category·object 다 — my_qtcheck 는 category 만 보므로 CSV 에서 흔한
    # object 컬럼을 놓쳐 CatBoost 가 학습에 실패한다
    cat_features = list(x_train.select_dtypes(include=['category', 'object']).columns)

    # 탐색을 마친 객체를 {이름: GridSearchCV} 형식으로 모아 마지막 비교표에 넘긴다
    tuned = {}
    skipped = []
    total = len(models)

    # --- 5) 모델별 탐색 ---
    for i, (name, model) in enumerate(models.items(), start=1):
        # --- 5-1) 탐색 대상 모델 확인 ---
        # 이미 튜닝된 객체를 다시 넘겨도 되도록 탐색 객체면 최적 모델을 꺼낸다.
        # 그대로 두면 탐색 객체를 또 감싸 param_grid 의 접두어가 어긋난다
        estimator = getattr(model, 'best_estimator_', model)

        # 파이프라인이면 안쪽 모델을, 단독 모델이면 자기 자신을 꺼낸다
        base_model, _, _, _ = _unwrap_estimator(estimator)
        classname = type(base_model).__name__

        # 모델에 맞는 튜닝 조합을 꺼낸다. 정의되지 않은 모델이면 건너뛴다
        param_grid = param_grids.get(classname)

        if param_grid is None:
            skipped.append(name)
            if verbose:
                print(f'탐색 범위가 없어 건너뜀 [{i:2d}/{total:2d}] : {name}')
            continue

        # CatBoost 는 더미 인코딩 없이 원본 범주형 컬럼을 그대로 받으므로,
        # 어떤 컬럼이 범주형인지를 fit 인자로 알려줘야 한다
        if classname == 'CatBoostRegressor':
            fit_params = {'model__cat_features': cat_features}
        else:
            fit_params = {}

        # --- 5-2) 병렬 처리 설정 ---
        # 병렬 처리는 GridSearchCV 한 곳에서만 한다. 안쪽 모델(RF·XGB·LGBM·KNN·CatBoost)까지
        # 코어를 전부 쓰면 프로세스 수 × 스레드 수가 코어 수를 크게 넘어 오히려 느려진다.
        # 넘겨받은 원본은 그대로 두고 복제본에만 적용한다 (저장되는 튜닝 모델은 단일 스레드)
        if n_jobs not in (None, 1):
            # 병렬을 켜 둔(값이 None·1 이 아닌) 단계만 바꾼다. 값이 없는 곳까지 건드리면
            # n_jobs 가 폐기된 모델(sklearn 1.8 의 LogisticRegression)에서 경고가 난다
            inner_jobs = {k: 1 for k, v in estimator.get_params().items()
                          if k.endswith('n_jobs') and v not in (None, 1)}

            # CatBoost 는 스레드 수를 n_jobs 가 아니라 thread_count 로 받는다
            if classname == 'CatBoostRegressor':
                inner_jobs['model__thread_count'] = 1

            estimator = clone(estimator).set_params(**inner_jobs)

        # --- 5-3) 파라미터 탐색 수행 ---
        started = perf_counter()    # 시간 측정 시작
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid,
                          cv=cv, scoring=scoring, n_jobs=n_jobs)
        gs.fit(x_train, y_train, **fit_params)

        # 탐색 객체는 내부 모델을 clone 해서 재학습하므로 안쪽 name_ 이 사라진다.
        # 비교표에서 이름이 전부 'GridSearchCV' 가 되지 않도록 자신에게 이름을 붙인다
        tuned_name = f'{name}_tuned'
        gs.name_ = tuned_name
        save_model(gs, workdir / f'{tuned_name}.pkl')
        tuned[tuned_name] = gs

        if verbose:
            # 탐색 점수는 neg_* 부호가 뒤집혀 있어 되돌려야 지표 단위와 맞는다
            best = gs.best_score_ if scoring.startswith('r2') else -gs.best_score_
            print(f'탐색 완료 [{i:2d}/{total:2d}] : {name} '
                  f'({perf_counter() - started:.1f}초, CV {primary}={best:.4f})')
            print(f'    최적 조합: {gs.best_params_}')

    # 전부 건너뛰었다면 비교할 대상이 없다
    if not tuned:
        print('튜닝된 모델이 없습니다. models 에 그리드가 정의된 회귀 모델을 넣어 주세요.')
        return

    # --- 6) 모델간 성능 비교 + 튜닝 결과 시각화 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다.
    # 그래프는 순위를 매기는 쪽(_rank_score_table)이 함께 그린다
    score_table = reg_compare_models(tuned, x_test, y_test,
                                     primary=primary, aux=aux, verbose=False,
                                     plot=plot, title='하이퍼파라미터 튜닝 결과 성능 비교',
                                     width=width, height=height, save_path=save_path)

    return score_table





# --------------------------------------------------------
# 모델 성능 평가
# --------------------------------------------------------
def cls_score(estimator, x_test, y_test, average='auto'):
    """학습된 분류 모델의 성능 지표 7종을 계산해 1행짜리 표로 반환한다.

        - 임계값 0.5 로 자른 예측 라벨 기준: Accuracy · Precision · Recall · F1
        - 혼동행렬의 오즈비 기준: DOR (진단오즈비 = TP·TN / FP·FN, 1 = 무작위, 클수록 좋다)
        - 점수의 순위 기준(임계값 무관): ROC_AUC · PR_AUC

    Args:
        estimator: 학습이 완료된 사이킷런 분류 모델 또는 파이프라인. GridSearchCV 같은
            하이퍼파라미터 탐색 객체를 주면 내부의 best_estimator_ 로 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
            'auto' 는 이진이면 'binary'(양성 클래스만), 다중분류면 'macro'.
            'micro'·'weighted' 도 쓸 수 있다.

    Returns:
        DataFrame: 모델 클래스명을 인덱스로 하는 지표 1행. 컬럼=[Accuracy, Precision,
            Recall, F1, ROC_AUC, PR_AUC, DOR].
    """
    # --- 1) 평가할 모델을 확정하고 예측을 수행한다 ---
    # 탐색 객체면 최적 모델을, 파이프라인이면 마지막 단계에서 모델명을 꺼낸다.
    # 예측 자체는 전처리가 붙은 파이프라인 전체(base_est)로 해야 한다.
    model, _, base_est, _ = _unwrap_estimator(estimator)
    classname = type(model).__name__

    y_pred = base_est.predict(x_test)
    y_true = np.asarray(y_test).ravel()     # DataFrame·Series·ndarray 를 1차원 배열로 통일

    # 클래스 목록 — 모델이 학습한 순서를 그대로 따른다 (마지막이 양성 클래스)
    # (수정 전) classes_ 가 없을 때의 폴백. 학습된 분류기는 모두 classes_ 를 가지므로 불필요 (2026-09-17)
    # classes = np.asarray(getattr(model, 'classes_', np.unique(y_true)))
    classes = np.asarray(model.classes_)
    binary = len(classes) == 2

    # --- 2) 라벨 기준 지표의 평균 방식 ---
    # 이진이면 양성 클래스(마지막) 하나만, 다중분류면 클래스 평균으로 계산한다
    if binary and average in ('auto', 'binary'):
        kwargs = {'average': 'binary', 'pos_label': classes[-1]}
    else:
        kwargs = {'average': 'macro' if average in ('auto', 'binary') else average}

    # --- 3) 진단오즈비(DOR) ---
    # 클래스별 일대다(OvR) 혼동행렬 [[TN, FP], [FN, TP]] 에서 (TP·TN)/(FP·FN) 을 구한다.
    # 이진이면 양성 클래스 하나만 쓰고, 다중분류는 average 방식대로 합치거나 평균한다.
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=classes).astype(float)

    if kwargs['average'] == 'binary':
        mcm = mcm[-1:]                              # 양성 클래스 하나만
    elif average == 'micro':
        mcm = mcm.sum(axis=0, keepdims=True)        # 네 칸을 클래스 합산으로 통합

    mcm[(mcm == 0).any(axis=(1, 2))] += 0.5         # 0 인 칸이 있으면 네 칸에 0.5 를 더한다 (Haldane 보정)
    tn, fp, fn, tp = mcm[:, 0, 0], mcm[:, 0, 1], mcm[:, 1, 0], mcm[:, 1, 1]
    dor = np.average(tp * tn / (fp * fn), weights=tp + fn if average == 'weighted' else None)

    # --- 4) 순위 기준 지표 — 확률이 없으면 결정함수 마진을 쓴다 ---
    # (수정 전) 둘 다 없으면 NaN 으로 두는 폴백. 사이킷런 분류기는 둘 중 하나는 있어 도달하지 않음 (2026-09-17)
    # if hasattr(base_est, 'predict_proba'):
    #     y_score = np.asarray(base_est.predict_proba(x_test))
    # elif hasattr(base_est, 'decision_function'):
    #     y_score = np.asarray(base_est.decision_function(x_test))
    # else:
    #     y_score = None
    #
    # if y_score is None:
    #     roc_auc = pr_auc = np.nan
    # elif binary:
    if hasattr(base_est, 'predict_proba'):
        y_score = np.asarray(base_est.predict_proba(x_test))
    else:
        y_score = np.asarray(base_est.decision_function(x_test))

    if binary:
        # 이진 — 양성 클래스 점수 한 줄만 있으면 된다
        pos = y_score[:, -1] if y_score.ndim == 2 else y_score
        y_bin = (y_true == classes[-1]).astype(int)
        roc_auc = roc_auc_score(y_bin, pos)
        pr_auc = average_precision_score(y_bin, pos)
    else:
        # 다중분류 — 클래스마다 '이 클래스인가 아닌가' 의 0/1 정답으로 일대다 곡선을 그려 평균한다
        y_bin = label_binarize(y_true, classes=classes)
        # (수정 전) 평균 방식 재계산. 2) 에서 만든 kwargs['average'] 와 같은 값이라 재사용 (2026-09-17)
        # avg = 'macro' if average in ('auto', 'binary') else average
        # roc_auc = roc_auc_score(y_bin, y_score, average=avg)
        # pr_auc = average_precision_score(y_bin, y_score, average=avg)
        roc_auc = roc_auc_score(y_bin, y_score, average=kwargs['average'])
        pr_auc = average_precision_score(y_bin, y_score, average=kwargs['average'])

    # --- 5) 지표를 모델명 1행짜리 표로 정리해 반환 ---
    scores = {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0, **kwargs),
        'Recall':    recall_score(y_true, y_pred, zero_division=0, **kwargs),
        'F1':        f1_score(y_true, y_pred, zero_division=0, **kwargs),
        'ROC_AUC':   roc_auc,
        'PR_AUC':    pr_auc,
        'DOR':       dor,
    }

    score_df = DataFrame(scores, index=[classname])
    score_df.index.name = 'Model'

    return score_df


# --------------------------------------------------------
# 모델 성능 비교
# --------------------------------------------------------
def cls_compare_models(estimator, x_test, y_test, primary='F1',
                       aux=['ROC_AUC', 'Accuracy'], average='auto', verbose=True,
                       plot=True, title='모델 성능 비교', width=1280, height=640, save_path=None):
    """여러 분류 모델의 지표를 계산하고 4단계 전략으로 'Rank' 를 매긴 비교표와 그래프를 만든다.

    Args:
        estimator (list | dict): 비교할 모델의 리스트 또는 {'이름': 모델} 딕셔너리.
            리스트면 모델의 `name_` 속성을, 없으면 `Model 1` … 을 이름으로 쓴다.
            GridSearchCV 같은 탐색 객체는 내부의 best_estimator_ 를 꺼내 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'F1').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        verbose (bool): 판정 과정 출력 여부 (기본값: True).
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: '모델 성능 비교').
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).

    primary·aux 에는 cls_score 가 계산하는 Accuracy·Precision·Recall·F1·ROC_AUC·
    PR_AUC·DOR 를 쓴다.

    Returns:
        DataFrame: Rank 순 비교표. 맨 앞에 `Rank`·`Group`(Contender=근소 격차 그룹 /
            Outside=그룹 외부), 맨 끝에 `{primary}_Gap`(1등 대비 격차, 양수일수록 나쁨) 컬럼.

    Raises:
        TypeError: estimator 가 리스트도 딕셔너리도 아닌 경우.
        ValueError: primary·aux 에 계산되지 않는 지표명을 준 경우.
    """
    # --- 1) 지표 메타데이터 ---
    # better    : 'higher'  — 분류 지표는 모두 높을수록 좋다
    # flaw_type : 보조 지표의 결정적 결함 판정 방식
    #     abs_drop  :  값 < 1등 - threshold          상한이 1 로 정해진 지표용
    #     rel_drop  :  값 < 1등 * (1 - threshold)    상한이 없는 지표용(DOR)
    # threshold : 1등 대비 결함으로 판정할 임계치
    metric_specs = {
        'Accuracy':  {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'Precision': {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'Recall':    {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'F1':        {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'ROC_AUC':   {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'PR_AUC':    {'better': 'higher', 'flaw_type': 'abs_drop', 'threshold': 0.05},
        'DOR':       {'better': 'higher', 'flaw_type': 'rel_drop', 'threshold': 0.10},
    }

    # --- 2) 파라미터 검증 ---
    if isinstance(aux, str):
        aux = [aux]     # 보조 지표를 문자열 하나로 준 경우도 허용

    # `주 지표+보조 지표`가 위 metric_specs 에 있는 이름인지 확인한다
    for m in [primary] + aux:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    # --- 3) 모델별 점수 계산 ---
    # 딕셔너리면 키가 곧 이름이고, 리스트면 아래 루프에서 이름을 정한다
    if isinstance(estimator, dict):
        models = list(estimator.items())
    elif isinstance(estimator, list):
        models = [(None, m) for m in estimator]
    else:
        raise TypeError('estimator 는 모델의 리스트 또는 딕셔너리여야 합니다: '
                        f'{type(estimator).__name__}')

    # --- 4) 모델별 점수 결과표 만들기 ---
    score_tables = []
    for i, (name, model) in enumerate(models):
        # 리스트로 받았으면 name_ 을 이름으로 쓴다. 탐색 객체는 내부 모델을 clone 해서
        # 재학습하므로 안쪽 name_ 은 사라지고, 탐색 객체 자신에 붙여둔 이름만 남는다.
        # 탐색 객체를 푸는 일은 cls_score 가 내부에서 하므로 여기서는 그대로 넘긴다.
        # (수정 전) 탐색 객체 풀기와 안쪽 name_ 조회. cls_score 가 내부에서 풀고, 안쪽 name_ 은 늘 None 이라 정리 (2026-09-17)
        # best_model = getattr(model, 'best_estimator_', model)
        #
        # if name is None:
        #     name = (getattr(model, 'name_', None)
        #             or getattr(best_model, 'name_', None)
        #             or f'Model {i + 1}')
        #
        # score_df = cls_score(best_model, x_test, y_test, average=average)
        if name is None:
            name = getattr(model, 'name_', None) or f'Model {i + 1}'

        score_df = cls_score(model, x_test, y_test, average=average)
        score_df.reset_index(inplace=True)   # 모델 클래스명을 'Model' 컬럼으로 내린다
        score_df.index = [name]
        score_tables.append(score_df)

    final_score_table = concat(score_tables)
    final_score_table.index.name = 'name'

    return _rank_score_table(final_score_table, metric_specs, primary, aux, verbose,
                             plot=plot, title=title, width=width, height=height,
                             save_path=save_path)


# --------------------------------------------------------
# 베이스라인 일괄 적합
# --------------------------------------------------------
def cls_baseline(project_name, x_train, y_train, x_test, y_test,
                 primary='F1', aux=['ROC_AUC', 'Accuracy'], average='auto', plot=True,
                 width=1280, height=640, save_path=None, verbose=True,
                 impute=False, outlier=False, pca=False, vif=True):
    """분류 베이스라인 모델을 모두 학습·저장하고, 성능 순위표와 비교 그래프를 만든다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'F1').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 학습 진행 상황 출력 여부 (기본값: True).
        impute (bool): 결측치 대체 여부 (기본값: False).
        outlier (bool): 독립변수의 이상치를 경계값으로 대체(클리핑)할지 여부 (기본값: False).
        pca (bool): PCA 차원 축소 여부 (기본값: False).
        vif (bool): VIF 기준 다중공선성 제거 여부 (기본값: True). 텍스트 DTM 처럼 컬럼이 매우 많을 때만 끈다.

    결측치·이상치·PCA 는 데이터 상황에 따라 분석가가 정하므로 인자로 받아 모든 모델에 같이 적용한다.
    정규화·더미변수 인코딩은 모델 계열의 특성에 맞춰야 하므로 함수 안에 고정한다 (회귀 baseline 과 같은 규칙).

    마지막에 순위표를 화면에 출력한다. 별도로 반환하는 값은 없다.
    """
    # 부스팅 3종은 무겁고 별도 설치가 필요한 패키지라 모듈 로드 시가 아니라 함수 안에서 import 한다
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    # --- 1) 학습 결과물을 담을 작업 폴더 생성 ---
    workdir = Path(project_name) / f'baseline'
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')

    # --- 2) 종속변수를 XGBoost 가 받는 형태(0부터 시작하는 연속된 정수)로 맞춘다 ---
    # 라벨이 문자열('Yes'/'No')이거나 띄엄띄엄한 정수(1/2)면 XGBoost 만 학습에 실패한다.
    # 9종을 같은 조건에서 비교해야 하므로 훈련·검증 모두에 같은 인코딩을 적용한다
    # (수정 전) DataFrame 분기. np.asarray 가 DataFrame 도 받으므로 cls_tunes 와 같은 한 줄로 통일 (2026-09-17)
    # y_labels = np.unique(y_train.values.ravel() if isinstance(y_train, DataFrame)
    #                      else np.asarray(y_train).ravel())
    y_labels = np.unique(np.asarray(y_train).ravel())

    if not (np.issubdtype(y_labels.dtype, np.integer)
            and np.array_equal(y_labels, np.arange(len(y_labels)))):
        encoder = LabelEncoder().fit(y_labels)
        y_train = Series(encoder.transform(np.asarray(y_train).ravel()),
                         index=getattr(y_train, 'index', None), name='y')
        y_test = Series(encoder.transform(np.asarray(y_test).ravel()),
                        index=getattr(y_test, 'index', None), name='y')

        if verbose:
            print('   ▷ 종속변수를 정수로 인코딩했습니다 (XGBoost 요구사항): '
                  f'{ {str(c): i for i, c in enumerate(encoder.classes_)} }')

    # 학습을 마친 모델을 {이름: 파이프라인} 으로 모아 마지막 비교표에 넘긴다
    models = {}

    # --- 3) 선형 계열 ---
    # 결측치·이상치·PCA·VIF 는 함수 인자를 그대로 넘기고,
    # 정규화·더미변수는 모델 계열에 맞춘 고정값을 적는다 (이하 모든 모델 동일)
    # 계수 해석과 규제의 전제가 되는 다중공선성을 VIF 로 제거하고, 더미 트랩도 함께 막는다.
    # 두 모델 모두 계수의 크기에 벌점을 매기므로 정규화가 반드시 필요하다
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model_name = model.__class__.__name__.lower().removesuffix('regression')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [1/9] : {model_name}')

    # RidgeClassifier 는 확률을 내지 못하지만 ROC_AUC·PR_AUC 는 결정함수 마진으로 계산된다
    model = RidgeClassifier(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [2/9] : {model_name}')

    # --- 4) 비선형 계열 ---
    # 거리·마진으로 학습하는 모델이라 변수의 단위가 다르면 큰 값의 변수가 거리를 독점한다
    model = KNeighborsClassifier(n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=True, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [3/9] : {model_name}')

    model = SVC(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=True, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [4/9] : {model_name}')

    # --- 5) 트리 계열 ---
    # 분기 기준이 값의 대소 관계뿐이라 정규화가 필요 없고, 더미도 전부 남겨야
    # 각 범주가 독립적인 분기 후보가 된다 (drop_first 를 쓰지 않는 이유)
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [5/9] : {model_name}')

    # --- 6) 앙상블 계열 ---
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [6/9] : {model_name}')

    model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [7/9] : {model_name}')

    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [8/9] : {model_name}')

    # CatBoost 는 범주형을 자체 방식(Ordered Target Statistics)으로 처리하므로
    # 더미 인코딩을 끄고, 어떤 컬럼이 범주형인지만 fit 인자로 알려준다.
    # 범주형 판정은 fit_pipeline 의 명목형 자동 선택과 같은 기준(category·object)으로 한다 —
    # my_qtcheck 는 category 만 보므로 CSV 에서 흔한 object 컬럼을 놓쳐 CatBoost 가 학습에 실패한다
    model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        impute=impute, outlier=outlier, pca=pca,
        vif=vif, scale=False, encode=False,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False,
        model__cat_features=list(x_train.select_dtypes(include=['category', 'object']).columns))

    if verbose:
        print(f'학습 완료 [9/9] : {model_name}')

    # --- 7) 모델간 성능 비교 + 성능 비교 그래프 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다.
    # 그래프는 순위를 매기는 쪽(_rank_score_table)이 함께 그린다
    score_table = cls_compare_models(models, x_test, y_test, primary=primary,
                                     aux=aux, average=average, verbose=False,
                                     plot=plot, width=width, height=height,
                                     save_path=save_path)

    # --- 8) 최종 순위표 출력 ---
    # 순위표가 이 함수의 결과물이므로 화면에 직접 출력한다.
    # 학습된 모델은 이미 pkl 로 저장했으니, 다시 쓸 때는 workdir 에서 load_model 로 불러온다
    print(f'\n모델 {len(models)}개 학습·저장 완료 → {workdir}')
    display(score_table)




# --------------------------------------------------------
# 베이스라인 모델들을 모델별 하이퍼파라미터 그리드로 튜닝하고 성능 순위표를 만든다
# --------------------------------------------------------
def cls_tunes(project_name, models, x_train, y_train, x_test, y_test,
              primary='F1', aux=['ROC_AUC', 'Accuracy'], average='auto', practice=False,
              cv=5, n_jobs=-1, workdir="tuned", plot=True,
              width=1280, height=640, save_path=None, verbose=True):
    """넘겨받은 분류 모델들을 GridSearchCV 로 튜닝·저장하고, 순위표와 비교 그래프를 만든다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        models (dict): 튜닝할 {'이름': 모델} 딕셔너리. 값은 fit_pipeline 이 만든
            파이프라인, 단독 모델, 이미 튜닝된 탐색 객체 중 무엇이든 된다.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 탐색 기준이자 순위를 가르는 주 지표 (기본값: 'F1').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        practice (bool): 실습용 축소 그리드 사용 여부. False 면 실무용 (기본값: False).
        cv (int): 교차검증 폴드 수 (기본값: 5). 분류는 층화(Stratified) 분할이 자동 적용된다.
        n_jobs (int): 탐색에 쓸 프로세스 수 (기본값: -1, 전체 코어).
        workdir (str): 모델별 튜닝 결과를 저장할 폴더명 (기본값: "tuned").
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 탐색 진행 상황·최적 조합 출력 여부 (기본값: True).

    Raises:
        ValueError: 잘못된 성능평가지표 이름이 전달된 경우
    """
    # 탐색 도구와 소요 시간 측정은 이 함수에서만 쓰므로 여기서 import 한다
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone
    from time import perf_counter

    # --- 1) 탐색 기준 지표를 사이킷런 scoring 문자열로 변환 ---
    # 분류 지표는 모두 높을수록 좋은 지표라 회귀의 "neg_*" 처럼 부호가 뒤집히지 않는다.
    # 이진이면 양성 클래스 기준 scorer 를, 다중분류면 클래스 평균 방식(average)이 붙은 scorer 를 쓴다.
    # DOR 은 사이킷런 scorer 가 없어 탐색 기준으로는 쓸 수 없다 (순위표의 보조 지표로는 가능)
    y_labels = np.unique(np.asarray(y_train).ravel())
    binary = len(y_labels) == 2

    if binary and average in ('auto', 'binary'):
        scoring_map = {
            'Accuracy':  'accuracy',
            'Precision': 'precision',
            'Recall':    'recall',
            'F1':        'f1',
            'ROC_AUC':   'roc_auc',
            'PR_AUC':    'average_precision',
        }
    else:
        # 다중분류의 ROC_AUC 는 일대다(OvR) 평균만 지원한다 (micro 는 사이킷런에 없어 macro 로 계산)
        avg = 'macro' if average in ('auto', 'binary') else average
        scoring_map = {
            'Accuracy':  'accuracy',
            'Precision': f'precision_{avg}',
            'Recall':    f'recall_{avg}',
            'F1':        f'f1_{avg}',
            'ROC_AUC':   'roc_auc_ovr_weighted' if avg == 'weighted' else 'roc_auc_ovr',
        }

    if primary not in scoring_map:
        raise ValueError(f'탐색 기준으로 쓸 수 없는 지표입니다: {primary} '
                         f'(가능한 값: {", ".join(scoring_map)})')

    scoring = scoring_map[primary]

    # --- 2) 모델별 탐색 범위(실무용) ---
    # {클래스이름: 튜닝 옵션} 형식의 딕셔너리로 정의한다.
    full_grids = {
        # C 는 규제 강도의 역수라 작을수록 규제가 강하다 (Ridge 의 alpha 와 방향이 반대).
        # class_weight='balanced' 는 소수 클래스의 오차에 더 큰 벌점을 매겨 Recall 을 끌어올린다
        'LogisticRegression': {
            'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],           # 규제 강도의 역수
            'model__solver': ['lbfgs', 'liblinear'],             # 최적화 알고리즘
            'model__class_weight': [None, 'balanced'],           # 클래스 불균형 보정
        },
        # alpha 가 클수록 계수를 0 쪽으로 강하게 눌러 분산을 줄인다
        'RidgeClassifier': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],       # 규제 강도(L2)
            'model__solver': ['auto', 'lsqr'],                   # 최적화 알고리즘
            'model__class_weight': [None, 'balanced'],           # 클래스 불균형 보정
        },
        # 이웃 수가 적으면 과대적합, 많으면 과소적합 쪽으로 기운다
        'KNeighborsClassifier': {
            'model__n_neighbors': [3, 5, 10, 20, 30, 50],        # 참조할 이웃 수
            'model__weights': ['uniform', 'distance'],           # 거리에 따른 가중 방식
            'model__p': [1, 2],                                  # 거리 척도(1=맨해튼, 2=유클리드)
        },
        # 표본 수의 제곱에 비례해 학습 시간이 늘어나고, probability=True 면 확률을 얻기 위해
        # 내부에서 교차검증을 한 번 더 돌리므로 이 그리드가 가장 오래 걸린다
        'SVC': {
            'model__C': [0.1, 1.0, 10.0],                        # 오분류에 대한 벌점
            'model__gamma': ['scale', 0.01, 0.1],                # RBF 커널의 영향 반경
            'model__class_weight': [None, 'balanced'],           # 클래스 불균형 보정
        },
        # 기본값(max_depth=None)은 잎이 한 클래스만 남을 때까지 분할해 훈련 Accuracy 가 1.0 이 되는
        # 전형적인 과대적합 상태다. 깊이와 잎 크기로 가지치기를 건다
        'DecisionTreeClassifier': {
            'model__criterion': ['gini', 'entropy'],             # 불순도 측정 방식
            'model__max_depth': [4, 6, 10, 14, 20, None],        # 트리의 최대 깊이
            'model__min_samples_leaf': [1, 5, 10, 20],           # 잎 노드의 최소 표본 수
        },
        # 트리를 n_estimators 개 만큼 학습하므로 조합 하나가 비싸다
        'RandomForestClassifier': {
            'model__n_estimators': [100, 300, 500],              # 숲을 이루는 트리 개수
            'model__max_depth': [10, 20, None],                  # 트리의 최대 깊이
            'model__min_samples_leaf': [1, 5],                   # 잎 노드의 최소 표본 수
        },
        # learning_rate 를 낮추면 n_estimators 를 늘려야 하므로 두 값은 짝지어 움직인다
        'XGBClassifier': {
            'model__n_estimators': [300, 600, 1000],             # 부스팅 라운드 수
            'model__max_depth': [3, 4, 6],                       # 트리 깊이
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
        # LightGBM 은 깊이 대신 잎 개수(num_leaves)로 복잡도를 조절한다
        'LGBMClassifier': {
            'model__n_estimators': [300, 600, 1000],             # 부스팅 라운드 수
            'model__num_leaves': [15, 31, 63],                   # 트리 하나가 가질 잎 개수
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
        # 조합 하나마다 CatBoost 를 cv 회 학습하므로 전체 학습 횟수가 가장 많다
        'CatBoostClassifier': {
            'model__iterations': [500, 1000, 2000],              # 부스팅 라운드 수
            'model__depth': [4, 6, 8, 10],                       # 트리 깊이
            'model__learning_rate': [0.03, 0.05, 0.1],           # 각 트리의 반영 비율
        },
    }


    # --- 3) 모델별 탐색 범위(실습용) ---
    # 실무용과 같은 하이퍼파라미터를 쓰되, 값의 개수만 줄여 수업 시간 안에 탐색이 끝나도록 한 축소판
    practice_grids = {
        # 8개 조합. 최적화 알고리즘은 빼고 규제 강도와 클래스 불균형 보정의 영향에 집중한다
        'LogisticRegression': {
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced'],
        },
        # 8개 조합. alpha 가 커질수록 계수가 눌리는 흐름을 보기 위해 자릿수를 남긴다
        'RidgeClassifier': {
            'model__alpha': [0.1, 1.0, 10.0, 100.0],
            'model__class_weight': [None, 'balanced'],
        },
        # 8개 조합. 거리 척도(p)는 빼고 이웃 수의 영향에 집중한다
        'KNeighborsClassifier': {
            'model__n_neighbors': [5, 10, 20, 30],
            'model__weights': ['uniform', 'distance'],
        },
        # 8개 조합. 가장 느린 모델이라 값을 두 개씩만 남긴다
        'SVC': {
            'model__C': [0.1, 1.0],
            'model__gamma': [0.01, 0.1],
            'model__class_weight': [None, 'balanced'],
        },
        # 8개 조합. 불순도 방식은 빼고 max_depth=None 을 남겨 가지치기 전의 과대적합을 함께 본다
        'DecisionTreeClassifier': {
            'model__max_depth': [6, 10, 14, None],
            'model__min_samples_leaf': [1, 10],
        },
        # 4개 조합. 조합 하나가 트리 n_estimators 개라서 가장 크게 줄인다
        'RandomForestClassifier': {
            'model__n_estimators': [100, 300],
            'model__min_samples_leaf': [1, 5],
        },
        # 8개 조합. learning_rate 와 n_estimators 가 짝지어 움직이는 것만 확인한다
        'XGBClassifier': {
            'model__n_estimators': [300, 600],
            'model__max_depth': [4, 6],
            'model__learning_rate': [0.05, 0.1],
        },
        # 8개 조합. LightGBM 은 학습이 빨라 조합 수를 유지해도 부담이 적다
        'LGBMClassifier': {
            'model__n_estimators': [300, 600],
            'model__num_leaves': [31, 63],
            'model__learning_rate': [0.05, 0.1],
        },
        # 18개 조합. 값을 하나씩만 덜어내 세 하이퍼파라미터의 상호작용은 남긴다
        'CatBoostClassifier': {
            'model__iterations': [500, 1000],
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.03, 0.05, 0.1],
        },
    }

    # --- 4) 튜닝 조합 채택, 튜닝 결과물을 담을 작업 폴더 생성, 결과 조합 결과를 저장할 자료구조 정의 ---
    # 하이퍼파라미터 조합 채택
    #  --> 실습용은 명시적으로 요청했을 때만 쓰고, 기본은 실무용 범위로 탐색한다
    param_grids = practice_grids if practice else full_grids

    # 작업폴더 구성
    workdir = Path(project_name) / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')
        print(f'탐색 범위: {"실습용(축소)" if practice else "실무용"}')

    # --- 5) 종속변수를 XGBoost 가 받는 형태(0부터 시작하는 연속된 정수)로 맞춘다 ---
    # 라벨이 문자열('Yes'/'No')이거나 띄엄띄엄한 정수(1/2)면 XGBoost 만 학습에 실패한다.
    # 모든 모델을 같은 조건에서 비교해야 하므로 훈련·검증 모두에 같은 인코딩을 적용한다
    if not (np.issubdtype(y_labels.dtype, np.integer)
            and np.array_equal(y_labels, np.arange(len(y_labels)))):
        encoder = LabelEncoder().fit(y_labels)
        y_train = Series(encoder.transform(np.asarray(y_train).ravel()),
                         index=getattr(y_train, 'index', None), name='y')
        y_test = Series(encoder.transform(np.asarray(y_test).ravel()),
                        index=getattr(y_test, 'index', None), name='y')

        if verbose:
            print('   ▷ 종속변수를 정수로 인코딩했습니다 (XGBoost 요구사항): '
                  f'{ {str(c): i for i, c in enumerate(encoder.classes_)} }')

    # CatBoost 에 넘길 범주형 컬럼명. 판정 기준은 fit_pipeline 의 명목형 자동 선택과
    # 같은 category·object 다 — my_qtcheck 는 category 만 보므로 CSV 에서 흔한
    # object 컬럼을 놓쳐 CatBoost 가 학습에 실패한다
    cat_features = list(x_train.select_dtypes(include=['category', 'object']).columns)

    # 탐색을 마친 객체를 {이름: GridSearchCV} 형식으로 모아 마지막 비교표에 넘긴다
    tuned = {}
    skipped = []
    total = len(models)

    # --- 6) 모델별 탐색 ---
    for i, (name, model) in enumerate(models.items(), start=1):
        # --- 6-1) 탐색 대상 모델 확인 ---
        # 이미 튜닝된 객체를 다시 넘겨도 되도록 탐색 객체면 최적 모델을 꺼낸다.
        # 그대로 두면 탐색 객체를 또 감싸 param_grid 의 접두어가 어긋난다.
        # _unwrap_estimator 가 (안쪽 모델, 전처리, 최적 모델을 푼 파이프라인, best_params) 를 한 번에 돌려준다
        # (수정 전) getattr 로 푼 뒤 다시 _unwrap_estimator 를 부르던 두 단계 (2026-09-17)
        # estimator = getattr(model, 'best_estimator_', model)
        #
        # # 파이프라인이면 안쪽 모델을, 단독 모델이면 자기 자신을 꺼낸다
        # base_model, _, _, _ = _unwrap_estimator(estimator)
        base_model, _, estimator, _ = _unwrap_estimator(model)
        classname = type(base_model).__name__

        # 모델에 맞는 튜닝 조합을 꺼낸다. 정의되지 않은 모델이면 건너뛴다
        param_grid = param_grids.get(classname)

        if param_grid is None:
            skipped.append(name)
            if verbose:
                print(f'탐색 범위가 없어 건너뜀 [{i:2d}/{total:2d}] : {name}')
            continue

        # CatBoost 는 더미 인코딩 없이 원본 범주형 컬럼을 그대로 받으므로,
        # 어떤 컬럼이 범주형인지를 fit 인자로 알려줘야 한다
        if classname == 'CatBoostClassifier':
            fit_params = {'model__cat_features': cat_features}
        else:
            fit_params = {}

        # --- 6-2) 파라미터 탐색 수행 ---
        # 병렬 처리는 GridSearchCV 한 곳에서만 한다. 안쪽 모델(RF·XGB·LGBM·KNN·CatBoost)까지
        # 코어를 전부 쓰면 프로세스 수 × 스레드 수가 코어 수를 크게 넘어 오히려 느려진다.
        # 넘겨받은 원본은 그대로 두고 복제본에만 적용한다 (저장되는 튜닝 모델은 단일 스레드)
        if n_jobs not in (None, 1):
            # 병렬을 켜 둔(값이 None·1 이 아닌) 단계만 바꾼다. 값이 없는 곳까지 건드리면
            # n_jobs 가 폐기된 모델(sklearn 1.8 의 LogisticRegression)에서 경고가 난다
            inner_jobs = {k: 1 for k, v in estimator.get_params().items()
                          if k.endswith('n_jobs') and v not in (None, 1)}

            # CatBoost 는 스레드 수를 n_jobs 가 아니라 thread_count 로 받는다
            if classname == 'CatBoostClassifier':
                inner_jobs['model__thread_count'] = 1

            estimator = clone(estimator).set_params(**inner_jobs)

        started = perf_counter()
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid,
                          cv=cv, scoring=scoring, n_jobs=n_jobs)
        gs.fit(x_train, y_train, **fit_params)

        # 탐색 객체는 내부 모델을 clone 해서 재학습하므로 안쪽 name_ 이 사라진다.
        # 비교표에서 이름이 전부 'GridSearchCV' 가 되지 않도록 자신에게 이름을 붙인다
        tuned_name = f'{name}_tuned'
        gs.name_ = tuned_name
        save_model(gs, workdir / f'{tuned_name}.pkl')
        tuned[tuned_name] = gs

        if verbose:
            # 분류 지표는 높을수록 좋은 값 그대로라 회귀처럼 부호를 되돌릴 필요가 없다
            print(f'탐색 완료 [{i:2d}/{total:2d}] : {name} '
                  f'({perf_counter() - started:.1f}초, CV {primary}={gs.best_score_:.4f})')
            print(f'    최적 조합: {gs.best_params_}')

    # 전부 건너뛰었다면 비교할 대상이 없다
    if not tuned:
        print('튜닝된 모델이 없습니다. models 에 그리드가 정의된 분류 모델을 넣어 주세요.')
        return

    # --- 7) 모델간 성능 비교 + 튜닝 결과 시각화 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다.
    # 그래프는 순위를 매기는 쪽(_rank_score_table)이 함께 그린다
    score_table = cls_compare_models(tuned, x_test, y_test, primary=primary,
                                     aux=aux, average=average, verbose=False,
                                     plot=plot, title='하이퍼파라미터 튜닝 결과 성능 비교',
                                     width=width, height=height, save_path=save_path)

    # --- 8) 최종 순위표 출력 ---
    # 순위표가 이 함수의 결과물이므로 화면에 직접 출력한다.
    # 튜닝된 모델은 이미 pkl 로 저장했으니, 다시 쓸 때는 workdir 에서 load_model 로 불러온다
    print(f'\n모델 {len(tuned)}개 튜닝·저장 완료 → {workdir}')

    if skipped:
        print(f'탐색 범위가 없어 건너뛴 모델: {", ".join(skipped)}')

    display(score_table)
