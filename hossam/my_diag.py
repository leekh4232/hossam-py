# ---- 기본 참조 ----------------------------------------------------
import numpy as np                              # 배열 연산
from pandas import DataFrame, concat            # 데이터프레임 처리·행 결합
from pathlib import Path                        # 작업 폴더 경로 조립
from IPython.display import display             # 출력 기능

from . import RANDOM_STATE                      # 재현성을 위한 랜덤시드
from . import my_plot                           # 시각화 참조

# ---- my_ml 과 공유하는 도구 -------------------------------------------
from .my_ml import (
    _unwrap_estimator,      # 탐색 객체·파이프라인 → (모델, 전처리, 파이프라인, best_params)
    save_model,             # 객체를 joblib 으로 직렬화해 저장 (모델이 아닌 객체도 가능)
    reg_score,              # 회귀 지표 6종을 1행 표로 계산
    cls_score,              # (분류용) 분류 지표 7종을 1행 표로 계산
)

# ---- 모델·파이프라인 구조 탐색 ----------------------------------------
from sklearn.base import is_classifier               # 회귀·분류 과제 판별
from sklearn.pipeline import Pipeline                # 전처리 + 모델 연결
from sklearn.preprocessing import OneHotEncoder      # 더미 컬럼명 → 원본 컬럼명 복원

# ---- 과적합 판정 ----------------------------------------------------
from sklearn.model_selection import cross_validate, learning_curve as sk_learning_curve

# ---- SHAP 분석 관련 ----------------------------------------------------
# SHAP 이 직접 그린 그래프의 축 라벨·제목을 덧칠할 때만 사용한다.
# SHAP 그래프(beeswarm·dependence·waterfall)는 shap 패키지가 자체 캔버스에 그리므로
# my_plot 의 그리기 함수로는 제어할 수 없고, my_plot.init·show 로 감싸는 것이 한계다.
from matplotlib import pyplot as plt


# --------------------------------------------------------
# 훈련·교차검증·검증 성능을 한 표로 비교해 과적합 여부를 판정
# --------------------------------------------------------
def overfit(estimator, x_train, y_train, x_test, y_test,
            metrics=None, average='auto', threshold=0.20, underfit_threshold=None,
            cv=5, fit_params=None, learning_curve=True,
            width=1280, height=640, grid=True, save_path=None, verbose=True):
    """Train / CV / Test 성능을 한 표로 보여주고 과적합 여부를 판정한다 (회귀·분류 공용).

    Args:
        estimator: 학습된 회귀·분류 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        x_train, y_train, x_test, y_test: 훈련·검증 데이터셋.
        metrics (list): 표시·판정할 지표 (기본값: None → 회귀 ['RMSE','MAE','R2'], 
                                                    분류 ['F1','ROC_AUC','Accuracy']).
            분류의 DOR 은 사이킷런 scorer 가 없어 쓸 수 없다.
        average (str): 분류에서 다중클래스의 평균 방식 (기본값: 'auto').
        threshold (float): 과대적합으로 볼 Gap% (기본값: 0.20).
        underfit_threshold (float): 과소적합으로 볼 훈련 성능 하한 (기본값: None → 회귀 0.05, 분류 0.6).
        cv (int): 교차검증 폴드 수 (기본값: 5).
        fit_params (dict): CV 재학습에 넘길 인자 (예: {'model__cat_features': [...]}) (기본값: None).
        learning_curve (bool): 학습곡선 출력 여부 (기본값: True).
        width (int): 학습곡선 가로 크기(픽셀) (기본값: 1280).
        height (int): 학습곡선 세로 크기(픽셀) (기본값: 640).
        grid (bool): 학습곡선 격자 표시 여부 (기본값: True).
        save_path (str): 학습곡선 이미지 저장 경로 (기본값: None).
        verbose (bool): 판정 과정 출력 여부 (기본값: True).

    Raises:
        ValueError: metrics 에 지원하지 않는 지표명을 준 경우.
    """
    # --- 1) 탐색 객체에서 최적 모델 꺼내기 ---
    # 탐색 객체(GridSearchCV 등)를 그대로 교차검증에 넣으면 폴드마다 하이퍼파라미터
    # 탐색을 다시 도는 중첩 교차검증이 되어 버리므로, 진단 전에 최적 모델을 꺼낸다.
    model, _, base_est, _ = _unwrap_estimator(estimator)
    classname = type(model).__name__        # 모델 클래스명 (표·그래프 제목에 쓴다)

    if classname.startswith("CatBoost"):    # CatBoost 계열인 경우
        # 교차검증은 폴드마다 모델을 새로 학습하므로 어떤 컬럼이 범주형인지 다시 알려줘야 한다.
        # 데이터 타입으로 추측하면 object 컬럼이나 nominal_cols 로 직접 지정한 컬럼을 놓치므로,
        # 학습을 마친 모델이 실제로 범주형으로 쓴 컬럼의 위치(인덱스)를 그대로 꺼내 쓴다
        cat_features = [int(i) for i in model.get_cat_feature_indices()]

        # 파이프라인이면 '마지막 단계명__cat_features', 단독 모델이면 'cat_features' 로 넘긴다
        if isinstance(base_est, Pipeline):
            key = f'{base_est.steps[-1][0]}__cat_features'
        else:
            key = 'cat_features'

        # 넘겨받은 fit_params 원본은 건드리지 않고, 이미 지정한 값이 있으면 그 값을 따른다
        fit_params = dict(fit_params or {})
        fit_params.setdefault(key, cat_features)

    # --- 2) 모델 유형 판별 ---
    # is_classifier 는 모델이 분류기인지 여부를 돌려준다. 이 한 줄이 아래 모든 분기의 기준이다.
    if is_classifier(model):        # 분류 모형인 경우
        task = 'classification'
    else:                           # 예측 모형인 경우
        task = 'regression'

    # --- 3) 과제별 지표표 정의 ---
    # 지표표는 {지표: (최적 방향, scoring, 폴드 점수 보정 코드)} 형태다. 
    # 보정 코드는 사이킷런이 돌려준 폴드 점수를 reg_score·cls_score 와 같은 단위로 되돌리는 방법이다.
    #   as_is    : 그대로 사용
    #   neg      : 부호만 뒤집기 (neg_* scorer 는 '클수록 좋게' 뒤집혀 있다)
    #   neg_sqrt : 부호를 되돌린 뒤 제곱근 (MSLE → RMSLE)
    if task == 'regression':        # 예측 모형인 경우
        # 회귀는 사이킷런이 이름만 보고 알아듣는 scoring 문자열을 그대로 쓴다.
        # MPE 는 0 에 가까울수록 좋은 지표라 '훈련보다 얼마나 나쁜가' 를 정의할 수 없어 제외한다
        metric_specs = {
            'R2':    ('higher', 'r2',                                 'as_is'),
            'MAE':   ('lower',  'neg_mean_absolute_error',            'neg'),
            'MSE':   ('lower',  'neg_mean_squared_error',             'neg'),
            'RMSE':  ('lower',  'neg_root_mean_squared_error',        'neg'),
            'RMSLE': ('lower',  'neg_mean_squared_log_error',         'neg_sqrt'),
            'MAPE':  ('lower',  'neg_mean_absolute_percentage_error', 'neg'),
        }
    else:                           # 분류 모형인 경우
        # 분류도 회귀와 같이 사이킷런 scoring 문자열을 그대로 쓴다 (cls_tunes 와 같은 규칙).
        # 이진이면 양성 클래스(1) 기준 scorer 를, 다중분류면 평균 방식(average)이 붙은 scorer 를 쓴다.
        #   --> cls_baseline·cls_tunes 가 종속변수를 0 부터 시작하는 정수로 맞춰 두므로 양성 클래스는 1 이다
        # 분류 지표는 모두 클수록 좋고 부호가 뒤집히지 않으므로 보정 코드는 전부 as_is 다.
        # DOR 은 사이킷런 scorer 가 없어 지표표에서 제외한다 (회귀에서 MPE 를 제외한 것과 같은 처리).
        y_labels = np.unique(np.asarray(y_train).ravel())
        binary = len(y_labels) == 2

        if binary and average in ('auto', 'binary'):    # 이진분류 — 양성 클래스 기준
            metric_specs = {
                'Accuracy':  ('higher', 'accuracy',          'as_is'),
                'Precision': ('higher', 'precision',         'as_is'),
                'Recall':    ('higher', 'recall',            'as_is'),
                'F1':        ('higher', 'f1',                'as_is'),
                'ROC_AUC':   ('higher', 'roc_auc',           'as_is'),
                'PR_AUC':    ('higher', 'average_precision', 'as_is'),
            }
        else:                                           # 다중분류 — 클래스 평균
            # ROC_AUC 는 일대다(OvR) 평균만 지원한다 (micro 는 사이킷런에 없어 macro 로 계산).
            # PR_AUC 는 다중분류용 문자열 scorer 가 없어 제외한다
            avg = 'macro' if average in ('auto', 'binary') else average
            roc_auc = 'roc_auc_ovr_weighted' if avg == 'weighted' else 'roc_auc_ovr'
            metric_specs = {
                'Accuracy':  ('higher', 'accuracy',         'as_is'),
                'Precision': ('higher', f'precision_{avg}', 'as_is'),
                'Recall':    ('higher', f'recall_{avg}',    'as_is'),
                'F1':        ('higher', f'f1_{avg}',        'as_is'),
                'ROC_AUC':   ('higher', roc_auc,            'as_is'),
            }

    # --- 4) 판정 기준값 설정 ---
    # metrics            : 표에 실을 지표
    # underfit_metric    : 과소적합 판정 지표. 스케일에 좌우되지 않아 임계값을 고정할 수
    #                      있는 지표로 고른다 (회귀 R2 · 분류 ROC_AUC).
    # underfit_threshold : 훈련 성능이 이보다 낮으면 과소적합
    # threshold          : 훈련↔CV 격차(Gap%)가 이 값 이상이면 과대적합 (인자로 받는다)
    if task == 'regression':        # 예측 모형인 경우
        underfit_metric = 'R2'      # 과소적합 판정 지표 (스케일에 좌우되지 않는다)

        if metrics is None:    # 지정하지 않았으면 기본 지표를 쓴다
            metrics = ['RMSE', 'MAE', 'R2']     # 회귀 기본 지표

        if underfit_threshold is None:    # 지정하지 않았으면 기본 임계값을 쓴다
            underfit_threshold = 0.05           # train R2 가 이보다 낮으면 과소적합
    else:                           # 분류 모형인 경우
        underfit_metric = 'ROC_AUC'     # 과소적합 판정 지표 (0.5 = 동전 던지기)

        if metrics is None:    # 지정하지 않았으면 기본 지표를 쓴다
            metrics = ['F1', 'ROC_AUC', 'Accuracy']     # 분류 기본 지표

        if underfit_threshold is None:    # 지정하지 않았으면 기본 임계값을 쓴다
            underfit_threshold = 0.6                    # train ROC_AUC 가 이보다 낮으면 과소적합

    # --- 5) 파라미터 검증 ---
    if isinstance(metrics, str):    # 지표를 문자열 하나로 준 경우도 허용
        metrics = [metrics]

    for m in metrics:    # 요청한 지표를 하나씩 확인
        if m not in metric_specs:       # 계산할 수 없는 지표명
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    # --- 6) 훈련(Train), 검증(Test) 점수 계산 ---
    # 기출문제에 대한 성적 — 학습에 쓴 데이터에서의 성능이다.
    if task == 'regression':    # 예측 모형인 경우
        train_scores = reg_score(base_est, x_train, y_train)
    else:                      # 분류 모형인 경우
        train_scores = cls_score(base_est, x_train, y_train, average=average)

    # 학습에 쓰지 않은 데이터에서의 성능. 판정에는 쓰지 않고 참고용으로만 표에 싣는다.
    if task == 'regression':    # 예측 모형인 경우
        test_scores = reg_score(base_est, x_test, y_test)
    else:                      # 분류 모형인 경우
        test_scores = cls_score(base_est, x_test, y_test, average=average)

    # --- 7) 교차검증 준비 (1) — 종속변수 형태 통일과 지표 목록 ---
    # 과적합 판정은 Train ↔ CV 격차로 한다.
    if isinstance(y_train, DataFrame):      # 종속변수가 DataFrame 인 경우
        y_cv = y_train.values.ravel()       # 1차원 배열로 통일
    else:                                   # Series·ndarray 인 경우
        y_cv = np.asarray(y_train).ravel()

    wanted = list(metrics)                  # 표에 실을 지표

    if underfit_metric not in wanted:       # 과소적합 판정 지표가 빠져 있으면
        wanted.append(underfit_metric)      # 판정을 위해 계산 목록에만 넣는다

    # --- 8) 교차검증 준비 (2) — scoring 딕셔너리 구성 ---
    # cross_validate 는 {이름: scoring} 딕셔너리를 받으면 
    # 폴드마다 여러 지표를 한 번에 계산해 test_<이름> 키로 돌려준다. 
    scoring = {}

    for m in wanted:    # 계산할 지표를 하나씩
        scoring[m] = metric_specs[m][1]         # 지표명 → scoring 문자열·scorer

    # --- 9) 교차검증 수행 ---
    # sklearn 의 cross_validate는 CV 점수만 리턴한다.
    # 훈련 점수는 앞에서 미리 계산해 두었다.
    out = cross_validate(base_est, x_train, y_cv, cv=cv, scoring=scoring,
                         n_jobs=-1, params=fit_params)

    # --- 10) 폴드 점수 보정 ---
    cv_scores = {}

    for m in wanted:    # 지표마다 폴드 점수를 보정한다
        fold = out[f'test_{m}']     # 폴드별 점수
        code = metric_specs[m][2]   # 부호·단위 보정 코드

        if code == 'neg':           # 부호만 뒤집는 지표
            fold = -fold
        elif code == 'neg_sqrt':    # 부호를 되돌린 뒤 제곱근을 취하는 지표 (MSLE → RMSLE)
            fold = np.sqrt(-fold)

        cv_scores[m] = fold

    # --- 11) 과소적합 판정 ---
    # 과소적합은 지표별로 따지지 않고 모델 수준에서 한 번만 판정하므로, 
    # 그 기준으로 사용할 지표(회귀 R2 · 분류 ROC_AUC)의 훈련 점수와 CV 평균을 먼저 꺼내 둔다.
    train_base = train_scores[underfit_metric].iloc[0]      # 판정 지표의 훈련 점수
    cv_base = float(np.mean(cv_scores[underfit_metric]))    # 판정 지표의 CV 점수

    # 과소적합 여부
    # 훈련 점수가 임계값보다 낮으면 과소적합으로 본다. CV 점수는 판정에 쓰지 않는다.
    # (수정 전) 판정 지표가 NaN 일 때 판정을 보류하는 분기. cls_score 가 NaN 을 내지 않게 되어 불필요 (2026-09-17)
    # underfit_available = not np.isnan(train_base)
    # underfit = bool(underfit_available and train_base < underfit_threshold)
    underfit = bool(train_base < underfit_threshold)

    # --- 12) 지표별 격차 계산과 판정 ---
    rows = {}

    for m in metrics:    # 표에 실을 지표를 하나씩
        # --- 12-1) 훈련·CV·검증 점수와 격차 계산 ---
        direction = metric_specs[m][0]              # 'higher' = 클수록 좋은 지표
        tr = train_scores[m].iloc[0]                # 훈련 점수
        te = test_scores[m].iloc[0]                 # 검증 점수
        cv_mean = float(np.mean(cv_scores[m]))      # CV 평균
        cv_std = float(np.std(cv_scores[m]))        # CV 표준편차

        # 'CV 가 훈련보다 나쁜 정도' 가 항상 양수가 되도록 방향을 맞춘다
        # 클수록 좋은 지표 (R2·F1·ROC_AUC)와 그렇지 않은 지표 구분
        gap = tr - cv_mean if direction == 'higher' else cv_mean - tr

        # Gap% : 격차를 스케일 보정.
        # 상한이 1 인 지표(R2·분류 지표)는 점수차를 그대로 사용하고, 상한이 없는 회귀 오차 지표는 크기로 상대화한다.
        # (수정 전) DOR 이 지표표에 있을 때의 예외 처리. 문자열 scorer 로 바꾸며 DOR 이 빠져 불필요해짐 (2026-09-16)
        # unbounded = direction != 'higher' or m == 'DOR'
        # denom = max(abs(tr), abs(cv_mean)) if unbounded else 1.0
        denom = 1.0 if direction == 'higher' else max(abs(tr), abs(cv_mean))

        # # 분모가 0 이 아닌 경우 비율을 계산하고 0이면 NaN 으로 둔다. (0으로 나누면 inf 가 되므로)
        gap_pct = gap / denom if denom else np.nan

        # --- 12-2) 지표별 판정 ---
        if underfit:                  # 모델 수준 판정이 지표별 판정에 우선한다
            label = '과소적합'
        # (수정 전) ROC_AUC 가 NaN 이던 모델을 위한 'N/A' 라벨. 위 수정으로 도달하지 않음 (2026-09-17)
        # elif np.isnan(gap_pct):       # 격차를 계산하지 못한 경우
        #     label = 'N/A'
        elif gap_pct >= threshold:    # 격차가 임계값 이상
            label = '과대적합'
        else:                         # 격차가 임계값 미만
            label = '일반화'

        rows[m] = {'Train': tr, 'CV': cv_mean, 'CV_Std': cv_std, 'Test': te,
                   'Gap': gap, 'Gap%': round(gap_pct * 100, 2), 'Overfit': label}

    # --- 13) 결과표 조립 ---
    result = DataFrame.from_dict(rows, orient='index')
    result.index.name = 'Metric'

    # --- 14) 지표별 판정 취합 ---
    # 지표마다 스케일과 민감도가 달라, 한 지표에서만 드러나는 격차도 실제 신호일 수 있다.
    has_overfit = False

    for m in metrics:    # 지표별 판정을 훑는다
        # 한 지표라도 과대적합이면 모델도 과대적합으로 본다
        if rows[m]['Overfit'] == '과대적합':
            has_overfit = True

    # --- 15) 모델 수준 최종 진단 ---
    # 과소적합(高편향) > 과대적합(高분산) > 일반화 순으로 우선한다. 
    if underfit:         # 高편향 — 가장 우선
        diagnosis = '과소적합'
    elif has_overfit:    # 高분산
        diagnosis = '과대적합'
    else:                # 둘 다 아니면 일반화
        diagnosis = '일반화'

    result.attrs['diagnosis'] = diagnosis   # 모델 수준 최종 진단
    result.attrs['task'] = task             # 'regression' | 'classification'

    # --- 16) 학습곡선 ---
    # 수치 판정보다 먼저 그려서 '곡선을 보고 표로 확인' 하는 흐름을 만든다.
    if learning_curve:    # 학습곡선을 그리는 경우
        # --- 16-1) 표본 수를 늘려 가며 점수 산출 ---
        _, lc_scoring, lc_code = metric_specs[metrics[0]]

        sizes, train_curve, cv_curve = sk_learning_curve(
            base_est, x_train, y_cv, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring=lc_scoring, n_jobs=-1, params=fit_params,
        )

        # 점수 보정
        if lc_code == 'neg':           # 부호만 뒤집는 지표
            train_curve = -train_curve
            cv_curve = -cv_curve
        elif lc_code == 'neg_sqrt':    # 부호를 되돌린 뒤 제곱근을 취하는 지표
            train_curve = np.sqrt(-train_curve)
            cv_curve = np.sqrt(-cv_curve)

        # --- 16-2) 긴 형식(long format) 표로 변환 ---
        curve_rows = []

        for i in range(len(sizes)):    # 훈련 표본 수마다
            for j in range(train_curve.shape[1]):    # 폴드마다
                curve_rows.append({'훈련 표본 수': int(sizes[i]),
                                   '점수': float(train_curve[i, j]),
                                   '구분': 'Train'})
                curve_rows.append({'훈련 표본 수': int(sizes[i]),
                                   '점수': float(cv_curve[i, j]),
                                   '구분': 'Validation (CV)'})

        curve_df = DataFrame(curve_rows)

        # --- 16-3) 시각화 ---
        my_plot.lineplot(data=curve_df, x='훈련 표본 수', y='점수', hue='구분',
                         marker='o', errorbar='sd',
                         title=f'Learning Curve: {classname}',
                         xlabel='훈련 표본 수', ylabel=f'{metrics[0]} 점수',
                         width=width, height=height, save_path=save_path)

    # --- 17) 최종 판정 결과 출력 ---
    display(result) # 결과표 출력

    if verbose:    # 판정 과정을 출력하는 경우
        print('\n' + '=' * 78)
        print(f'◆ Fit Diagnosis: {classname}  '
              f'(threshold={threshold:.0%}, 기준=Train↔CV {cv}-Fold)')
        print('   ※ threshold 는 학술 표준이 아닌 경험칙입니다. '
              '임계값보다 학습곡선 추세를 함께 보세요.')
        print('=' * 78)

        for m in metrics:    # 지표마다 한 줄씩 출력
            r = rows[m]

            if r['Overfit'] == '과대적합':
                tag = ' [주의]'
            elif r['Overfit'] == '과소적합':
                tag = ' [경고]'
            else:
                tag = ' [정상]'

            # 지표명 폭을 6 → 8 로 넓힌다. 분류 지표명(ROC_AUC·Accuracy)이 6자를 넘어
            # 회귀(RMSE·MAE·R2)에서는 맞던 열이 분류에서는 어긋났다 (2026-09-16, LAB-04 06 실습)
            # print(f"   - {m:<6}  Train={r['Train']:>11.4f}  "
            #       f"CV={r['CV']:>11.4f} (±{r['CV_Std']:.4f})  Test={r['Test']:>11.4f}  "
            #       f"Gap%={r['Gap%']:>8.2f}%  [{r['Overfit']}]{tag}")
            print(f"   - {m:<8}  Train={r['Train']:>11.4f}  "
                  f"CV={r['CV']:>11.4f} (±{r['CV_Std']:.4f})  Test={r['Test']:>11.4f}  "
                  f"Gap%={r['Gap%']:>8.2f}%  [{r['Overfit']}]{tag}")

        if diagnosis == '과대적합':      # 高분산
            message = '⚠ 과대적합 (Overfit · 高분산) — 훈련↔CV 격차가 큼'
        elif diagnosis == '과소적합':    # 高편향
            message = '⚑ 과소적합 (Underfit · 高편향) — 훈련 성능 자체가 낮음'
        else:                        # 정상
            message = '✔ 일반화 (Good fit) — 격차가 작고 훈련 성능도 양호'

        print(f'\n   ▶ 진단: {message}')

        # (수정 전) 판정을 보류한 경우의 안내 분기. underfit_available 제거에 따라 함께 정리 (2026-09-17)
        # if underfit_available:    # 판정 지표를 계산할 수 있었던 경우
        #     print(f'     · train {underfit_metric}={train_base:.4f} '
        #           f'(과소적합 기준 < {underfit_threshold}) · '
        #           f'CV {underfit_metric}={cv_base:.4f}')
        # else:                     # 계산할 수 없어 판정을 보류한 경우
        #     print(f'     · {underfit_metric} 를 계산할 수 없어 과소적합 판정은 보류했습니다 '
        #           f'(확률을 내지 못하는 모델).')
        print(f'     · train {underfit_metric}={train_base:.4f} '
              f'(과소적합 기준 < {underfit_threshold}) · '
              f'CV {underfit_metric}={cv_base:.4f}')

        print('=' * 78 + '\n')


# --------------------------------------------------------
# 변수 중요도를 산출해 상위 변수를 추려서 반환
# --------------------------------------------------------
def feature_importance(estimator, cum_ratio=0.95, 
                       plot=True, title=None, xlabel=None, ylabel=None, palette='tab10'):
    """학습된 회귀·분류 모델에서 변수 중요도를 도출해 상위 변수만 추려서 반환한다.

    Args:
        estimator: 학습된 회귀·분류 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        cum_ratio (float): 채택할 누적 중요도 비율 (기본값: 0.95).
        plot (bool): 결과를 시각화할지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).

    Returns:
        DataFrame: 중요도 내림차순 전체 변수표.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 중요도 총합이 0 인 경우.
        TypeError: 변수 중요도를 도출할 수 없거나 변수명을 얻을 수 없는 모델인 경우.
    """
    # --- 1) 임계값(cum_ratio) 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    # --- 2) 최종 모델 추출 ---
    # 모델 객체, 전처리 객체, 기본 추정기로 분리한다 (탐색 결과는 쓰지 않는다).
    model, pre, base_est, _ = _unwrap_estimator(estimator)
    model_class = type(model).__name__      # 모델 클래스명

    # --- 3) 변수명 추출 ---
    if pre is not None:                          # 파이프라인인 경우
        feature_names = np.asarray(pre.get_feature_names_out())     # 변환 후 변수명 (원핫·PCA 뒤)
    elif hasattr(model, 'feature_names_in_'):    # 단독 모델인데 컬럼명을 기억하는 경우
        feature_names = np.asarray(model.feature_names_in_)
    else:                                        # 이름을 알 수 없는 경우
        raise TypeError(
            f"'{model_class}' 모델에서 변수명을 얻을 수 없습니다. "
            f"DataFrame 으로 학습했거나 fit_pipeline 으로 만든 파이프라인이어야 "
            f"변수 중요도를 변수명과 짝지을 수 있습니다."
        )

    # --- 4) 변수 중요도 도출 (모델 유형 분기) ---
    if hasattr(model, 'feature_importances_'):                      # 트리·부스팅 계열
        if model_class in ('XGBRegressor', 'XGBClassifier'):        # XGBoost
            # 학습된 booster 에서 gain 을 직접 추출한다 (모델 속성은 건드리지 않는다).
            # get_score 는 분할에 쓰인 변수만 dict 로 주므로 booster 의 변수 순서대로 0 을 채운다
            booster = model.get_booster()
            score = booster.get_score(importance_type='gain')
            names = booster.feature_names           # booster 가 아는 변수 순서

            imp = []
            for f in names:    # booster 의 변수 순서대로
                imp.append(score.get(f, 0.0))       # 분할에 쓰이지 않은 변수는 0

            importances = np.array(imp, dtype=float)
            importance_metric = 'gain'

        elif model_class in ('LGBMRegressor', 'LGBMClassifier'):    # LightGBM
            imp = model.booster_.feature_importance(importance_type='gain')
            importances = np.asarray(imp, dtype=float).ravel()
            importance_metric = 'gain'

        else:                                                       # 그 외 트리 계열
            # 모델이 제공하는 feature_importances_ 를 그대로 사용
            importances = np.asarray(model.feature_importances_, dtype=float).ravel()

            labels = {
                'DecisionTreeRegressor':  'MDI',
                'RandomForestRegressor':  'MDI',
                'DecisionTreeClassifier': 'MDI',
                'RandomForestClassifier': 'MDI',
                'CatBoostRegressor':      'PredictionValuesChange',
                'CatBoostClassifier':     'PredictionValuesChange',
            }
            importance_metric = labels.get(model_class, 'feature_importances_')

    elif hasattr(model, 'coef_'):                                   # 선형 계열
        coef = np.abs(np.asarray(model.coef_, dtype=float))     # 부호는 무관하므로 절대값

        if coef.ndim == 2 and coef.shape[0] > 1:    # 다중클래스 (클래스 × 변수) 인 경우 → 클래스축 평균
            importances = coef.mean(axis=0)
        else:                                       # 단일 출력 (회귀·이진분류) 인 경우
            importances = coef.ravel()

        importance_metric = '|coef_|'

    else:    # 중요도를 정의할 수 없는 모델
        raise TypeError(f"'{model_class}' 모델은 변수 중요도를 도출할 수 없습니다.")

    # --- 5) 원핫 더미와 원본 컬럼의 매핑 만들기 ---
    # OneHotEncoder 의 get_feature_names_out 출력은 입력 컬럼 순서대로 묶여 있고, 
    # 입력 컬럼별 더미 개수는 카테고리 수에서 drop 된 개수를 뺀 값이다. 
    # 이 개수만큼 출력명을 끊어 원본 컬럼에 귀속시킨다.
    name_map = {}       # {더미 컬럼명: 원본 컬럼명}

    if isinstance(base_est, Pipeline):    # 파이프라인이어야 인코더를 들여다볼 수 있다
        pre_step = base_est.named_steps.get('preprocessor')

        if pre_step is not None:    # 전처리 단계가 있는 경우
            for _name, trans, _cols in pre_step.transformers_:  # 연속형·명목형 분기별로
                # --- 원핫 인코더를 찾아서 더미 컬럼명을 원본 컬럼명으로 매핑 ---
                if isinstance(trans, Pipeline):                 # 변환기가 파이프라인인 경우
                    ohe = trans.named_steps.get('onehot')
                elif isinstance(trans, OneHotEncoder):          # 인코더가 바로 붙은 경우
                    ohe = trans
                else:                                           # 원핫이 아닌 변환기
                    ohe = None

                # 원핫이 아니거나 categories_ 속성이 없다면 건너뛴다
                if ohe is None or not hasattr(ohe, 'categories_'):
                    continue

                # --- 원핫 인코딩 전후 컬럼명을 추출하여 더미 컬럼을 원본 컬럼에 귀속 ---
                in_cols = list(ohe.feature_names_in_)                   # 인코딩 전 원본 컬럼
                out_names = list(ohe.get_feature_names_out(in_cols))    # 인코딩 후 더미 컬럼

                drop_idx = getattr(ohe, 'drop_idx_', None)  # drop='first' 등으로 제거된 카테고리
                pos = 0                                     # 더미 컬럼 순서 커서

                for i, col in enumerate(in_cols):    # 원본 컬럼마다
                    n_out = len(ohe.categories_[i])         # 이 컬럼이 만든 더미 개수

                    if drop_idx is not None and drop_idx[i] is not None:    # 카테고리 하나가 제거된 경우
                        n_out = n_out - 1

                    for _ in range(n_out):    # 그 컬럼이 만든 더미 개수만큼
                        name_map[out_names[pos]] = col      # 더미를 원본 컬럼에 귀속
                        pos = pos + 1

    # --- 6) 더미 중요도를 원본 컬럼 단위로 합산 ---
    if name_map:    # 합산할 더미가 있는 경우
        agg = {}

        for fname, imp in zip(feature_names, importances):      # 변수마다 원본 컬럼에 더한다
            origin = name_map.get(fname, fname)                 # 매핑이 없으면 자기 자신
            agg[origin] = agg.get(origin, 0.0) + imp

        feature_names = np.array(list(agg.keys()))              # 합산 후 변수명
        importances = np.array(list(agg.values()), dtype=float) # 합산 후 변수 중요도

    # --- 7) 중요도 총합 검사 ---
    # 중요도의 절대 크기는 라이브러리마다 기준이 달라 비율로 바꿔서 읽는데,
    # 총합이 0 이면 비율을 만들 수 없으므로 먼저 걸러낸다.
    total = importances.sum()

    if total <= 0:    # 모든 중요도가 0 인 경우
        raise ValueError(
            f"중요도 총합이 0 입니다 ({model_class}). 모델이 학습되지 않았거나 "
            f"(Lasso·ElasticNet 등) 모든 계수가 0 으로 규제되었을 수 있습니다."
        )

    # --- 8) 합이 1 이 되도록 정규화 ---
    ratio = importances / total     # 각 변수가 전체 중요도에서 차지하는 비율

    # --- 9) 결과표 구성 ---
    result = DataFrame({
        'Importance': importances,
        'Ratio': ratio,
    }, index=feature_names)
    result.index.name = 'Feature'

    # 중요도 기준 내림차순 정렬
    result.sort_values(by='Importance', ascending=False, inplace=True)

    # --- 10) 누적 비율 계산 ---
    # 위에서부터 비율을 차례로 더한 값 = 상위 몇 개까지 쓰면 몇 % 를 설명하는가
    result['CumRatio'] = result['Ratio'].cumsum()

    # --- 11) 채택 여부 판정 ---
    # 누적 비율이 cum_ratio 에 처음 도달하는 변수까지 채택한다 (경계 변수 포함)
    reached = np.flatnonzero(result['CumRatio'].to_numpy() >= cum_ratio)

    if len(reached) > 0:    # 기준에 도달한 경우
        k = int(reached[0]) + 1     # 도달 지점보다 하나 큰 값이 채택 개수
    else:                   # 부동소수점 오차 등으로 도달하지 못한 경우
        k = len(result)             # 전체를 채택

    result['채택여부'] = np.where(np.arange(len(result)) < k, '채택', '탈락')

    # --- 12) 결과 시각화 및 리턴 ---
    if plot:    # 그래프를 그리는 경우
        # 제목, x축 라벨, y축 라벨이 주어지지 않으면 기본값을 설정한다
        if title is None:   title = f'Feature Importance ({importance_metric})'
        if xlabel is None:  xlabel = 'Ratio'
        if ylabel is None:  ylabel = 'Feature'

        # 변수 하나당 높이를 확보한다 (변수 수 × 65px + 제목 공간 50px)
        h = 65 * len(result) + 50
        fig, ax = my_plot.init(title=title, xlabel=xlabel, ylabel=ylabel, height=h)

        # 채택·탈락을 색으로 구분한 가로 막대그래프
        my_plot.barplot(data=result, x='Ratio', y=result.index, hue='채택여부',
                        palette=palette, ax=ax)

        for i, v in enumerate(result['Ratio']):    # 막대 오른쪽에 비율 값을 표시
            ax.text(v + 0.001, i, f"{v:.3f}", color='black', va='center')

        my_plot.show()

    return result


# --------------------------------------------------------
# SHAP 값을 계산해 변수 기여도 요약표를 반환
# --------------------------------------------------------
def shap_analysis(project_name, estimator, x, max_samples='auto',
                  background_clusters=50, workdir='shap'):
    """학습된 예측(회귀) 모델을 SHAP 으로 분석해 변수 기여도 요약표를 반환한다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        estimator: 학습된 파이프라인 또는 그것을 감싼 GridSearchCV 등 탐색 객체.
        x (DataFrame): 설명에 사용할 원본 입력 (보통 x_train 또는 x_test).
        max_samples (str or int): 설명 대상 행 수 (기본값: 'auto').
            'auto' 면 선형·트리는 전체, KernelExplainer 는 200행을 쓴다.
            정수를 주면 explainer 종류와 무관하게 그 행 수만큼, None 이면 언제나 전체.
        background_clusters (int): KernelExplainer 의 배경을 kmeans 로 압축할 대표점 수 (기본값: 50).
        workdir (str): 분석 결과 pkl 을 저장할 폴더명 (기본값: "shap").

    Returns:
        DataFrame: mean_abs_shap 내림차순 요약표.
    """
    # KernelExplainer에 적용할 행 수 상한.
    # Kernel은 행 하나에 수 초가 걸려 전체를 돌리면 십수 시간이 된다.
    _KERNEL_MAX_SAMPLES = 200

    # shap 은 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
    import shap

    # --- 1) 최종 모델 추출 ---
    model, pre, _, _ = _unwrap_estimator(estimator)
    model_class = type(model).__name__      # 모델 클래스명

    if pre is None:    # 전처리 단계가 없는 단독 모델인 경우
        raise TypeError(f"'{model_class}' 단독 모델은 받지 않습니다. 전처리를 포함한 파이프라인을 넘기세요.")

    # 이 함수는 회귀 전용이다. 분류 구현은 helpers/backup/shap_analysis_classification.py 에 있다
    # (수정 전) 분류 대응을 위한 과제 판별. 분류 분기가 미완성(pass)이라 회귀 전용으로 정리 (2026-09-17)
    # if is_classifier(model):        # 분류 모형인 경우
    #     task = 'classification'
    # else:                           # 예측 모형인 경우
    #     task = 'regression'
    if is_classifier(model):
        raise TypeError(f"'{model_class}' 분류 모델은 지원하지 않습니다. 회귀 모델·파이프라인을 넘기세요.")

    # --- 2) 전처리 재현하기 ---
    x_df = pre.transform(x).copy()

    # --- 3) explainer 생성 ---
    explainer = None
    explainer_type = None

    if hasattr(model, 'coef_'):     # 선형 — 계수로 기여도를 바로 풀 수 있다
        # LinearExplainer 는 kmeans 요약을 받지 못하므로 원시 데이터를 그대로 넘긴다
        explainer = shap.LinearExplainer(model, x_df)
        explainer_type = 'LinearExplainer'
    else:    # 선형이 아닌 경우
        try:                 # 트리 구조면 정확·고속이라 먼저 시도한다
            explainer = shap.TreeExplainer(model)
            explainer_type = 'TreeExplainer'
        except Exception:    # shap 이 거부한 경우 (KNN·SVM 등)
            explainer = None

    if explainer is None:           # 폴백 — 모델 구조를 따지지 않지만 느리다
        # Kernel 은 (배경 대표점 수 × 설명 행 수) 에 비례해 느리므로 배경을 kmeans 로 압축한다
        bg = shap.kmeans(x_df, min(background_clusters, len(x_df)))
        explainer = shap.KernelExplainer(model.predict, bg)
        explainer_type = 'KernelExplainer'

    # --- 4) 행 샘플링 ---
    # 선형·트리는 정확 계산이라 전체를 돌려도 대개 몇 초면 끝난다. 
    # Kernel 만 행 하나에 수 초가 걸려 (이 규모에서 전체를 돌리면 십수 시간) 자동으로 행을 줄인다.
    if max_samples == 'auto':    # 기본 — explainer 종류에 맡긴다
        if explainer_type == 'KernelExplainer':    # 근사 계산 — 행을 줄인다
            n_target = _KERNEL_MAX_SAMPLES
        else:                                      # 정확 계산 — 전체를 쓴다
            n_target = None
    else:                        # 사용자가 직접 지정한 경우
        n_target = max_samples

    if n_target is not None and len(x_df) > n_target:    # 상한을 넘으면 표본을 뽑는다
        # 재현 가능한 샘플링으로 계산량을 줄인다
        rng = np.random.RandomState(RANDOM_STATE)
        pick = rng.choice(len(x_df), size=n_target, replace=False)
        pick.sort()                     # 원본 행 순서를 유지한다
        x_explain = x_df.iloc[pick]     # SHAP 을 계산할 행
    else:    # 상한이 없거나 상한 이하인 경우 — 전체를 쓴다
        x_explain = x_df

    # --- 5) SHAP 값 계산 ---
    if explainer_type == 'KernelExplainer':    # Kernel 은 진행 막대를 끈다
        raw = explainer.shap_values(x_explain, silent=True)
    else:                                      # Tree·Linear
        raw = explainer.shap_values(x_explain)

    expected_value = explainer.expected_value      # base value (평균 예측)

    # --- 6) SHAP 값 배열 정리 ---
    # 회귀 모형의 결과 배열은 (n, f) 단일 출력이다.
    arr = np.asarray(raw)

    # (수정 전) 클래스축이 있는 (n, f, c) 분류 출력을 위한 자리(pass). 회귀 전용으로 정리 (2026-09-17)
    # if arr.ndim == 3:    # (n, f, c) — 클래스축이 있는 분류 모형
    #     pass 
    # else:                # (n, f) — 회귀 모형의 단일 출력
    #     shap_2d = arr.astype(float)
    #     base_value = float(np.ravel(expected_value)[0])
    shap_2d = arr.astype(float)                         # (n, f) 기여도 배열
    base_value = float(np.ravel(expected_value)[0])     # base value (평균 예측)

    # --- 7) 가산성으로 모델 출력 복원 ---
    # SHAP 은 '가산성' 을 만족한다 — 행마다 (base value + 그 행의 SHAP 합) 이 모델 출력과 같아진다. 
    # 그래서 이 합을 모델의 실제 출력과 맞춰보면 단위를 역으로 알 수 있다.
    preds = shap_2d.sum(axis=1) + base_value    # 행별로 복원한 모델 출력

    # --- 8) 예측값과 대조 ---
    # 예측 모형은 후보가 하나뿐이라 model.predict 와만 맞춰 보면 된다.
    # (수정 전) 분류 분기 자리(pass). 회귀 전용으로 정리하며 분기를 풀었다 (2026-09-17)
    # if task == 'classification':    # 분류 모형인 경우
    #     pass
    # else:                           # 예측 모형인 경우
    # SHAP 값의 단위를 판정하기 위해 모델 예측값을 가져온다
    y_hat = np.asarray(model.predict(x_explain), dtype=float).ravel()
    # 0으로 나누는 것을 방지하기 위해 작은 수를 더한다
    scale = float(np.abs(y_hat).mean()) + 1e-9
    # SHAP 으로 복원한 예측값과 실제 예측값의 평균 절대 오차를 계산한다
    error = float(np.abs(preds - y_hat).mean())

    if error < scale * 1e-3:    # 예측값과 사실상 일치하면
        output_space = '예측값(y 단위)'
    else:                       # 어긋나면 단위를 특정할 수 없다
        output_space = '알 수 없음'

    # --- 9) SHAP 값을 데이터프레임으로 ---
    # 행이 관측치, 열이 변수다. 한 칸이 '이 행에서 이 변수가 예측을 얼마나 밀었나' 이다.
    feat_names = list(x_explain.columns)
    shap_df = DataFrame(shap_2d, columns=feat_names, index=x_explain.index)

    # --- 10) 변수별 통계량 계산 ---
    mean_abs = shap_df.abs().mean().values      # 영향력 크기 (부호 무관)
    mean_s = shap_df.mean().values              # 평균 기여 (부호 = 방향)
    std_s = shap_df.std().values                # 기여의 흔들림

    direction = []      # 평균 기여의 부호를 방향 라벨로 바꾼다

    for v in mean_s:    # 변수마다
        if v > 0:      # 평균 기여가 양수
            direction.append('증가')
        elif v < 0:    # 평균 기여가 음수
            direction.append('감소')
        else:          # 평균 기여가 0
            direction.append('중립')

    # --- 11) 요약표로 조립 ---
    summary = DataFrame({
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_s,
        'std_shap': std_s,
        'direction': direction,
        # 변동계수 — 평균 기여보다 흔들림이 크면 비선형·상호작용을 의심한다
        'cv': std_s / (mean_abs + 1e-9),
    }, index=feat_names)

    summary['stability'] = np.where(summary['cv'] < 1, '안정적', '비선형/불안정')
    summary.index.name = 'Feature'
    summary = summary.sort_values('mean_abs_shap', ascending=False)     # 영향력 내림차순

    # --- 12) 비율과 누적 비율 ---
    total_abs = float(summary['mean_abs_shap'].sum())

    if total_abs > 0:    # 영향력 총합이 0 보다 큰 경우
        summary['ratio'] = summary['mean_abs_shap'] / total_abs         # 합 1 로 정규화
    else:                # 모든 기여가 0 인 경우
        summary['ratio'] = 0.0

    summary['cum_ratio'] = summary['ratio'].cumsum()                    # 내림차순 누적 비율

    # 컬럼 순서 정리 — 영향력 → 비율 → 누적 비율 → 해석용 컬럼
    summary = summary[['mean_abs_shap', 'ratio', 'cum_ratio',
                       'mean_shap', 'std_shap', 'direction', 'cv', 'stability']]

    # --- 13) 원시 데이터를 attrs 에 담기 ---
    # 후속 Dependence·Waterfall 이 재계산 없이 쓰는 원시 데이터를 담는다.
    # explainer 객체 자체는 담지 않는다 — 복사가 안 되는 객체라 슬라이스할 때 깨진다.
    summary.attrs['shap_values'] = shap_2d              # (n, f) 기여도 배열
    summary.attrs['data'] = x_explain                   # 모델공간 입력
    summary.attrs['expected_value'] = base_value        # base value (평균 예측)
    summary.attrs['feature_names'] = feat_names         # 변환 후 변수명
    summary.attrs['explainer_type'] = explainer_type    # 사용한 explainer 종류
    summary.attrs['model_class'] = model_class          # 모델 클래스명
    # (수정 전) 회귀 전용으로 정리하며 과제 구분 attrs 제거 (2026-09-17)
    # summary.attrs['task'] = task                        # 'regression' | 'classification'
    summary.attrs['output_space'] = output_space        # SHAP 값의 단위

    # --- 14) 분석 결과 저장 ---
    # 계산은 무겁고 시각화는 가볍다. 결과를 파일로 남겨 두면 shap_bar_plot·shap_beeswarm_plot 이
    # 모델도 explainer 도 없이 이 파일만으로 그래프를 다시 그릴 수 있다.
    workdir = Path(project_name) / workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # reg_tunes 가 붙여 둔 이름('xgb_tuned' 등)이 있으면 이어 쓰고, 없으면 모델 클래스명을 쓴다
    save_path = workdir / f'{getattr(estimator, "name_", model_class.lower())}_shap.pkl'
    save_model(summary, save_path)

    print(f'SHAP 분석 결과 저장 완료 → {save_path}')

    return summary


# --------------------------------------------------------
# SHAP 분석 결과를 영향력 순위 막대그래프로 시각화
# --------------------------------------------------------
def shap_bar_plot(summary, cum_ratio=0.95, palette=None, column_means=None,
                  title=None, xlabel=None, ylabel=None,
                  width=1280, height=None, save_path=None):
    """SHAP 분석 결과로 mean|SHAP| 순위 막대그래프를 그린다.

    Args:
        summary (DataFrame): shap_analysis 가 반환한(또는 my_ml.load_model 로 불러온) 결과표.
        cum_ratio (float): 채택·탈락을 가르는 누적 비율 (기본값: 0.95).
        palette (str or list): 색상 팔레트 (기본값: None).
        column_means (dict): `{변수명: 실제 의미}` 사전 (기본값: None → 변수명을 그대로 쓴다).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: None → 변수 개수에 맞춰 자동 계산).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, shap_analysis 결과가 아닌 경우.
    """
    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    if 'shap_values' not in summary.attrs:    # attrs 가 비어 있는 경우
        raise ValueError("shap_analysis 가 반환한 요약표가 아닙니다")

    # --- 2) 메타정보 꺼내기 ---
    # 계산부가 attrs 에 얹어 둔 값들이다. 저장·로드를 거쳐도 그대로 살아 있다.
    model_class = summary.attrs.get('model_class', '')          # 모델 클래스명
    # (수정 전) 분류용 꼬리표(클래스·단위). shap_analysis 가 회귀 전용이라 도달하지 않아 정리 (2026-09-17)
    # task = summary.attrs.get('task', 'regression')              # 회귀 | 분류
    # class_names = summary.attrs.get('class_names', [])          # 클래스 목록
    # used_class = summary.attrs.get('class_index', None)         # 설명 대상 클래스 위치
    # output_space = summary.attrs.get('output_space', '알 수 없음')  # SHAP 값의 단위
    #
    # # --- 3) 그래프에 붙일 꼬리표 ---
    # cls_tag = ''
    # unit_tag = ''
    #
    # if task == 'classification':    # 분류인 경우
    #     if class_names:             # 클래스 목록이 있는 경우 — 어느 클래스를 설명했는가
    #         cls_tag = f' · class={class_names[used_class]}'
    #
    #     unit_tag = f' [{output_space}]'

    # --- 4) 채택 개수(k) 결정 ---
    # 누적 비율이 cum_ratio 에 처음 도달하는 위치를 찾고 1을 더해, 경계에 걸친 변수까지 채택한다.
    
    # 전체 변수 개수
    n_all = len(summary)

    # 채택 개수
    k = int(np.searchsorted(summary['cum_ratio'].values, cum_ratio)) + 1

    # k 가 전체 변수 수를 넘으면 전체를 채택한다
    k = max(1, min(k, n_all))

    # 변수 하나가 막대 한 줄을 차지하므로 높이를 고정하면 막대가 서로 붙는다.
    # 여백·축(140) + 제목(80) + 변수당 40 으로 잡는다.
    if height is None:    # 높이를 주지 않은 경우
        plot_height = 220 + 40 * n_all
    else:                 # 사용자가 준 높이
        plot_height = height

    # --- 5) 그릴 데이터 만들기 ---
    plot_df = summary.reset_index()

    # 막대의 채택 여부를 문자열로 만들어 색 구분에 쓴다
    plot_df['채택여부'] = np.where(np.arange(n_all) < k, '채택', '탈락')

    # y축에 적을 이름 — 사전에 있으면 실제 의미로, 없으면 변수명 그대로
    if column_means is None:    # 의미 사전을 주지 않은 경우
        column_means = {}

    plot_df['의미'] = [column_means.get(f, f) for f in summary.index]

    # --- 6) 제목·축 라벨 ---
    if title is None:    # 제목을 주지 않은 경우
        # (수정 전) plot_title = f'SHAP Bar (mean|SHAP| · 누적 {cum_ratio:.0%} 채택): {model_class}{cls_tag}'  (2026-09-17)
        plot_title = f'SHAP Bar (mean|SHAP| · 누적 {cum_ratio:.0%} 채택): {model_class}'
    else:                # 사용자가 준 제목
        plot_title = title

    if xlabel is None:    # x축 라벨을 주지 않은 경우
        # (수정 전) plot_xlabel = f'mean|SHAP|{unit_tag}'  (2026-09-17)
        plot_xlabel = 'mean|SHAP|'
    else:                 # 사용자가 준 라벨
        plot_xlabel = xlabel

    if ylabel is None:    # y축 라벨을 주지 않은 경우
        plot_ylabel = '변수'
    else:                 # 사용자가 준 라벨
        plot_ylabel = ylabel

    # --- 7) 시각화 ---
    # 막대 오른쪽에 누적 비율을 적어야 해서 ax 를 직접 받는다.
    # 캔버스 생성·막대 그리기·표시는 모두 my_plot 에 맡긴다.
    fig, ax = my_plot.init(title=plot_title, width=width, height=plot_height,
                           xlabel=plot_xlabel, ylabel=plot_ylabel)
    my_plot.barplot(data=plot_df, x='mean_abs_shap', y='의미', hue='채택여부',
                    palette=palette, errorbar=None, ax=ax)

    vmax = float(plot_df['mean_abs_shap'].max())
    ax.set_xlim(0, vmax * 1.20)         # 누적 비율 글자가 들어갈 여백

    for i in range(n_all):              # '상위 k개가 영향력의 N% 를 설명' 을 읽게 한다
        value = float(plot_df['mean_abs_shap'].iloc[i])
        cum_text = f"{plot_df['cum_ratio'].iloc[i]:.0%}"
        ax.text(value + vmax * 0.01, i, cum_text,
                va='center', ha='left', fontsize=10, color='tab:red')

    my_plot.show(save_path=save_path)


# --------------------------------------------------------
# SHAP 분석 결과를 Beeswarm 으로 시각화
# --------------------------------------------------------
def shap_summary_plot(summary, cum_ratio=0.95, title=None, xlabel=None,
                       column_means=None, width=1280, height=None, save_path=None):
    """SHAP 분석 결과로 Beeswarm 그래프를 그린다.

    Args:
        summary (DataFrame): shap_analysis 가 반환한(또는 my_ml.load_model 로 불러온) 결과표.
        cum_ratio (float): 그릴 변수를 고르는 누적 비율 (기본값: 0.95).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        column_means (dict): `{변수명: 실제 의미}` 사전 (기본값: None → 변수명을 그대로 쓴다).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: None → 채택 변수 개수에 맞춰 자동 계산).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, shap_analysis 결과가 아닌 경우.
    """
    # shap 은 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
    import shap

    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    if 'shap_values' not in summary.attrs:    # attrs 가 비어 있는 경우
        raise ValueError("shap_analysis 가 반환한 요약표가 아닙니다 (attrs['shap_values'] 없음)")

    # --- 2) 원시 데이터·메타정보 꺼내기 ---
    shap_2d = summary.attrs['shap_values']                      # (n, f) 기여도 배열
    x_explain = summary.attrs['data']                           # 모델공간 입력 (점의 색)
    model_class = summary.attrs.get('model_class', '')          # 모델 클래스명
    # (수정 전) 분류용 꼬리표(클래스·단위). shap_analysis 가 회귀 전용이라 도달하지 않아 정리 (2026-09-17)
    # task = summary.attrs.get('task', 'regression')              # 회귀 | 분류
    # class_names = summary.attrs.get('class_names', [])          # 클래스 목록
    # used_class = summary.attrs.get('class_index', None)         # 설명 대상 클래스 위치
    # output_space = summary.attrs.get('output_space', '알 수 없음')  # SHAP 값의 단위
    #
    # # --- 3) 그래프에 붙일 꼬리표 ---
    # cls_tag = ''
    # unit_tag = ''
    #
    # if task == 'classification':    # 분류인 경우
    #     if class_names:             # 클래스 목록이 있는 경우 — 어느 클래스를 설명했는가
    #         cls_tag = f' · class={class_names[used_class]}'
    #
    #     unit_tag = f' [{output_space}]'

    # --- 4) 시각화 할 변수 결정 ---
    # 요약표는 이미 mean_abs_shap 내림차순이고 누적 비율까지 들어 있다.
    # 표에서 바로 읽어 채택 개수를 정하므로 막대그래프의 '채택' 변수와 언제나 일치한다.
    n_all = len(summary)                                                     # 전체 변수 개수
    k = int(np.searchsorted(summary['cum_ratio'].values, cum_ratio)) + 1     # 채택 개수
    k = max(1, min(k, n_all))

    keep = list(summary.index[:k])    # 시각화 할 변수 결정 (영향력 내림차순)

    # shap_2d 의 열 순서는 x_explain 의 열 순서다. 변수명을 열 위치로 바꿔 같이 잘라낸다.
    pos = [x_explain.columns.get_loc(f) for f in keep]
    shap_2d = shap_2d[:, pos]
    x_explain = x_explain[keep]

    # shap 은 열 이름을 그대로 줄 이름으로 쓴다 — 사전이 있으면 실제 의미로 바꿔 준다
    if column_means:    # 의미 사전을 준 경우
        x_explain = x_explain.rename(columns=column_means)

    # --- 5) 그래프 높이와 제목 ---
    # 변수 하나가 점 한 줄을 차지하므로 높이를 고정하면 줄이 서로 붙는다.
    # 여백·축(140) + 제목(80) + 변수당 40 으로 잡는다.
    if height is None:    # 높이를 주지 않은 경우
        plot_height = 220 + 40 * k
    else:                 # 사용자가 준 높이
        plot_height = height

    if title is None:    # 제목을 주지 않은 경우
        # (수정 전) plot_title = f'SHAP Summary (Beeswarm · 누적 {cum_ratio:.0%} 채택): {model_class}{cls_tag}'  (2026-09-17)
        plot_title = f'SHAP Summary (Beeswarm · 누적 {cum_ratio:.0%} 채택): {model_class}'
    else:                # 사용자가 준 제목
        plot_title = title

    # --- 6) 시각화 ---
    my_plot.init(title=plot_title, width=width, height=plot_height)

    # shap 은 max_display 를 주지 않으면 상위 20개만 그리므로 채택 개수를 직접 넘긴다.
    # plot_size=None 을 줘야 shap 이 캔버스 크기를 제 기본값으로 되돌리지 않는다.
    shap.summary_plot(shap_2d, x_explain, plot_type='dot',
                      max_display=k, show=False, plot_size=None)

    if xlabel is None:    # x축 라벨을 주지 않은 경우
        # (수정 전) plt.xlabel(f'SHAP value{unit_tag}')  (2026-09-17)
        plt.xlabel('SHAP value')                # shap 이 붙인 라벨을 덮어쓴다
    else:                 # 사용자가 준 라벨
        plt.xlabel(xlabel)

    my_plot.show(save_path=save_path)


# --------------------------------------------------------
# 변수값과 SHAP 값의 관계를 그리는 Dependence Plot
# --------------------------------------------------------
def shap_dependence_plot(summary, cum_ratio=0.95, title=None, column_means=None,
                         width=1280, height=640, save_path=None):
    """SHAP 분석 결과로 Dependence Plot 을 그린다 (변수값과 기여의 관계).

    Args:
        summary (DataFrame): shap_analysis 가 반환한(또는 my_ml.load_model 로 불러온) 결과표.
        cum_ratio (float): 주변수로 채택할 누적 비율 (기본값: 0.95).
        title (str): 그래프 제목 (기본값: None). 주면 뒤에 변수쌍이 붙는다.
        column_means (dict): `{변수명: 실제 의미}` 사전 (기본값: None → 변수명을 그대로 쓴다).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None). 파일명 뒤에 변수명이 붙는다.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, shap_analysis 결과가 아닌 경우.
    """
    # shap 은 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
    import shap

    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    if 'shap_values' not in summary.attrs:    # attrs 가 비어 있는 경우
        raise ValueError("shap_analysis 가 반환한 요약표가 아닙니다")

    # --- 2) 원시 데이터 꺼내기 ---
    shap_2d = summary.attrs['shap_values']    # (n, f) 기여도 배열
    x_df = summary.attrs['data']              # 모델공간 입력 (x축·점의 색)

    # (수정 전) 분류용 단위 꼬리표. shap_analysis 가 회귀 전용이라 도달하지 않아 정리 (2026-09-17)
    # unit_tag = ''
    #
    # if summary.attrs.get('task') == 'classification':
    #     unit_tag = f" [{summary.attrs.get('output_space', '알 수 없음')}]"


    # --- 3) 주변수 고르기 ---
    # 요약표는 이미 mean_abs_shap 내림차순이고 누적 비율까지 들어 있다.
    # 표에서 바로 읽어 채택 개수를 정하므로 막대그래프의 '채택' 변수와 언제나 일치한다.
    k = int(np.searchsorted(summary['cum_ratio'].values, cum_ratio)) + 1
    main_features = list(summary.index[:k])

    # --- 4) 주변수마다 상호작용이 가장 강한 짝 찾기 ---
    partners = []    # 상호작용 짝을 저장할 리스트

    for f in main_features:              # 주변수마다
        fi = x_df.columns.get_loc(f)     # 변수명의 열 인덱스 찾기
        # 구간 안에 남아 있는 SHAP 값의 흩어짐을 가장 잘 설명하는 순서로 돌려준다
        inter = shap.utils.approximate_interactions(fi, shap_2d, x_df)
        # 0번째 항목 = 가장 강한 짝
        partners.append((f, x_df.columns[inter[0]]))

    # --- 5) 시각화 — 변수쌍마다 한 장씩 ---
    if column_means is None:    # 의미 사전을 주지 않은 경우
        column_means = {}       # 변수명을 그대로 제목에 쓴다

    for f, partner in partners:    # 변수쌍마다
        # 사전에 있으면 실제 의미로, 없으면 변수명 그대로 적는다
        pair_tag = f'{column_means.get(f, f)}  ×  {column_means.get(partner, partner)}'

        if title is None:    # 제목을 주지 않은 경우
            # (수정 전) plot_title = f'SHAP Dependence: {pair_tag}{unit_tag}'  (2026-09-17)
            plot_title = f'SHAP Dependence: {pair_tag}'
        else:                # 사용자가 준 제목 — 장마다 변수쌍을 덧붙여 구분한다
            plot_title = f'{title} — {pair_tag}'

        fig, ax = my_plot.init(title=plot_title, width=width, height=height)

        shap.dependence_plot(f, shap_2d, x_df,
                             interaction_index=partner, ax=ax, show=False)

        # (0,0)을 지나는 가로 직선
        ax.axhline(0, color='#990000', linestyle=':', linewidth=2)

        if save_path is None:    # 저장하지 않는 경우
            path = None
        else:                    # 여러 장이므로 파일명 뒤에 변수명을 덧붙인다
            p = Path(save_path)
            path = str(p.with_name(f'{p.stem}_{f}{p.suffix}'))

        my_plot.show(save_path=path)


# --------------------------------------------------------
# 개별 관측치의 기여를 분해하는 Waterfall Plot
# --------------------------------------------------------
def shap_waterfall_plot(summary, index, cum_ratio=0.95, title=None, label=None,
                        column_means=None, width=1280, height=None, save_path=None):
    """SHAP 분석 결과로 관측치 한 건의 Waterfall Plot 을 그린다 (개별 사례 기여 분해).

    Args:
        summary (DataFrame): shap_analysis 가 반환한(또는 my_ml.load_model 로 불러온) 결과표.
        index (int): 설명할 관측치의 행 위치. 인덱스 라벨이 아니라 0부터 세는 순번이다.
        cum_ratio (float): 막대로 보여줄 변수 개수를 정하는 누적 비율 (기본값: 0.95).
        title (str): 그래프 제목 (기본값: None).
        label (str): 이 관측치를 부르는 이름 (기본값: None). 기본 제목의 분위 자리에 대신 적는다.
        column_means (dict): `{변수명: 실제 의미}` 사전 (기본값: None → 변수명을 그대로 쓴다).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: None → 변수 개수에 맞춰 자동 계산).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: 설명에 사용된 관측치 한 행 (quantile·pred 컬럼이 앞에 붙는다).

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, shap_analysis 결과가 아닌 경우.
    """
    # shap 은 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
    import shap

    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    if 'shap_values' not in summary.attrs:    # attrs 가 비어 있는 경우
        raise ValueError("shap_analysis 가 반환한 요약표가 아닙니다")

    # --- 2) 원시 데이터·메타정보 꺼내기 ---
    shap_values = np.asarray(summary.attrs['shap_values'], dtype=float)  # (n, f) 기여도 배열
    data = summary.attrs['data']                                  # 막대 왼쪽의 변수값
    base_value = float(summary.attrs['expected_value'])           # base value (평균 예측)
    feat_names = list(summary.attrs['feature_names'])             # 변환 후 변수명
    # (수정 전) 분류용 꼬리표(클래스·단위). shap_analysis 가 회귀 전용이라 도달하지 않아 정리 (2026-09-17)
    # task = summary.attrs.get('task', 'regression')                # 회귀 | 분류
    # class_names = summary.attrs.get('class_names', [])            # 클래스 목록
    # used_class = summary.attrs.get('class_index', None)           # 설명 대상 클래스 위치
    # output_space = summary.attrs.get('output_space', '알 수 없음')  # SHAP 값의 단위
    #
    # cls_tag = ''
    # unit_tag = ''
    #
    # if task == 'classification':    # 분류인 경우
    #     if class_names:    # 클래스 목록이 있는 경우
    #         cls_tag = f' · class={class_names[used_class]}'
    #
    #     unit_tag = f' [{output_space}]'


    # --- 3) 행별 예측값 복원 ---
    # SHAP 의 가산성 — 행별 예측 = base value + 그 행의 SHAP 합
    pred = base_value + shap_values.sum(axis=1)
    total = len(pred)

    # --- 4) 설명할 관측치 ---
    i = int(index)

    # 예측값 분포에서 몇 % 지점인지 — 자기보다 작은 예측이 몇 개인가로 센다
    if total > 1:    # 행이 둘 이상인 경우
        quantile = float(np.searchsorted(np.sort(pred), pred[i]) / (total - 1))
    else:            # 행이 하나뿐인 경우
        quantile = 0.0

    # 그린 그래프를 표로도 확인할 수 있게 한 행짜리 결과를 만든다
    pick = data.iloc[[i]].copy()
    pick.insert(0, 'pred', pred[i])          # 복원한 예측값
    pick.insert(0, 'quantile', quantile)     # 예측값 분포에서의 위치

    # --- 5) 막대로 보여줄 개수와 높이 ---
    n_all = len(feat_names)     # 전체 변수 개수

    # 막대그래프와 같은 기준으로 채택 개수를 읽는다.
    # shap 은 max_display 개까지만 막대로 그리고 나머지는 'N other features' 한 줄로 합치므로
    # 채택된 k 개를 모두 보여주려면 k + 1 을 넘긴다.
    # (다만 기여 순서는 행마다 다르므로 여기 찍히는 k 개는 그 행 기준 상위 k 개다.)
    k = int(np.searchsorted(summary['cum_ratio'].values, cum_ratio)) + 1
    max_display = min(k + 1, n_all)

    # 변수 하나가 막대 한 줄을 차지하므로 높이를 고정하면 막대가 서로 붙는다.
    # 여백·축(140) + 제목(80) + 줄당 40 으로 잡는다 — shap_bar_plot 과 같은 기준이다.
    if height is None:    # 높이를 주지 않은 경우
        plot_height = 220 + 40 * max_display
    else:                 # 사용자가 준 높이
        plot_height = height

    # --- 6) 제목 ---
    if title is None:    # 제목을 주지 않은 경우
        # 부르는 이름을 주면 분위 대신 그 이름을 적는다 (예: ... (중앙값))
        if label is None:    # 이름을 주지 않은 경우
            pick_tag = f'분위 {quantile:.0%}'
        else:                # 사용자가 준 이름
            pick_tag = label

        # (수정 전) f'pred≈{pred[i]:.4g}{unit_tag} ({pick_tag}){cls_tag}' — 분류용 꼬리표 제거 (2026-09-17)
        plot_title = (f'SHAP Waterfall: obs#{i} — '
                      f'pred≈{pred[i]:.4g} ({pick_tag})')
    else:                # 사용자가 준 제목
        plot_title = title

    # --- 7) 시각화 ---
    # 사전에 있으면 실제 의미로, 없으면 변수명 그대로 막대 옆에 적는다
    if column_means is None:    # 의미 사전을 주지 않은 경우
        column_means = {}

    plot_names = [column_means.get(f, f) for f in feat_names]

    # shap 이 요구하는 설명 객체 — 기여도·기준값·실제 변수값을 함께 담는다
    expl = shap.Explanation(
        values=shap_values[i],
        base_values=base_value,
        data=data.iloc[i].values,
        feature_names=plot_names,
    )

    fig, _ = my_plot.init(width=width, height=plot_height)

    shap.plots.waterfall(expl, max_display=max_display, show=False)

    # shap.plots.waterfall 은 ax 를 받지 않고, 
    # 그리는 도중 캔버스 크기를 8 × (행수) 인치로 되돌린다. 
    # 그래서 그래프를 그린 뒤에 크기를 다시 잡고 제목을 붙인다.
    # (matplotlib 은 인치 단위, my_plot 은 픽셀 단위라 100 으로 나눈다)
    fig.set_size_inches(width / 100, plot_height / 100)
    plt.title(plot_title, fontsize=18, fontweight=500, pad=25)

    my_plot.show(save_path=save_path)

    return pick


# --------------------------------------------------------
# 예측값 분포의 대표 지점마다 Waterfall Plot 을 그리는 기능
# --------------------------------------------------------
def shap_waterfall_stats_plot(summary, cum_ratio=0.95, title=None, column_means=None,
                              width=1280, height=None, save_path=None):
    """예측값의 대표 통계마다 Waterfall Plot 을 그린다 (최소・1사분위・중앙값・평균・3사분위・최대).

    통계값과 예측이 정확히 일치하는 행은 없을 수 있으므로(짝수 개일 때의 중앙값·평균 등)
    통계값과 차이가 가장 작은 행을 대표로 골라 shap_waterfall_plot 을 반복해서 부른다.

    Args:
        summary (DataFrame): shap_analysis 가 반환한(또는 my_ml.load_model 로 불러온) 결과표.
        cum_ratio (float): 막대로 보여줄 변수 개수를 정하는 누적 비율 (기본값: 0.95).
        title (str): 그래프 제목 (기본값: None). 주면 뒤에 통계 이름이 붙는다.
        column_means (dict): `{변수명: 실제 의미}` 사전 (기본값: None → 변수명을 그대로 쓴다).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: None → 변수 개수에 맞춰 자동 계산).
        save_path (str): 그래프 저장 경로 (기본값: None). 파일명 뒤에 행 위치가 붙는다.

    Returns:
        DataFrame: 대표로 뽑힌 행들 (stat・quantile・pred 컬럼이 앞에 붙는다).

    Raises:
        ValueError: cum_ratio 가 (0, 1] 범위 밖이거나, shap_analysis 결과가 아닌 경우.
    """
    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:    # (0, 1] 범위를 벗어난 경우
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    if 'shap_values' not in summary.attrs:    # attrs 가 비어 있는 경우
        raise ValueError("shap_analysis 가 반환한 요약표가 아닙니다 (attrs['shap_values'] 없음)")

    # --- 2) 행별 예측값 복원 ---
    # SHAP 의 가산성 — 행별 예측 = base value + 그 행의 SHAP 합
    shap_values = np.asarray(summary.attrs['shap_values'], dtype=float)
    pred = float(summary.attrs['expected_value']) + shap_values.sum(axis=1)

    # --- 3) 예측값 분포의 대표 통계 ---
    stats = {
        '최소값':  float(pred.min()),
        '1사분위': float(np.percentile(pred, 25)),
        '중앙값':  float(np.median(pred)),
        '평균':    float(pred.mean()),
        '3사분위': float(np.percentile(pred, 75)),
        '최대값':  float(pred.max()),
    }

    # --- 4) 통계값마다 가장 가까운 행 ---
    # 두 통계가 같은 행을 가리키면 같은 그래프를 두 번 그리게 되므로 이름만 합친다.
    picks = {}    # {행 위치: 그 행이 대표하는 통계 이름}

    for name, value in stats.items():    # 통계마다
        i = int(np.argmin(np.abs(pred - value)))    # 차이가 가장 작은 행

        if i in picks:    # 다른 통계가 이미 집은 행인 경우
            picks[i] = f'{picks[i]}・{name}'
        else:             # 처음 집는 행인 경우
            picks[i] = name

    # --- 5) 대표 행마다 한 장씩 ---
    rows = []

    for i, name in picks.items():    # 대표 행마다
        if save_path is None:    # 저장하지 않는 경우
            path = None
        else:                    # 여러 장이므로 파일명 뒤에 행 위치를 덧붙인다
            p = Path(save_path)
            path = str(p.with_name(f'{p.stem}_obs{i}{p.suffix}'))

        if title is None:    # 제목을 주지 않은 경우 — 통계 이름만 넘기고 나머지는 맡긴다
            plot_title = None
        else:                # 사용자가 준 제목 — 장마다 통계 이름을 덧붙여 구분한다
            plot_title = f'{title} — {name}'

        rows.append(shap_waterfall_plot(summary, index=i, cum_ratio=cum_ratio,
                                        title=plot_title, label=name,
                                        column_means=column_means,
                                        width=width, height=height, save_path=path))

    # --- 6) 뽑힌 행을 한 표로 ---
    pick = concat(rows)
    pick.insert(0, 'stat', list(picks.values()))    # 어떤 통계의 대표인가

    return pick
