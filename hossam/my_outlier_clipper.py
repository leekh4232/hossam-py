import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    수치형 컬럼의 이상치를 IQR 또는 Z-score 기반 경계값으로 클리핑한다.
    학습 단계에서 컬럼별로 상/하한 경계를 통해 해당 범위 밖의 값을 경계값으로 잘라낸다.

    Args:
        method (str): 이상치 판단 방식. 'iqr' 또는 'zscore' (기본값: 'iqr')
            - 'iqr'   : [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 범위로 클리핑
            - 'zscore': [mean - 3*std, mean + 3*std] 범위로 클리핑
    """

    def __init__(self, method='iqr'):
        """이상치 판단 방식을 개발자가 직접 지정한다. (생략시 기본값 'iqr')

        Args:
            method (str): 이상치 판단 방식. 'iqr' 또는 'zscore'
        """
        self.method = method

    def fit(self, X, y=None):
        """개발자가 직접 부르지 않는다. 정의만 해두면 sklearn이 알아서 부른다.
        수행 준비과정이다. 
        이상치 판단을 위한 상/하한 경계값을 학습 데이터로부터 계산한다.

        Args:
            X (array-like): sklearn이 넘겨주는 학습용 2차원 수치형 데이터
            y (array-like): 사용하지 않음. sklearn이 항상 넘기므로 받아만 둔다

        Returns:
            OutlierClipper: 자기 자신. 
            sklearn이 다음 단계로 넘기려면 반드시 self를 반환해야 한다
        """
        # 입력 피처 이름 저장 (set_output(transform='pandas') 지원용)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = np.asarray(
                [f'x{i}' for i in range(np.asarray(X).shape[1])]
            )

        # 입력 검증 및 변환 (ndarray -> DataFrame 자동 변환)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got ndim={X.ndim}")

        if self.method == 'iqr':
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - 1.5 * iqr
            self.upper_ = q3 + 1.5 * iqr
        elif self.method == 'zscore':
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            self.lower_ = mean - 3.0 * std
            self.upper_ = mean + 3.0 * std
        else:
            raise ValueError(
                f"method must be 'iqr' or 'zscore', got {self.method!r}"
            )

        return self

    def transform(self, X):
        """skelarn에 의해서 호출되는 기능이다. 개발자가 직접 부르지 않는다.
        실제 이 클래스의 목적을 수행한다.
        -> 학습 단계에서 계산된 상/하한 경계값을 이용해 입력 배열의 이상치를 클리핑한다.

        Args:
            X (array-like): sklearn이 넘겨주는 2차원 수치형 데이터

        Returns:
            np.ndarray: 클리핑된 배열. 다음 단계(스케일러·모델)의 입력이 된다
        """
        check_is_fitted(self, ['lower_', 'upper_'])
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        """sklearn이 호출한다. 개발자가 직접 사용하지 않는다. (BaseEstimator 규약)

        sklearn이 현재 작업중인 데이터의 컬럼명을 물을 때 호출된다.
        클리핑은 컬럼 수를 바꾸지 않으므로 입력 피처 이름을 그대로 돌려준다.

        Args:
            input_features (array-like): sklearn이 지정한 피처 이름. 
                None이면 fit 때 저장한 이름 사용

        Returns:
            np.ndarray: 출력 피처 이름 배열
        """
        check_is_fitted(self, ['feature_names_in_'])
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features)
