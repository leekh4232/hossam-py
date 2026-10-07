import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import my_stats

class VIFSelector(BaseEstimator, TransformerMixin):
    """
    VIF(Variance Inflation Factor) 기반 다중공선성 제거기

    fit 단계에서 결측치를 제거한 후 VIF 임계값을 초과하는 변수를
    VIF가 가장 높은 변수부터 반복적으로 제거한다.
    """

    def __init__(self, threshold=10.0, check_cols=None):
        """VIF 임계값과 검사 대상 열을 개발자가 직접 지정
        (생략시 기본값 10.0 / 전체 열)

        Args:
            threshold (float): VIF 임계값 (기본값: 10.0)
            check_cols (list or None): VIF를 계산할 열 목록. None이면 모든 열
        """
        self.threshold = threshold
        self.check_cols = check_cols

    def fit(self, X, y=None):
        """개발자가 직접 부르지 않는다. 정의만 해두면 sklearn이 알아서 부른다.
        VIF 임계값을 초과하는 변수를 제거할 때까지 반복적으로 VIF를 계산하여 삭제 대상 컬럼을 결정한다.(실제 제거는 안함)

        Args:
            X (DataFrame or ndarray): sklearn이 넘겨주는 학습용 2차원 데이터
            y (array-like): 사용하지 않음. sklearn이 항상 넘기므로 받아만 둔다

        Returns:
            VIFSelector: 자기 자신. sklearn이 다음 단계로 넘기려면 반드시 self를 반환해야 한다
        """
        # --- 1) 입력 검증 및 DataFrame 변환 (DataFrame 이 아니면 ndarray 만 허용) ---
        if not isinstance(X, pd.DataFrame):
            if not isinstance(X, np.ndarray):
                raise TypeError(f'X 는 pandas DataFrame 또는 numpy ndarray 여야 합니다. '
                                f'전달된 타입: {type(X).__name__}')

            if X.ndim != 2:
                raise ValueError('X 는 2차원 데이터여야 합니다.')

            # ndarray 에는 컬럼명이 없으므로 숫자로 생성해서 붙인다
            auto_cols = []
            for i in range(X.shape[1]):
                auto_cols.append(f'col_{i}')

            X = pd.DataFrame(X, columns=auto_cols)

        # --- 2) 결측치 제거 (VIF 는 회귀 계산이므로 결측치가 있으면 안 된다) ---
        # 학습 데이터에서만 지우며, transform 에서는 지우지 않는다
        X_clean = X.dropna()

        if X_clean.empty:
            raise ValueError('결측치 제거 결과 남은 데이터가 없습니다.')

        # --- 3) 입력 컬럼 이름 기록 (sklearn 표준 속성) ---
        self.feature_names_in_ = X.columns.tolist()

        # --- 4) VIF 를 계산할 대상 컬럼 결정 (지정이 없으면 전체 컬럼) ---
        if self.check_cols:
            self.vif_cols_ = self.check_cols
        else:
            self.vif_cols_ = self.feature_names_in_

        # 원본을 보존하기 위해 대상 컬럼만 복사본으로 작업
        X_vif = X_clean[self.vif_cols_].copy()

        # --- 5) 반복 제거에 사용할 저장소 준비 ---
        self.drop_cols_ = []    # 제거하기로 확정된 컬럼 이름
        history = []            # 라운드별 VIF 표
        round_no = 0            # 반복 단계 카운터

        # --- 6) 다중 공선성이 사라질 때까지 VIF 가 가장 큰 변수를 하나씩 제거 ---
        while X_vif.shape[1] > 0:
            round_no += 1

            # --- 6-1) 남아있는 컬럼들의 VIF 를 계산해 표로 정리 ---
            # 완전한 선형관계로 R²=1 이면 divide-by-zero 경고가 나므로 잠시 꺼둔다
            old_err = np.seterr(divide='ignore', invalid='ignore')

            # VIF 계산은 my_stats.compute_vif 를 그대로 재사용한다
            vdf = my_stats.compute_vif(X_vif, columns=list(X_vif.columns))

            # 경고 설정을 원래대로 되돌린다
            np.seterr(divide=old_err['divide'], invalid=old_err['invalid'])

            # compute_vif 는 변수명을 인덱스로, VIF 내림차순(동점은 이름순)으로 돌려준다
            round_df = vdf.reset_index()
            round_df.columns = ['Variable', 'VIF']

            # nan: 분산이 0 인 상수 컬럼 등으로 VIF 계산 자체가 실패한 경우.
            # 최악으로 취급해 가장 먼저 제거되도록 100만(1e6)을 채운다 -> "사실상 무한대"를 뜻하는 상한값
            # VIF 는 보통 10만 넘어도 심각한 다중공선성이므로 100만이면 충분히 크다
            round_df['VIF'] = round_df['VIF'].fillna(1e6)

            # inf: 다른 컬럼들로 완벽히 설명되는(R²=1) 경우. 그대로 두면 표 출력과
            # 이후 계산이 깨지므로 100만으로 눌러 유한한 값으로 만든다
            round_df['VIF'] = round_df['VIF'].clip(upper=1e6)

            # nan 이었던 행은 compute_vif 의 정렬에서 맨 아래에 놓였다가 방금 1e6 이 되었으므로
            # 다시 내림차순으로 정렬해야 첫 행이 진짜 최대값이 된다
            # (수정 전) 재정렬 없이 첫 행을 읽어 상수 컬럼(VIF nan→1e6) 이 끝까지 제거되지 않았다 (2026-09-23)
            round_df = (round_df.sort_values(by='VIF', ascending=False, kind='stable')
                                .reset_index(drop=True))

            # --- 6-2) 가장 VIF 가 큰 변수를 찾고 이번 라운드 표를 기록 ---
            # 재정렬했으므로 첫 번째 행이 곧 최대값이다
            max_col = round_df.iloc[0]['Variable']
            max_vif = round_df.iloc[0]['VIF']

            round_df.insert(0, 'Round', round_no)
            round_df['Removed'] = False

            # --- 6-3) 종료 조건을 확인하고, 아니라면 최대 VIF 변수를 제거 ---
            # 가장 큰 VIF 도 임계값 이하라면 더 제거할 것이 없다
            if max_vif <= self.threshold:
                history.append(round_df)
                break

            round_df.loc[round_df['Variable'] == max_col, 'Removed'] = True
            history.append(round_df)

            X_vif = X_vif.drop(columns=[max_col])
            self.drop_cols_.append(max_col)

        # --- 7) 라운드별 표를 하나로 합쳐 보관 ---
        self.vif_history_ = pd.concat(history, ignore_index=True)

        # --- 8) sklearn 규약에 따라 자기 자신을 반환 ---
        return self

    def transform(self, X):
        """sklearn에 의해서 호출되는 기능이다. 개발자가 직접 부르지 않는다.

        pipeline의 fit / predict / score 과정마다. fit에서 확정한 drop_cols_만 떨어내며,
        train·test에 같은 열 목록을 적용하므로 데이터 누수가 생기지 않는다.

        Args:
            X (DataFrame or ndarray): sklearn이 넘겨주는 2차원 데이터

        Returns:
            DataFrame: 다중공선성 열을 제거한 데이터. 다음 단계(스케일러·모델)의 입력이 된다
        """
        # fit 여부 확인
        check_is_fitted(self, ['drop_cols_', 'feature_names_in_'])
        
        # 입력 검증 및 변환 (ndarray -> DataFrame 자동 변환)
        if not isinstance(X, pd.DataFrame):
            if not isinstance(X, np.ndarray):
                raise TypeError(f'X 는 pandas DataFrame 또는 numpy ndarray 여야 합니다. '
                                f'전달된 타입: {type(X).__name__}')
            if X.ndim != 2:
                raise ValueError('X 는 2차원 데이터여야 합니다.')

            # fit에서 저장해 둔 이름을 그대로 붙인다
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # 피처 일치성 확인
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError(
                f'X 의 컬럼이 학습 시점과 다릅니다. '
                f'전달된 컬럼: {X.columns.tolist()} / 학습 컬럼: {self.feature_names_in_}'
            )

        return X.drop(columns=self.drop_cols_)

    def get_feature_names_out(self, input_features=None):
        """sklearn이 호출한다. 개발자가 직접 사용하지 않는다. (BaseEstimator 규약)

        sklearn이 현재 작업중인 데이터의 컬럼명을 물을 때 호출된다.
        이 변환기는 열을 삭제하므로 drop_cols_를 뺀 나머지 이름만 돌려준다.

        Args:
            input_features (array-like): sklearn이 지정한 피처 이름. 열 목록은 fit 결과로
                이미 정해지므로 사용하지 않는다

        Returns:
            np.ndarray: 제거되고 남은 출력 피처 이름 배열
        """
        check_is_fitted(self, ['drop_cols_', 'feature_names_in_'])
        kept = []
        for c in self.feature_names_in_:
            if c not in self.drop_cols_:
                kept.append(c)

        return np.asarray(kept)

    
    