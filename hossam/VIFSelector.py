import statsmodels.api as sm
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------
# VIF 기반 다중공선성 제거기
# --------------------------------------------------------
class VIFSelector(BaseEstimator, TransformerMixin):
    """
    VIF(Variance Inflation Factor) 기반 다중공선성 제거기

    Args:
        threshold (float): VIF 임계값 (기본값: 10.0)
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