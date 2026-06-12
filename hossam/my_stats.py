from math import sqrt
from pandas import DataFrame
from scipy.stats import t
from scipy.stats import normaltest, bartlett, levene

def ci(data, column=None, clevel=0.95):
    """
    주어진 데이터에 대한 모평균의 신뢰구간을 계산하는 함수

    Args:
        data (Series | list | ndarray | DataFrame): 연속형 데이터 또는 데이터프레임
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None)
        clevel (float): 신뢰수준 (기본값: 0.95)

    Returns:
        tuple: (신뢰구간 하한, 신뢰구간 상한)
    """
    # 데이터프레임 + 컬럼명 형태로 전달된 경우 해당 컬럼만 추출
    if column is not None:
        data = data[column]

    n = len(data)                           # 표본 크기
    dof = n - 1                             # 자유도
    sample_mean = data.mean()               # 표본 평균
    sample_std = data.std()                 # 표본 표준편차
    sample_std_error = sample_std / sqrt(n) # 표준 오차

    # 신뢰구간을 계산하여 리턴한다.
    return t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)



def test_assumptions(data, columns=None, alpha=0.05, center="median"):
    """
    가설검정의 가정(정규성, 등분산성)을 일괄적으로 검정하여 결과표를 반환하는 함수

    각 변수에 대해 정규성 검정(normaltest)을 수행하고, 변수가 두 개 이상인 경우
    등분산성 검정을 수행한다. 이때 모든 변수가 정규성을 충족하면 Bartlett 검정을,
    하나라도 충족하지 못하면 Levene 검정을 선택적으로 사용한다.

    Args:
        data (DataFrame): 검정 대상이 되는 데이터프레임
        columns (list): 검정에 사용할 컬럼명 목록 (기본값: None → 수치형 컬럼 전체)
        alpha (float): 유의수준 (기본값: 0.05)
        center (str): Levene 검정 시 사용할 중심 경향값 (기본값: "median")

    Returns:
        DataFrame: field를 인덱스로 하는 검정 결과표
                   (test, statistic, p-value, result 컬럼 포함)
    """
    # 검정에 사용할 컬럼 결정 (지정하지 않으면 수치형 컬럼 전체 사용)
    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()

    report = []         # 결과를 누적할 리스트
    normal_dist = True  # 모든 변수가 정규성을 충족하는지 여부

    # 1) 각 변수에 대한 정규성 검정
    for c in columns:
        s, p = normaltest(data[c])
        normalize = p >= alpha

        report.append({
            "field": c,
            "test": "normaltest",
            "statistic": s,
            "p-value": p,
            "result": normalize
        })

        normal_dist = normal_dist and normalize

    # 2) 변수가 두 개 이상인 경우 등분산성 검정
    if len(columns) > 1:
        # 각 컬럼을 실수형으로 변환하여 리스트로 추출 (Bartlett은 실수형 필요)
        samples = [data[c].astype("float") for c in columns]

        if normal_dist:
            # 모든 변수가 정규성을 충족 → Bartlett 검정
            name = "Bartlett"
            s, p = bartlett(*samples)
        else:
            # 하나라도 정규성을 충족하지 못함 → Levene 검정 (중앙값 기준)
            name = "Levene"
            s, p = levene(*samples, center=center)

        report.append({
            "field": name,
            "test": "equal_var",
            "statistic": s,
            "p-value": p,
            "result": p >= alpha
        })

    # 결과표 리턴
    return DataFrame(report).set_index("field")