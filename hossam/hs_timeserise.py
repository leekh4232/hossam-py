# -*- coding: utf-8 -*-
# -------------------------------------------------------------
from typing import Callable
import numpy as np
import datetime as dt
import concurrent.futures as futures

# -------------------------------------------------------------
from pandas import DataFrame, Series, date_range

# -------------------------------------------------------------
import seaborn as sb
from matplotlib import pyplot as plt

# -------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose         # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------------------------------------
from pmdarima.arima import auto_arima

# -------------------------------------------------------------
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

# -------------------------------------------------------------
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------------------
from .hs_util import pretty_table
from .hs_plot import lineplot


# ===================================================================
# 차분
# ===================================================================
def diff(
    data: DataFrame,
    yname: str,
    plot: bool = True,
    max_diff: int | None = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DataFrame:
    """시계열 데이터의 정상성을 검정하고 차분을 통해 정상성을 확보한다.

    ADF(Augmented Dickey-Fuller) 검정을 사용하여 시계열 데이터의 정상성을 확인한다.
    정상성을 만족하지 않으면(p-value > 0.05) 차분을 반복 수행하여 정상 시계열로 변환한다.
    ARIMA 모델링 전 필수적인 전처리 과정이다.

    Args:
        data (DataFrame): 시계열 데이터프레임. 인덱스가 datetime 형식이어야 한다.
        yname (str): 정상성 검정 및 차분을 수행할 대상 컬럼명.
        plot (bool, optional): 각 차분 단계마다 시계열 그래프를 표시할지 여부.
            기본값은 True.
        max_diff (int | None, optional): 최대 차분 횟수 제한. None이면 정상성을 만족할 때까지 반복.
            과도한 차분을 방지하기 위해 설정 권장. 기본값은 None.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        DataFrame: 정상성을 만족하는 차분된 데이터프레임.

    Notes:
        - ADF 검정의 귀무가설: 시계열이 단위근(unit root)을 가진다 (비정상).
        - p-value ≤ 0.05일 때 귀무가설 기각 → 정상 시계열.
        - 일반적으로 1~2차 차분으로 충분하며, 과도한 차분은 모델 성능을 저하시킬 수 있다.
        - 각 반복마다 ADF 검정 통계량, p-value, 기각값을 출력한다.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame, date_range

        # 기본 사용 (정상성 만족까지 자동 차분):
        df = DataFrame({'value': [100, 102, 105, 110, 120]},
                          index=date_range('2020-01', periods=5, freq='M'))
        stationary_df = hs_timeseries.diff(df, 'value')

        # 최대 2차 차분으로 제한:
        stationary_df = hs_timeseries.diff(df, 'value', max_diff=2)

        # 그래프 없이 실행:
        stationary_df = hs_timeseries.diff(df, 'value', plot=False)
        ```
    """
    df = data.copy()

    # 데이터 정상성 여부
    stationarity = False

    # 반복 수행 횟수
    count = 0

    # 데이터가 정상성을 충족하지 않는 동안 반복
    while not stationarity:
        if count == 0:
            print("=========== 원본 데이터 ===========")
        else:
            print("=========== %d차 차분 데이터 ===========" % count)

        if count > 0:
            # 차분 수행
            df = df.diff().dropna()

        if plot:
            lineplot(df=df, yname=yname, xname=df.index, figsize=figsize, dpi=dpi)  # type: ignore

        # ADF Test
        ar = adfuller(df[yname])

        ardict = {
            "검정통계량(ADF Statistic)": [ar[0]],
            "유의수준(p-value)": [ar[1]],
            "최적차수(num of lags)": [ar[2]],
            "관측치 개수(num of observations)": [ar[3]],
        }

        for key, value in ar[4].items():    # type: ignore
            ardict["기각값(Critical Values) %s" % key] = value

        stationarity = ar[1] <= 0.05
        ardict["데이터 정상성 여부"] = "정상" if stationarity else "비정상"

        ardf = DataFrame(ardict, index=["ADF Test"]).T
        pretty_table(ardf)

        # 반복회차 1 증가
        count += 1

        # 최대 차분 횟수가 지정되어 있고, 반복회차가 최대 차분 횟수에 도달하면 종료
        if max_diff is not None and count == max_diff:
            break

    return df


# ===================================================================
# 이동평균 계산
# ===================================================================
def rolling(
    data: Series,
    window: int,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> Series:
    """단순 이동평균(Simple Moving Average, SMA)을 계산한다.

    지정된 윈도우(기간) 내 데이터의 산술평균을 계산하여 시계열의 추세를 파악한다.
    노이즈를 제거하고 장기 추세를 시각화하는 데 유용하다.

    Args:
        data (Series): 시계열 데이터. 인덱스가 datetime 형식이어야 한다.
        window (int): 이동평균 계산 윈도우 크기 (기간).
            예: window=7이면 최근 7개 데이터의 평균을 계산.
        plot (bool, optional): 이동평균 그래프를 표시할지 여부. 기본값은 True.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        Series: 이동평균이 계산된 시계열 데이터.
            처음 (window-1)개 값은 NaN으로 표시된다.

    Notes:
        - 단순 이동평균은 모든 과거 데이터에 동일한 가중치를 부여한다.
        - 윈도우가 클수록 더 부드러운 곡선이 되지만 최신 변화에 덜 민감해진다.
        - 계절성 파악을 위해서는 계절 주기와 동일한 윈도우 사용을 권장한다.

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # 7일 이동평균 계산:
        data = Series([10, 12, 13, 15, 14, 16, 18],
                        index=date_range('2020-01-01', periods=7))
        ma7 = hs_timeseries.rolling(data, window=7)

        # 30일 이동평균, 그래프 없이:
        ma30 = hs_timeseries.rolling(data, window=30, plot=False)
        ```
    """
    rolling = data.rolling(window=window).mean()

    if plot:
        df = DataFrame({rolling.name: rolling}, index=rolling.index)

        lineplot(
            df=df,
            yname=rolling.name,     # type: ignore
            xname=df.index,         # type: ignore
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title(f"Rolling (window={window})"),
        )

    return rolling


# ===================================================================
# 지수가중이동평균
# ===================================================================
def ewm(
    data: Series, span: int, plot: bool = True, figsize: tuple = (10, 5), dpi: int = 100
) -> Series:
    """지수가중이동평균(Exponential Weighted Moving Average, EWMA)을 계산한다.

    최근 데이터에 더 높은 가중치를 부여하는 이동평균으로, 단순이동평균보다
    최신 변화에 민감하게 반응한다. 주가 분석, 수요 예측 등에 널리 사용된다.

    Args:
        data (Series): 시계열 데이터. 인덱스가 datetime 형식이어야 한다.
        span (int): 지수가중이동평균 계산 기간 (span).
            α(smoothing factor) = 2 / (span + 1)로 계산된다.
            span이 클수록 과거 데이터의 영향이 천천히 감소한다.
        plot (bool, optional): EWMA 그래프를 표시할지 여부. 기본값은 True.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        Series: 지수가중이동평균이 계산된 시계열 데이터.

    Notes:
        - EWMA는 최근 관측값에 지수적으로 감소하는 가중치를 부여한다.
        - 단순이동평균과 달리 모든 과거 데이터를 사용하되 시간에 따라 가중치가 감소한다.
        - span=12는 대략 12개 기간의 정보를 반영하는 것으로 해석할 수 있다.
        - α = 2/(span+1) 공식으로 smoothing factor가 결정된다.

    Examples:
        ```python
        from hossam import ewm
        from pandas import Series, date_range

        # 12기간 지수가중이동평균:
        data = Series([10, 12, 13, 15, 14, 16, 18],
                         index=date_range('2020-01-01', periods=7))
        ewma = hs_timeseries.ewm(data, span=12)

        # 단기 추세 파악 (span=5):
        ewma_short = hs_timeseries.ewm(data, span=5, plot=False)
        ```
    """
    ewm = data.ewm(span=span).mean()

    if plot:
        df = DataFrame({ewm.name: ewm}, index=ewm.index)

        lineplot(
            df=df,
            yname=ewm.name,     # type: ignore
            xname=df.index,     # type: ignore
            figsize=figsize,
            dpi=dpi,
            callback=lambda ax: ax.set_title(f"Ewm (span={span})"),
        )

    return ewm


# ===================================================================
# 시계열 분해
# ===================================================================
def seasonal_decompose(
    data: Series,
    model: str = "additive",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
):
    """시계열을 추세(Trend), 계절성(Seasonal), 잔차(Residual) 성분으로 분해한다.

    classical decomposition 기법을 사용하여 시계열을 구조적 성분으로 분해한다.
    계절성 패턴 파악, 추세 분석, 이상치 탐지 등에 유용하다.

    Args:
        data (Series): 시계열 데이터. 인덱스가 datetime 형식이며 일정한 주기를 가져야 한다.
        model (str, optional): 분해 모델 유형.
            - "additive": 가법 모델 (Y = T + S + R)
              계절성의 크기가 일정할 때 사용.
            - "multiplicative": 승법 모델 (Y = T × S × R)
              계절성의 크기가 추세에 비례하여 변할 때 사용.
            기본값은 "additive".
        plot (bool, optional): 분해된 4개 성분(원본, 추세, 계절, 잔차)의
            그래프를 표시할지 여부. 기본값은 True.
        figsize (tuple, optional): 각 그래프의 기본 크기 (width, height).
            실제 출력은 높이가 4배로 조정된다. 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        DataFrame: 분해된 성분을 포함한 데이터프레임. 다음 컬럼 포함:
            - original: 원본 시계열
            - trend: 추세 성분 (장기적 방향성)
            - seasonal: 계절 성분 (주기적 패턴)
            - resid: 잔차 성분 (설명되지 않는 불규칙 변동)

    Raises:
        ValueError: model이 "additive" 또는 "multiplicative"가 아닐 경우.

    Notes:
        - 가법 모델: 계절 변동폭이 일정한 경우 (예: 매년 여름 +10도).
        - 승법 모델: 계절 변동폭이 추세에 비례하는 경우 (예: 매년 여름 +20%).
        - 데이터에 0 또는 음수가 있으면 승법 모델 사용 불가.
        - 주기(period)는 데이터의 빈도(frequency)에서 자동 추론된다.

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # 월별 데이터 가법 분해:
        data = Series([100, 120, 110, 130, 150, 140],
                         index=date_range('2020-01', periods=6, freq='M'))
        components = hs_timeseries.seasonal_decompose(data, model='additive')

        # 승법 모델 사용:
        components = hs_timeseries.seasonal_decompose(data, model='multiplicative', plot=False)
        print(components[['trend', 'seasonal']].head())
        ```
    """
    if model not in ["additive", "multiplicative"]:
        raise ValueError("model은 'additive' 또는 'multiplicative'이어야 합니다.")

    sd = seasonal_decompose(data, model=model)

    sd_df = DataFrame(
        {
            "original": sd.observed,
            "trend": sd.trend,
            "seasonal": sd.seasonal,
            "resid": sd.resid,
        },
        index=data.index,
    )

    if plot:
        figure = sd.plot()
        figure.set_size_inches((figsize[0], figsize[1] * 4))
        figure.set_dpi(dpi)

        fig, ax1, ax2, ax3, ax4 = figure.get_children()

        ax1.set_ylabel("Original")
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.show()
        plt.close()

    return sd_df


# ===================================================================
# 시계열 데이터에 대한 학습/테스트 데이터 분할
# ===================================================================
def train_test_split(data: DataFrame, test_size: float = 0.2) -> tuple[DataFrame, DataFrame]:
    """시계열 데이터를 시간 순서를 유지하며 학습/테스트 세트로 분할한다.

    일반적인 random split과 달리 시간 순서를 엄격히 유지하여 분할한다.
    미래 데이터가 과거 예측에 사용되는 data leakage를 방지한다.

    Args:
        data (DataFrame): 시계열 데이터프레임.
            인덱스가 시간 순서대로 정렬되어 있어야 한다.
        test_size (float, optional): 테스트 데이터 비율 (0~1).
            예: 0.2는 전체 데이터의 마지막 20%를 테스트 세트로 사용.
            기본값은 0.2.

    Returns:
        tuple: (train, test) 형태의 튜플.
            - train (DataFrame): 학습 데이터 (앞부분)
            - test (DataFrame): 테스트 데이터 (뒷부분)

    Notes:
        - 시계열 데이터는 랜덤 분할이 아닌 순차 분할을 사용해야 한다.
        - 학습 데이터는 항상 테스트 데이터보다 시간적으로 앞선다.
        - Cross-validation이 필요한 경우 TimeSeriesSplit 사용을 권장한다.
        - 일반적으로 test_size는 0.1~0.3 범위를 사용한다.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame, date_range

        # 80:20 분할 (기본):
        df = DataFrame({'value': range(100)}, index=date_range('2020-01-01', periods=100))
        train, test = hs_timeseries.train_test_split(df)
        print(len(train), len(test))  # 80, 20

        # 70:30 분할:
        train, test = hs_timeseries.train_test_split(df, test_size=0.3)
        print(len(train), len(test))  # 70, 30
        ```
    """
    train_size = 1 - test_size

    # 처음부터 70% 위치 전까지 분할
    train = data[: int(train_size * len(data))]

    # 70% 위치부터 끝까지 분할
    test = data[int(train_size * len(data)) :]

    return (train, test)


# ===================================================================
# 자기상관함수(ACF, Autocorrelation Function) 그래프 시각화
# ===================================================================
def acf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: Callable | None = None
):
    """자기상관함수(ACF, Autocorrelation Function) 그래프를 시각화한다.

    시계열 데이터와 시차(lag)를 둔 자기 자신과의 상관계수를 계산하여
    시계열의 자기상관 구조를 파악한다. ARIMA 모델의 MA(q) 차수 결정에 사용된다.

    Args:
        data (Series): 시계열 데이터. 정상성을 만족하는 것이 권장된다.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.
        callback (Callable, optional): 추가 그래프 설정을 위한 콜백 함수.
            함수는 ax(Axes) 객체를 인자로 받아야 한다. 기본값은 None.

    Notes:
        - ACF는 y(t)와 y(t-k) 간의 상관계수를 lag k에 대해 표시한다.
        - 파란색 영역(신뢰구간)을 벗어나는 lag는 통계적으로 유의미한 자기상관을 나타낸다.
        - MA(q) 모델: ACF가 lag q 이후 급격히 0으로 수렴한다.
        - AR(p) 모델: ACF가 점진적으로 감소한다.
        - 계절성이 있으면 계절 주기마다 높은 ACF 값이 나타난다.

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # 기본 ACF 플롯:
        data = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         index=date_range('2020-01-01', periods=10))
        hs_timeseries.acf_plot(data)

        # 콜백으로 제목 추가:
        hs_timeseries.acf_plot(data, callback=lambda ax: ax.set_title('My ACF Plot'))
        ```
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    plot_acf(data, ax=ax)
    ax.grid()

    if callback:
        callback(ax)

    plt.show()
    plt.close()


# ===================================================================
# 편자기상관함수(PACF, Partial Autocorrelation Function) 그래프 시각화
# ===================================================================
def pacf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: Callable | None = None
):
    """편자기상관함수(PACF, Partial Autocorrelation Function) 그래프를 시각화한다.

    중간 lag의 영향을 제거한 순수한 자기상관을 계산하여 표시한다.
    ARIMA 모델의 AR(p) 차수 결정에 사용된다.

    Args:
        data (Series): 시계열 데이터. 정상성을 만족하는 것이 권장된다.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.
        callback (Callable, optional): 추가 그래프 설정을 위한 콜백 함수.
            함수는 ax(Axes) 객체를 인자로 받아야 한다. 기본값은 None.

    Notes:
        - PACF는 중간 시차의 영향을 제거한 y(t)와 y(t-k) 간의 상관계수이다.
        - ACF와 달리 간접적 영향을 배제하고 직접적 관계만 측정한다.
        - AR(p) 모델: PACF가 lag p 이후 급격히 0으로 수렴한다.
        - MA(q) 모델: PACF가 점진적으로 감소한다.
        - 파란색 영역(신뢰구간)을 벗어나는 lag가 AR 항의 개수를 나타낸다.

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # 기본 PACF 플롯:
        data = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         index=date_range('2020-01-01', periods=10))
        hs_timeseries.pacf_plot(data)

        # 콜백으로 커스터마이징:
        hs_timeseries.pacf_plot(data, callback=lambda ax: ax.set_ylabel('Partial Correlation'))
        ```
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    plot_pacf(data, ax=ax)
    ax.grid()

    if callback:
        callback(ax)

    plt.show()
    plt.close()


# ===================================================================
# ACF와 PACF 그래프 동시 시각화
# ===================================================================
def acf_pacf_plot(
    data: Series, figsize: tuple = (10, 5), dpi: int = 100, callback: Callable | None = None
):
    """ACF와 PACF 그래프를 동시에 시각화하여 ARIMA 차수를 결정한다.

    ACF와 PACF를 함께 분석하여 ARIMA(p,d,q) 모델의 p(AR 차수)와 q(MA 차수)를
    경험적으로 선택할 수 있다.

    Args:
        data (Series): 시계열 데이터. 정상성을 만족하는 것이 권장된다.
        figsize (tuple, optional): 각 그래프의 기본 크기 (width, height).
            실제 출력은 높이가 2배로 조정된다. 기본값은 (10, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.
        callback (Callable, optional): 추가 그래프 설정을 위한 콜백 함수.
            함수는 ax1, ax2(두 개의 Axes) 객체를 인자로 받아야 한다. 기본값은 None.

    Notes:
        ARIMA 차수 선택 가이드:

        - AR(p) 모델: ACF는 점진 감소, PACF는 lag p 이후 절단
        - MA(q) 모델: ACF는 lag q 이후 절단, PACF는 점진 감소
        - ARMA(p,q) 모델: 둘 다 점진 감소

        실전에서는 auto_arima를 사용한 자동 선택도 권장된다.

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # ARIMA 모델링 전 차수 탐색:
        data = Series([10, 12, 13, 15, 14, 16, 18, 20],
                         index=date_range('2020-01-01', periods=8))

        # 1차 차분 후 ACF/PACF 플롯:
        stationary = hs_timeseries.diff(data, 'value')
        hs_timeseries.acf_pacf_plot(stationary)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2), dpi=dpi)

    plot_acf(data, ax=ax1)
    ax1.grid()

    plot_pacf(data, ax=ax2)
    ax2.grid()

    if callback:
        callback(ax1, ax2)

    plt.show()
    plt.close()


# ===================================================================
# ARIMA 또는 SARIMA 모델을 학습하고 예측 결과를 시각화
# ===================================================================
def arima(
    train: Series,
    test: Series,
    auto: bool = False,
    p: int = 3,
    d: int = 3,
    q: int = 3,
    s: int | None = None,
    periods: int = 0,
    figsize: tuple = (15, 5),
    dpi: int = 100,
) -> ARIMA:
    """ARIMA 또는 SARIMA 모델을 학습하고 예측 결과를 시각화한다.

    ARIMA(p,d,q) 또는 SARIMA(p,d,q)(P,D,Q,s) 모델을 구축하여 시계열 예측을 수행한다.
    auto=True로 설정하면 pmdarima의 auto_arima로 최적 하이퍼파라미터를 자동 탐색한다.

    Args:
        train (Series): 학습용 시계열 데이터. 정상성을 만족해야 한다.
        test (Series): 테스트용 시계열 데이터. 모델 평가에 사용된다.
        auto (bool, optional): auto_arima로 최적 (p,d,q) 자동 탐색 여부.
            - False: 수동 지정한 p,d,q 사용
            - True: AIC 기반 그리드 서치로 최적 모델 탐색
            기본값은 False.
        p (int, optional): AR(AutoRegressive) 차수. 과거 값의 영향을 모델링.
            PACF 그래프를 참고하여 결정. auto=True일 때 max_p로 사용. 기본값은 3.
        d (int, optional): 차분(Differencing) 차수. 비정상 데이터를 정상화.
            diff() 결과를 참고하여 결정. 기본값은 3.
        q (int, optional): MA(Moving Average) 차수. 과거 오차의 영향을 모델링.
            ACF 그래프를 참고하여 결정. auto=True일 때 max_q로 사용. 기본값은 3.
        s (int | None, optional): 계절 주기(Seasonality). None이면 비계절 ARIMA.
            예: 월별 데이터는 s=12, 주별 데이터는 s=52.
            설정 시 SARIMA(p,d,q)(P,D,Q,s) 모델 사용. 기본값은 None.
        periods (int, optional): test 기간 이후 추가 예측 기간 수.
            0이면 test 기간까지만 예측. 기본값은 0.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (15, 5).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        ARIMA | AutoARIMA: 학습된 ARIMA 모델 객체.
            auto=False일 때 statsmodels.ARIMA,
            auto=True일 때 pmdarima.AutoARIMA 반환.

    Notes:
        - ARIMA(p,d,q): p=AR차수, d=차분차수, q=MA차수
        - SARIMA(p,d,q)(P,D,Q,s): 계절 성분 추가 (P,D,Q,s)
        - 모델 선택: ACF/PACF 패턴 분석 또는 auto_arima 사용
        - 데이터는 반드시 정상성을 만족해야 하므로 diff() 먼저 수행 권장
        - auto=True는 시간이 오래 걸릴 수 있으나 최적 모델을 찾아줌

    Examples:
        ```python
        from hossam import *
        from pandas import Series, date_range

        # 수동으로 ARIMA(2,1,2) 모델 생성:
        data = Series([100, 102, 105, 110, 115, 120, 125, 130],
                         index=date_range('2020-01', periods=8, freq='M'))
        train, test = hs_timeseries.train_test_split(data, test_size=0.25)
        model = hs_timeseries.arima(train, test, p=2, d=1, q=2)

        # auto_arima로 최적 모델 탐색:
        model = hs_timeseries.arima(train, test, auto=True)

        # 계절성 모델 SARIMA(1,1,1)(1,1,1,12):
        model = hs_timeseries.arima(train, test, p=1, d=1, q=1, s=12)
        ```
    """
    model = None

    if not auto:
        if s:
            model = ARIMA(train, order=(p, d, q), seasonal_order=(p, d, q, s))
        else:
            model = ARIMA(train, order=(p, d, q))

        fit = model.fit()
        print(fit.summary())

        start_index = 0
        end_index = len(train)
        test_pred = fit.predict(start=start_index, end=end_index)
        pred = fit.forecast(len(test) + periods)
    else:
        # 최적의 ARIMA 모델을 찾는다.
        if s:
            model = auto_arima(
                y=train,  # 모델링하려는 시계열 데이터 또는 배열
                start_p=0,  # p의 시작점
                max_p=p,  # p의 최대값
                d=d,  # 차분 횟수
                start_q=0,  # q의 시작점
                max_q=q,  # q의 최대값
                seasonal=True,  # 계절성 사용 여부
                m=s,  # 계절성 주기
                start_P=0,  # P의 시작점
                max_P=p,  # P의 최대값
                D=d,  # 계절성 차분 횟수
                start_Q=0,  # Q의 시작점
                max_Q=q,  # Q의 최대값
                trace=True,  # 학습 과정 표시 여부
            )
        else:
            model = auto_arima(
                y=train,  # 모델링하려는 시계열 데이터 또는 배열
                start_p=0,  # p의 시작점
                max_p=p,  # p의 최대값
                d=d,  # 차분 횟수
                start_q=0,  # q의 시작점
                max_q=q,  # q의 최대값
                seasonal=False,  # 계절성 사용 여부
                trace=True,  # 학습 과정 표시 여부
            )

        print(model.summary())
        pred = model.predict(n_periods=int(len(test)) + periods)
        pd = None

    # 예측 결과 그래프
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    sb.lineplot(data=train, x=train.index, y=train.columns[0], label="Train", ax=ax)    # type: ignore
    sb.lineplot(data=test, x=test.index, y=test.columns[0], label="Test", ax=ax)        # type: ignore

    if auto:
        sb.lineplot(
            x=pred.index, y=pred.values, label="Prediction", linestyle="--", ax=ax
        )
    else:
        sb.lineplot(x=test_pred.index, y=test_pred, label="Prediction", linestyle="--", ax=ax) # type: ignore
        sb.lineplot(x=pred.index, y=pred, label="Forecast", linestyle="--", ax=ax)

    ax.grid()
    ax.legend()

    plt.show()
    plt.close()

    return model


# ===================================================================
# Prophet 모델을 생성
# ===================================================================
def __prophet_execute(
    train: DataFrame,
    test: DataFrame | None = None,
    periods: int = 0,
    freq: str = "D",
    callback: Callable | None = None,
    **params,
):
    """Prophet 모델을 생성한다.

    Args:
        train (DataFrame): 훈련데이터
        test (DataFrame, optional): 검증데이터. Defaults to None.
        periods (int, optional): 예측기간. Defaults to 0.
        freq (str, optional): 예측주기(D,M,Y). Defaults to "D".
        callback (Callable, optional): 콜백함수. Defaults to None.
        **params (dict, optional): 하이퍼파라미터. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = Prophet(**params)

    if callback:
        callback(model)

    model.fit(train)

    size = 0 if test is None else len(test)
    size = size + periods

    future = model.make_future_dataframe(periods=size, freq=freq)
    forecast = model.predict(future)

    if test is not None:
        pred = forecast[["ds", "yhat"]][-size:]
        score = np.sqrt(mean_squared_error(test["y"].values, pred["yhat"].values))  # type: ignore
    else:
        pred = forecast[["ds", "yhat"]]
        score = np.sqrt(mean_squared_error(train["y"].values, pred["yhat"].values)) # type: ignore

    return model, score, dict(params), forecast, pred


# ===================================================================
# Facebook Prophet 모델을 학습하고 최적 모델을 반환
# ===================================================================
def prophet(
    train: DataFrame,
    test: DataFrame | None = None,
    periods: int = 0,
    freq: str = "D",
    report: bool = True,
    print_forecast: bool = False,
    figsize=(20, 8),
    dpi: int = 200,
    callback: Callable | None = None,
    **params,
) -> tuple[Prophet, dict, float, DataFrame, DataFrame]:
    """Facebook Prophet 모델을 학습하고 최적 모델을 반환한다.

    Facebook(Meta)의 Prophet 라이브러리를 사용하여 시계열 예측 모델을 구축한다.
    추세, 계절성, 휴일 효과를 자동으로 고려하며, 하이퍼파라미터 그리드 서치를 지원한다.

    Args:
        train (DataFrame): 학습 데이터. 'ds'(날짜)와 'y'(값) 컬럼 필수.
            - ds: datetime 형식의 날짜/시간 컬럼
            - y: 예측 대상 수치형 컬럼
        test (DataFrame, optional): 테스트 데이터. 동일한 형식(ds, y).
            제공시 모델 성능 평가(MAE, MSE, RMSE)를 수행. 기본값은 None.
        periods (int, optional): test 기간 이후 추가로 예측할 기간 수.
            기본값은 0.
        freq (str, optional): 예측 빈도.
            - 'D': 일별 (Daily)
            - 'M': 월별 (Monthly)
            - 'Y': 연별 (Yearly)
            - 'H': 시간별 (Hourly)
            기본값은 'D'.
        report (bool, optional): 예측 결과 시각화 및 성분 분해 그래프 표시 여부.
            기본값은 True.
        print_forecast (bool, optional): 예측 결과 테이블 출력 여부.
            기본값은 False.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (20, 8).
        dpi (int, optional): 그래프 해상도. 기본값은 200.
        callback (Callable, optional): Prophet 모델 객체를 커스터마이징하기 위한 콜백 함수.
            함수는 model 객체를 인자로 받아 add_regressor, add_seasonality 등 추가 설정 가능.
            기본값은 None.
        **params: Prophet 하이퍼파라미터 그리드 서치용 딕셔너리.
            주요 파라미터:
            - changepoint_prior_scale: 추세 변화점 유연성 (0.001~0.5)
            - seasonality_prior_scale: 계절성 강도 (0.01~10)
            - seasonality_mode: 'additive' 또는 'multiplicative'
            - changepoint_range: 변화점 탐지 범위 (0~1)
            예: changepoint_prior_scale=[0.001, 0.01, 0.1]

    Returns:
        tuple: (best_model, best_params, best_score, best_forecast, best_pred)
            - best_model (Prophet): RMSE 기준 최적 모델
            - best_params (dict): 최적 하이퍼파라미터
            - best_score (float): 최소 RMSE 값
            - best_forecast (DataFrame): 전체 예측 결과 (ds, yhat, yhat_lower, yhat_upper 등)
            - best_pred (DataFrame): test 기간 예측값 (ds, yhat)

    Notes:
        - Prophet은 결측치와 계절성을 자동 처리하므로 ARIMA보다 전처리가 적다.
        - changepoint_prior_scale이 클수록 더 유연하게 피팅 (과적합 주의).
        - 휴일 효과는 add_country_holidays() 또는 holidays 파라미터로 추가.
        - 외부 회귀변수는 callback에서 add_regressor()로 추가 가능.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame, date_range

        # 기본 사용 (단일 모델):
        train = DataFrame({
                    'ds': date_range('2020-01-01', periods=100),
                    'y': range(100)
                })
        model, params, score, forecast, pred = hs_timeseries.prophet(train)

        하이퍼파라미터 그리드 서치:

        model, params, score, forecast, pred = hs_timeseries.prophet(
                train,
                changepoint_prior_scale=[0.001, 0.01, 0.1],
                seasonality_prior_scale=[0.01, 0.1, 1.0],
                seasonality_mode=['additive', 'multiplicative']
        )

        # 휴일 효과 추가:
        def add_holidays(m):
            m.add_country_holidays(country_name='KR')

        model, _, _, _, _ = hs_timeseries.prophet(train, callback=add_holidays)
        ```
    """

    # logger = logging.getLogger("cmdstanpy")
    # logger.addHandler(logging.NullHandler())
    # logger.propagate = False
    # logger.setLevel(logging.CRITICAL)

    # ------------------------------------------------------
    # 분석모델 생성

    result = []
    processes = []

    if params:
        with futures.ThreadPoolExecutor() as executor:
            params = ParameterGrid(params)

            for p in params:
                processes.append(
                    executor.submit(
                        __prophet_execute,
                        train=train,
                        test=test,
                        periods=periods,
                        freq=freq,
                        callback=callback,
                        **p,
                    )
                )

            for p in futures.as_completed(processes):
                m, score, params, forecast, pred = p.result()
                result.append(
                    {
                        "model": m,
                        "params": params,
                        "score": score,
                        "forecast": forecast,
                        "pred": pred,
                    }
                )

    else:
        m, score, params, forecast, pred = __prophet_execute(
            train=train, test=test, periods=periods, freq=freq, callback=callback, **p  # type: ignore
        )
        result.append(
            {
                "model": m,
                "params": params,
                "score": score,
                "forecast": forecast,
                "pred": pred,
            }
        )

    result_df = DataFrame(result).sort_values("score").reset_index(drop=True)
    best_model, best_params, best_score, best_forecast, best_pred = result_df.iloc[0]

    # print_result = []
    # for i, v in enumerate(result):
    #     item = v["params"]
    #     item["score"] = v["score"]
    #     print_result.append(item)

    # pretty_table(
    #     DataFrame(print_result)
    #     .sort_values("score", ascending=True)
    #     .reset_index(drop=True)
    # )

    if report:
        hs_prophet_report(  # type: ignore
            best_model, best_forecast, best_pred, test, print_forecast, figsize, dpi
        )

    return best_model, best_params, best_score, best_forecast, best_pred


# ===================================================================
# Prophet 모델의 예측 결과와 성분 분해를 시각화하고 성능을 평가
# ===================================================================
def prophet_report(
    model: Prophet,
    forecast: DataFrame,
    pred: DataFrame,
    test: DataFrame | None = None,
    print_forecast: bool = False,
    figsize: tuple = (20, 8),
    dpi: int = 100,
) -> None:
    """Prophet 모델의 예측 결과와 성분 분해를 시각화하고 성능을 평가한다.

    학습된 Prophet 모델의 예측 결과, 변화점(changepoints), 신뢰구간을 시각화하고,
    추세, 계절성 등 성분을 분해하여 표시한다. test 데이터가 있으면 성능 지표를 계산한다.

    Args:
        model (Prophet): 학습된 Prophet 모델 객체.
        forecast (DataFrame): model.predict()의 반환값. 전체 예측 결과.
            다음 컬럼 포함: ds, yhat, yhat_lower, yhat_upper, trend, seasonal 등.
        pred (DataFrame): test 기간의 예측값. ds와 yhat 컬럼 포함.
        test (DataFrame, optional): 테스트 데이터. ds와 y 컬럼 필수.
            제공시 실제값과 비교하여 MAE, MSE, RMSE를 계산. 기본값은 None.
        print_forecast (bool, optional): 예측 결과 테이블 전체를 출력할지 여부.
            기본값은 False.
        figsize (tuple, optional): 그래프 크기 (width, height). 기본값은 (20, 8).
        dpi (int, optional): 그래프 해상도. 기본값은 100.

    Returns:
        None: 출력만 수행하고 반환값 없음.

    Notes:
        - 첨번째 그래프: 예측 결과 + 신뢰구간 + 변화점
        - 두번째 그래프: 성분 분해 (trend, weekly, yearly 등)
        - test 데이터가 있으면 붉은 점과 선으로 실제값 표시
        - 변화점은 모델이 추세 변화를 감지한 시점을 수직선으로 표시

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame, date_range

        # 기본 리포트 출력:
        model, _, _, forecast, pred = hs_timeseries.prophet(train)
        hs_timeseries.prophet_report(model, forecast, pred)

        # test 데이터와 함께 성능 평가:
        hs_timeseries.prophet_report(model, forecast, pred, test=test)

        # 예측 테이블 출력:
        hs_timeseries.prophet_report(model, forecast, pred, print_forecast=True)
        ```
    """

    # ------------------------------------------------------
    # 결과 시각화
    fig = model.plot(forecast, figsize=figsize, xlabel="Date", ylabel="Value")
    fig.set_dpi(dpi)
    ax = fig.gca()
    add_changepoints_to_plot(ax, model, forecast)

    if test is not None:
        sb.scatterplot(
            data=test,
            x="ds",
            y="y",
            size=1,
            color="#ff0000",
            marker="o",
            ax=ax,
        )

        sb.lineplot(
            data=test,
            x="ds",
            y="y",
            color="#ff6600",
            ax=ax,
            label="Test",
            alpha=0.7,
            linewidth=0.7,
            linestyle="--",
        )

    ax.set_ylim([forecast["yhat"].min() * 0.95, forecast["yhat"].max() * 1.05]) # type: ignore

    plt.legend()
    plt.show()
    plt.close()

    height = figsize[1] * (len(model.seasonalities) + 1)

    fig = model.plot_components(forecast, figsize=(figsize[0], height))
    fig.set_dpi(dpi)
    ax = fig.gca()

    plt.show()
    plt.close()

    # 예측 결과 테이블
    if print_forecast:
        pretty_table(forecast)

    if test is not None:
        yhat = forecast["yhat"].values[-len(test) :]
        y = test["y"].values

        result = {
            "평균절대오차(MAE)": mean_absolute_error(y, yhat),  # type: ignore
            "평균제곱오차(MSE)": mean_squared_error(y, yhat),   # type: ignore
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y, yhat))  # type: ignore
        }

        pretty_table(DataFrame(result, index=["Prophet"]).T)


# ===================================================================
# 주말 날짜를 포함하는 휴일 데이터프레임을 생성
# ===================================================================
def get_weekend_df(start: dt.datetime | str, end: dt.datetime | str | None = None) -> DataFrame:
    """주말 날짜를 포함하는 휴일 데이터프레임을 생성한다.

    Prophet 모델의 holidays 파라미터에 사용할 수 있는 형식의 주말 휴일
    데이터프레임을 생성한다. 토요일과 일요일을 'holiday'로 표시한다.

    Args:
        start (datetime | str): 시작일.
            datetime 객체 또는 문자열 형식 ("YYYY-MM-DD") 가능.
            예: "2021-01-01" 또는 datetime(2021, 1, 1)
        end (datetime | str, optional): 종료일.
            지정하지 않으면 현재 날짜까지. 기본값은 None.

    Returns:
        DataFrame: 주말 휴일 데이터프레임. 다음 컬럼 포함:
            - ds (datetime): 토요일 또는 일요일 날짜
            - holiday (str): 'holiday' 문자열 (고정값)

    Notes:
        - Prophet의 holidays 파라미터는 'ds'와 'holiday' 컬럼이 필요하다.
        - 주말뿐 아니라 다른 휴일도 추가하려면 이 함수 결과와 병합(concat)하여 사용.
        - Prophet은 add_country_holidays() 메서드로 국가별 공휴일도 지원한다.
        - 토요일(Saturday), 일요일(Sunday)을 자동 탐지하여 추출한다.

    Examples:
        ```python
        from hossam import *
        from pandas import DataFrame, date_range

        # 2020년 전체 주말 생성:
        weekends = hs_timeseries.get_weekend_df('2020-01-01', '2020-12-31')
        print(len(weekends))  # 104 (52주 × 2일)

        # 현재까지의 주말:
        weekends = hs_timeseries.get_weekend_df('2023-01-01')
        print(weekends.head())

        # Prophet 모델에 주말 효과 추가:
        weekends = hs_timeseries.get_weekend_df('2020-01-01', '2025-12-31')
        model = hs_timeseries.prophet(train, holidays=weekends)
        ```
    """
    if end is None:
        end = dt.datetime.now()

    date = date_range(start, end)
    df = DataFrame({"date": date, "weekend": date.day_name()}).set_index("date")

    df["weekend"] = df["weekend"].apply(
        lambda x: 1 if x in ["Saturday", "Sunday"] else 0
    )

    df2 = df[df["weekend"] == 1]
    df2["holiday"] = "holiday"
    df2.drop("weekend", axis=1, inplace=True)
    df2.reset_index(drop=False, inplace=True)
    df2.rename(columns={"date": "ds"}, inplace=True)

    return df2
