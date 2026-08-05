import warnings
import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame, Series, concat
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from . import my_plot


# 인덱스의 시간 단위(freq)별 기본 계절 주기
# 월간 데이터는 1년(12개월), 분기 데이터는 1년(4분기), 일간 데이터는 1주(7일)가 기본이다.
FREQ_PERIOD = {
    "M": 12, "MS": 12, "ME": 12,    # 월간
    "Q": 4, "QS": 4, "QE": 4,       # 분기
    "A": 1, "Y": 1, "YS": 1,        # 연간(계절성 없음)
    "W": 52,                        # 주간
    "D": 7,                         # 일간(요일 주기)
    "H": 24, "h": 24,               # 시간(하루 주기)
}


def _to_series(data, y=None):
    """DataFrame + 컬럼명 또는 Series/배열을 하나의 Series로 통일한다.

    이 모듈의 모든 함수가 `data`와 `y`를 같은 방식으로 받도록 하기 위한 내부 함수다.

    Args:
        data: 시계열 데이터프레임, 시리즈 또는 1차원 배열.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
            컬럼이 하나뿐이면 생략할 수 있다.

    Returns:
        Series: 분석 대상 시계열.

    Raises:
        KeyError: 지정한 컬럼이 데이터프레임에 존재하지 않는 경우.
        ValueError: 컬럼이 여러 개인데 `y`를 지정하지 않은 경우.
    """
    if isinstance(data, DataFrame):
        if y is None:
            if data.shape[1] != 1:
                raise ValueError(
                    "컬럼이 여러 개인 데이터프레임은 y 파라미터로 대상 컬럼을 지정해야 합니다.")
            return data.iloc[:, 0]

        if y not in data.columns:
            raise KeyError(f"'{y}' 컬럼이 데이터프레임에 존재하지 않습니다.")

        return data[y]

    if isinstance(data, Series):
        return data

    return Series(data)


def _infer_period(s, period=None):
    """시계열 인덱스의 시간 단위로부터 계절 주기를 추론한다.

    Args:
        s (Series): 시간 인덱스를 가진 시계열.
        period (int): 사용자가 직접 지정한 계절 주기. 지정하면 그대로 반환한다 (기본값: None).

    Returns:
        int | None: 추론된 계절 주기. 추론할 수 없으면 None.
    """
    if period is not None:
        return period

    # 인덱스에 freq 정보가 있으면 그 문자열로 주기를 찾는다
    freq = getattr(s.index, "freqstr", None)

    if freq:
        key = freq.split("-")[0]                    # "W-SUN" 같은 형태에서 앞부분만 사용
        if key in FREQ_PERIOD:
            return FREQ_PERIOD[key]

    # freq가 없으면 인덱스가 날짜형인지 확인 후 월 간격으로 추정
    if hasattr(s.index, "inferred_freq") and s.index.inferred_freq:
        key = s.index.inferred_freq.split("-")[0]
        if key in FREQ_PERIOD:
            return FREQ_PERIOD[key]

    return None


# ---------------------------------------------------------------------
# 1. 정상성 진단
# ---------------------------------------------------------------------

def test_stationary(data, y=None, alpha=0.05, name=None, verbose=True):
    """ADF(Augmented Dickey-Fuller) 검정으로 시계열의 정상성을 판정한다.

    귀무가설은 "단위근이 존재한다(=비정상)"이므로 **p-value가 작아야 정상**이다.
    정규성 검정과 방향이 반대라는 점에 주의한다.

    Args:
        data: 시계열 데이터프레임, 시리즈 또는 배열. 결측치는 자동 제거된다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        alpha (float): 유의수준 (기본값: 0.05).
        name (str): 결과표에 표시할 대상 이름. None이면 시리즈 이름을 사용한다 (기본값: None).
        verbose (bool): 결과표를 화면에 출력할지 여부 (기본값: True).

    Returns:
        dict: 검정 결과. 여러 시계열의 결과를 표로 묶기 쉽도록 딕셔너리로 반환한다.
            키는 대상·관측치 수·검정통계량(ADF)·p-value·사용 시차·1%/5%/10% 기각값·정상성·판정.
    """
    # --- 1) 검정 대상 확보 ---
    s = _to_series(data, y).dropna()

    if name is None:
        name = s.name if s.name else "데이터"

    # --- 2) ADF 검정 수행 ---
    ar = adfuller(s)
    statistic = float(ar[0])        # 검정통계량 (음수이고 작을수록 정상)
    pvalue = float(ar[1])           # 유의확률
    usedlag = int(ar[2])            # 자기상관 보정에 사용한 시차 수
    cvalues = ar[4]                 # 유의수준별 기각값
    stationarity = bool(pvalue <= alpha)    # 정상성 충족 여부

    # --- 3) 결과 딕셔너리 구성 ---
    result = {
        "대상": name,                                   # 검정 대상 이름
        "관측치 수": len(s),                             # 결측 제거 후 관측치 수
        "검정통계량(ADF)": round(statistic, 4),          # 검정통계량
        "p-value": round(pvalue, 4),                    # 유의확률
        "사용 시차": usedlag,                            # 사용한 시차 수
        "1% 기각값": round(cvalues["1%"], 3),            # 1% 유의수준 경계선
        "5% 기각값": round(cvalues["5%"], 3),            # 5% 유의수준 경계선
        "10% 기각값": round(cvalues["10%"], 3),          # 10% 유의수준 경계선
        "정상성": stationarity,                          # 정상성 충족 여부 (True/False)
        "판정": "정상" if stationarity else "비정상",     # 판정 결과 문자열
    }

    # --- 4) 결과표 출력 ---
    if verbose:
        display(DataFrame([result]).set_index("대상").T)

    return result


def plot_stationary(data, y=None, window=None, title=None, xlabel=None, ylabel=None,
                    width=1280, height=480, save_path=None):
    """원본·이동평균·이동표준편차를 한 그래프에 겹쳐 그려 정상성을 눈으로 확인한다.

    정상 시계열이라면 이동평균과 이동표준편차가 모두 **수평선에 가깝게** 나타난다.
    이동평균이 기울어져 있으면 평균이 일정하지 않고(추세 존재),
    이동표준편차가 기울어져 있으면 분산이 일정하지 않다(변환 필요).

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        window (int): 이동 계산에 사용할 창의 크기. None이면 인덱스에서 계절 주기를 추론한다 (기본값: None).
        title (str): 그래프 제목 (기본값: None → 자동 생성).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 480).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본·이동평균·이동표준편차 컬럼을 담은 데이터프레임.
    """
    # --- 1) 창 크기 결정 ---
    s = _to_series(data, y)
    window = _infer_period(s, window) or 12

    # --- 2) 이동평균·이동표준편차 계산 ---
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()

    # --- 3) 세 개의 선을 하나의 축에 겹쳐 그리기 ---
    fig, ax = my_plot.init(width=width, height=height,
                           title=title if title else f"원본 · 이동평균 · 이동표준편차 (window={window})",
                           xlabel=xlabel, ylabel=ylabel)

    my_plot.lineplot(x=s.index, y=s, label="원본", ax=ax)
    my_plot.lineplot(x=roll_mean.index, y=roll_mean, label="이동평균", ax=ax)
    my_plot.lineplot(x=roll_std.index, y=roll_std, label="이동표준편차", ax=ax)

    ax.legend()
    my_plot.show(save_path=save_path)

    # --- 4) 계산 결과 반환 ---
    return DataFrame({
        "원본": s,
        f"이동평균({window})": roll_mean,
        f"이동표준편차({window})": roll_std,
    })


def report_stationary(data, y=None, period=None, alpha=0.05, log=True):
    """여러 전처리 조합을 한꺼번에 적용해 ADF 검정 결과를 비교표로 만든다.

    "차분을 몇 번 해야 하는가"를 감으로 정하지 않고, 조합별 결과를 나란히 놓고
    **가장 적은 처리로 정상성을 얻는 조합**을 고르기 위한 함수다.

    비교 대상 조합:
        원본 / 로그변환 / 1차 차분 / 2차 차분 / 계절차분 /
        로그+1차 차분 / 로그+계절차분 / 로그+1차+계절차분

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        period (int): 계절 주기. None이면 인덱스에서 추론한다 (기본값: None).
        alpha (float): 유의수준 (기본값: 0.05).
        log (bool): 로그변환 조합을 포함할지 여부. 0 이하의 값이 있으면 자동으로 제외된다 (기본값: True).

    Returns:
        DataFrame: 조합별 관측치 수·검정통계량·p-value·5% 기각값·표준편차·판정 비교표.
    """
    # --- 1) 대상 시계열과 계절 주기 확보 ---
    s = _to_series(data, y).dropna()
    period = _infer_period(s, period)

    # 0 이하의 값이 있으면 로그를 취할 수 없으므로 로그 조합을 제외한다
    if log and (s <= 0).any():
        log = False

    # --- 2) 비교할 전처리 조합 구성 ---
    cases = {
        "① 원본": s,
        "② 1차 차분": s.diff(),
        "③ 2차 차분": s.diff().diff(),
    }

    if log:
        log_s = np.log(s)
        cases["④ 로그변환"] = log_s
        cases["⑤ 로그 + 1차 차분"] = log_s.diff()

    if period and period > 1:
        cases[f"⑥ 계절차분({period})"] = s.diff(period)

        if log:
            cases[f"⑦ 로그 + 계절차분({period})"] = log_s.diff(period)
            cases[f"⑧ 로그 + 1차 차분 + 계절차분({period})"] = log_s.diff().diff(period)

    # --- 3) 조합별로 ADF 검정 수행 ---
    rows = []
    for label, values in cases.items():
        r = test_stationary(values, alpha=alpha, name=label, verbose=False)
        r["표준편차"] = round(float(values.dropna().std()), 4)
        rows.append(r)

    # --- 4) 비교표 구성 및 반환 ---
    rdf = DataFrame(rows)
    return rdf[["대상", "관측치 수", "검정통계량(ADF)", "p-value", "5% 기각값", "표준편차", "판정"]]


def auto_diff(data, y=None, max_diff=2, alpha=0.05, plot=True, verbose=True,
              width=1280, height=400):
    """정상성을 만족할 때까지 차분을 반복하되, 과대차분을 경고한다.

    차분 횟수를 늘리면 p-value는 계속 작아지지만 **표준편차가 다시 커지는 지점**부터는
    데이터에 없던 노이즈를 만들어내는 과대차분(over-differencing)이다.
    이 함수는 차수별 표준편차를 함께 보고하여 그 지점을 드러낸다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        max_diff (int): 최대 차분 횟수. 과대차분 방지를 위해 기본값은 2 (기본값: 2).
        alpha (float): 유의수준 (기본값: 0.05).
        plot (bool): 차분 단계마다 시계열 그래프를 그릴지 여부 (기본값: True).
        verbose (bool): 차수별 검정 결과표를 출력할지 여부 (기본값: True).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 400).

    Returns:
        Series: 정상성을 만족하는(또는 max_diff까지 차분한) 시계열.
            차분 횟수는 `.attrs["diff_count"]`에 저장된다.
    """
    # --- 1) 차분을 반복하며 차수별 결과를 수집 ---
    s = _to_series(data, y).dropna()
    name = s.name if s.name else "데이터"

    rows = []           # 차수별 검정 결과
    count = 0           # 현재 차분 횟수
    result = s          # 반환할 시계열

    while True:
        label = "원본" if count == 0 else f"{count}차 차분"
        r = test_stationary(result, alpha=alpha, name=label, verbose=False)
        r["표준편차"] = round(float(result.std()), 4)
        rows.append(r)

        if plot:
            my_plot.lineplot(x=result.index, y=result, title=f"{name} - {label}",
                             width=width, height=height)

        # 정상성을 만족했거나 최대 차분 횟수에 도달하면 종료
        if r["정상성"] or count >= max_diff:
            break

        count += 1
        result = result.diff().dropna()

    rdf = DataFrame(rows)

    # --- 2) 과대차분 판정 ---
    # 표준편차가 최소가 되는 차수를 넘어서면 노이즈를 증폭시킨 것이다
    best = int(rdf["표준편차"].idxmin())
    over = count > best

    # --- 3) 결과 보고 ---
    if verbose:
        display(rdf[["대상", "관측치 수", "검정통계량(ADF)", "p-value", "판정", "표준편차"]])

        if not rdf.iloc[-1]["정상성"]:
            display(Markdown(
                f"> ⚠️ 최대 차분 횟수({max_diff})까지 수행했으나 정상성을 만족하지 못했다. "
                f"분산이 문제라면 **로그변환**을, 계절성이 문제라면 **계절차분**을 함께 고려한다."))
        elif over:
            display(Markdown(
                f"> ⚠️ **과대차분(over-differencing) 의심**: 표준편차가 {best}차에서 최소값 "
                f"({rdf.iloc[best]['표준편차']})을 찍고 {count}차에서 {rdf.iloc[count]['표준편차']}로 다시 커졌다. "
                f"차분 횟수를 늘리는 대신 **로그변환·계절차분**을 검토한다."))
        else:
            display(Markdown(f"> {count}차 차분으로 정상성을 확보했다. (ARIMA의 **d = {count}**)"))

    # --- 4) 차분 횟수를 속성에 기록하여 반환 ---
    result.attrs["diff_count"] = count
    return result


# ---------------------------------------------------------------------
# 2. 시계열 탐색
# ---------------------------------------------------------------------

def plot_rolling(data, y=None, windows=(3, 6, 12), title=None, width=960, height=400,
                 save_path=None):
    """여러 창 크기의 이동평균(단순이동평균)을 원본과 나란히 시각화한다.

    이동평균은 짧은 기간의 요동(노이즈)을 지워 **전체적인 흐름**을 드러낸다.
    창이 클수록 곡선이 부드러워지지만 최근 변화에 둔감해진다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        windows (tuple): 이동평균 창 크기의 목록 (기본값: (3, 6, 12)).
        title (str): 그래프 전체 제목 (기본값: None → 자동 생성).
        width (int): 그래프 한 칸의 너비 (기본값: 960).
        height (int): 그래프 한 칸의 높이 (기본값: 400).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본과 창 크기별 이동평균을 담은 데이터프레임.
    """
    return _plot_smoothing(data, y, windows, kind="rolling", title=title,
                           width=width, height=height, save_path=save_path)


def plot_ewm(data, y=None, spans=(3, 6, 12), title=None, width=960, height=400,
             save_path=None):
    """여러 기간(span)의 지수가중이동평균을 원본과 나란히 시각화한다.

    이동평균이 창 안의 값을 모두 똑같이 취급하는 것과 달리,
    지수평활은 **최근 값에 더 큰 가중치**를 준다. 그래서 변화에 더 빠르게 반응한다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        spans (tuple): 지수평활 기간(span)의 목록 (기본값: (3, 6, 12)).
            평활계수는 α = 2 / (span + 1) 로 결정된다.
        title (str): 그래프 전체 제목 (기본값: None → 자동 생성).
        width (int): 그래프 한 칸의 너비 (기본값: 960).
        height (int): 그래프 한 칸의 높이 (기본값: 400).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본과 기간별 지수가중이동평균을 담은 데이터프레임.
    """
    return _plot_smoothing(data, y, spans, kind="ewm", title=title,
                           width=width, height=height, save_path=save_path)


def _plot_smoothing(data, y, params, kind, title, width, height, save_path):
    """이동평균/지수평활 시각화의 공통 처리를 담당하는 내부 함수.

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): 분석 대상 컬럼명.
        params (tuple): 창 크기(rolling) 또는 기간(ewm)의 목록.
        kind (str): "rolling" 또는 "ewm".
        title (str): 그래프 전체 제목.
        width (int): 그래프 한 칸의 너비.
        height (int): 그래프 한 칸의 높이.
        save_path (str): 그래프 저장 경로.

    Returns:
        DataFrame: 원본과 평활 결과를 담은 데이터프레임.
    """
    # --- 1) 평활 결과 계산 ---
    s = _to_series(data, y)
    label = "이동평균" if kind == "rolling" else "지수평활"

    sdf = DataFrame({"원본": s})
    for p in params:
        if kind == "rolling":
            sdf[f"{label}({p})"] = s.rolling(p).mean()
        else:
            # ewm의 첫 번째 인자는 com(중심질량)이므로 span을 명시해야 의도한 기간이 된다
            sdf[f"{label}({p})"] = s.ewm(span=p).mean()

    # --- 2) 원본 + 평활 결과를 격자로 배치 ---
    cols = 2
    rows = int(np.ceil(len(sdf.columns) / cols))

    fig, ax = my_plot.init(rows=rows, cols=cols, width=width, height=height,
                           title=title if title else f"{label} 비교")

    for i, col in enumerate(sdf.columns):
        my_plot.lineplot(x=sdf.index, y=sdf[col], ax=ax[i])
        ax[i].set_title(col)

    # 남는 칸은 숨긴다
    for i in range(len(sdf.columns), rows * cols):
        ax[i].set_visible(False)

    my_plot.show(save_path=save_path)

    return sdf


def decompose(data, y=None, model="additive", period=None, plot=True,
              width=1280, height=800, save_path=None):
    """시계열을 추세·계절성·잔차 세 가지 성분으로 분해한다.

    예측이 아니라 **설명**을 위한 도구다. "왜 이런 모양이 나왔는가"를
    장기 흐름(추세) · 반복 패턴(계절성) · 나머지(잔차)로 나누어 보여준다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 인덱스가 날짜형이어야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        model (str): 성분의 결합 방식 (기본값: "additive").
            - "additive": 가법 모델 (원본 = 추세 + 계절 + 잔차). 변동폭이 일정할 때.
            - "multiplicative": 승법 모델 (원본 = 추세 × 계절 × 잔차). 변동폭이 추세에 비례할 때.
        period (int): 계절 주기. None이면 인덱스에서 추론한다 (기본값: None).
        plot (bool): 분해 결과를 4단 그래프로 시각화할지 여부 (기본값: True).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 800).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: original·trend·seasonal·resid 컬럼을 담은 데이터프레임.
            원본과 **같은 인덱스**를 사용하므로 그대로 이어붙여 쓸 수 있다.

    Raises:
        ValueError: model이 "additive" 또는 "multiplicative"가 아닌 경우.
    """
    # --- 1) 파라미터 검증 ---
    if model not in ("additive", "multiplicative"):
        raise ValueError("model은 'additive' 또는 'multiplicative'이어야 합니다.")

    s = _to_series(data, y).dropna()

    # 승법 모델은 0 이하의 값을 다룰 수 없다
    if model == "multiplicative" and (s <= 0).any():
        raise ValueError("승법 모델은 0 이하의 값을 포함한 시계열에 사용할 수 없습니다.")

    period = _infer_period(s, period)

    # --- 2) 분해 수행 ---
    sd = seasonal_decompose(s, model=model, period=period)

    # 인덱스는 반드시 분해에 사용한 시계열의 것을 써야 한다.
    # 원본 데이터프레임의 인덱스(예: RangeIndex)를 쓰면 전부 결측치가 된다.
    sdf = DataFrame({
        "original": sd.observed,        # 원본 시계열
        "trend": sd.trend,              # 추세 성분 (장기적 방향)
        "seasonal": sd.seasonal,        # 계절 성분 (주기적 반복)
        "resid": sd.resid,              # 잔차 성분 (설명되지 않는 변동)
    }, index=s.index)

    # --- 3) 4단 그래프로 시각화 ---
    if plot:
        fig, ax = my_plot.init(rows=4, cols=1, width=width, height=height // 4,
                               title=f"시계열 분해 ({model}, period={period})")

        for i, col in enumerate(sdf.columns):
            # 잔차는 불규칙하므로 점으로 그려야 패턴이 보인다
            if col == "resid":
                sb.scatterplot(x=sdf.index, y=sdf[col], s=8, ax=ax[i])
            else:
                my_plot.lineplot(x=sdf.index, y=sdf[col], ax=ax[i])

            ax[i].set_title(col)
            ax[i].set_ylabel("")

        # 승법 모델의 잔차는 1을, 가법 모델의 잔차는 0을 기준선으로 삼는다
        ax[3].axhline(1 if model == "multiplicative" else 0,
                      color="red", linestyle="--", linewidth=1)

        my_plot.show(save_path=save_path)

    return sdf


# ---------------------------------------------------------------------
# 3. ARIMA 차수 결정 (ACF / PACF)
# ---------------------------------------------------------------------

def _report_correlation(data, y, lags, alpha, kind):
    """ACF 또는 PACF 결과표를 만드는 내부 공통 함수.

    Args:
        data: 시계열 데이터프레임 또는 시리즈.
        y (str): 분석 대상 컬럼명.
        lags (int): 계산할 최대 시차.
        alpha (float): 유의수준 (사용하지 않고 2/√N 근사 기준을 쓴다).
        kind (str): "acf" 또는 "pacf".

    Returns:
        DataFrame: lag·계수·유의성·절단후보 컬럼을 담은 결과표.
    """
    # --- 1) 상관계수 계산 ---
    s = _to_series(data, y).dropna()
    values = acf(s, nlags=lags) if kind == "acf" else pacf(s, nlags=lags)

    # 유의성 판단 기준: 근사적 95% 신뢰구간 경계 2/√N
    threshold = 2 / np.sqrt(len(s))

    # --- 2) 결과표 구성 ---
    cdf = DataFrame({
        "lag": np.arange(len(values)),                  # 시차
        kind: np.round(values, 4),                      # 자기상관/부분자기상관 계수
    })

    cdf["절댓값"] = cdf[kind].abs().round(4)             # 방향과 무관한 영향의 크기
    cdf["유의성"] = cdf["절댓값"] > threshold            # 신뢰구간을 벗어나는가
    cdf["기준값"] = round(threshold, 4)                  # 유의성 판단 기준선

    # --- 3) lag=0 제외 후 절단 후보 계산 ---
    # lag 0은 자기 자신과의 상관이므로 항상 1이다. 판정에서 제외한다.
    cdf = cdf.query("lag > 0").reset_index(drop=True)

    # 직전 lag는 유의했는데 현재 lag가 유의하지 않으면 그 지점이 "절단"이다
    cdf["직전유의"] = cdf["유의성"].shift(1)
    cdf["절단후보"] = (cdf["직전유의"] == True) & (cdf["유의성"] == False)

    return cdf[["lag", kind, "절댓값", "기준값", "유의성", "절단후보"]]


def report_acf(data, y=None, lags=36, alpha=0.05):
    """ACF(자기상관함수) 결과표를 생성한다.

    ACF는 현재 값과 **k시점 전 값**의 상관계수를 시차별로 계산한 것이다.
    "과거의 영향이 몇 시점까지 이어지는가"를 전반적으로 보여주며,
    MA 차수 **q**를 판단하는 근거가 된다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 정상성을 만족한 상태여야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        lags (int): 계산할 최대 시차. 계절성 확인을 위해 주기의 2~3배를 권장한다 (기본값: 36).
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        DataFrame: lag·acf·절댓값·기준값·유의성·절단후보 컬럼의 결과표.
    """
    return _report_correlation(data, y, lags, alpha, kind="acf")


def report_pacf(data, y=None, lags=36, alpha=0.05):
    """PACF(부분자기상관함수) 결과표를 생성한다.

    PACF는 중간 시차의 영향을 **모두 제거한 뒤** 남는 순수한 상관계수다.
    "그 시차 자체의 직접적인 영향"만 보여주며, AR 차수 **p**를 판단하는 근거가 된다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 정상성을 만족한 상태여야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        lags (int): 계산할 최대 시차 (기본값: 36).
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        DataFrame: lag·pacf·절댓값·기준값·유의성·절단후보 컬럼의 결과표.
    """
    return _report_correlation(data, y, lags, alpha, kind="pacf")


def find_order(data, y=None, lags=36, period=None, verbose=True):
    """ACF/PACF의 절단 지점으로부터 ARIMA의 차수 후보 p, q를 도출한다.

    판정 규칙:
        - **q 후보**: ACF에서 연속으로 유의하던 구간이 끊기기 직전의 lag
        - **p 후보**: PACF에서 연속으로 유의하던 구간이 끊기기 직전의 lag
        - 계절 주기의 배수 위치에서 유의성이 관찰되면 계절 성분이 존재한다고 본다

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 정상성을 만족한 상태여야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        lags (int): 계산할 최대 시차 (기본값: 36).
        period (int): 계절 주기. None이면 인덱스에서 추론한다 (기본값: None).
        verbose (bool): 판정 결과를 표로 출력할지 여부 (기본값: True).

    Returns:
        dict: p·q(차수 후보)와 seasonal(계절 성분 존재 여부), seasonal_lags(유의한 계절 시차) 키를 갖는 딕셔너리.
            절단 지점을 찾지 못하면 해당 차수는 0이 된다.
    """
    # --- 1) ACF/PACF 결과표 생성 ---
    s = _to_series(data, y).dropna()
    period = _infer_period(s, period)

    adf_result = report_acf(s, lags=lags)
    pdf_result = report_pacf(s, lags=lags)

    # --- 2) 절단 지점에서 차수 후보 도출 ---
    def cut_order(cdf):
        """절단 후보 중 가장 빠른 lag의 직전 값을 차수 후보로 반환한다."""
        cut = cdf.loc[cdf["절단후보"], "lag"]
        # 절단 지점이 없으면(계속 유의하거나 계속 비유의) 차수를 특정할 수 없다
        return int(cut.min()) - 1 if len(cut) else 0

    q = cut_order(adf_result)
    p = cut_order(pdf_result)

    # --- 3) 계절 성분 존재 여부 확인 ---
    seasonal_lags = []
    if period and period > 1:
        # 계절 주기의 배수 위치에서 ACF가 유의한지 확인한다
        for lag in range(period, lags + 1, period):
            row = adf_result[adf_result["lag"] == lag]
            if len(row) and bool(row["유의성"].iloc[0]):
                seasonal_lags.append(lag)

    result = {
        "p": p,                                     # AR 차수 후보 (PACF 절단)
        "q": q,                                     # MA 차수 후보 (ACF 절단)
        "seasonal": len(seasonal_lags) > 0,         # 계절 성분 존재 여부
        "seasonal_lags": seasonal_lags,             # 유의성이 관찰된 계절 시차
    }

    # --- 4) 판정 결과 출력 ---
    if verbose:
        rdf = DataFrame([{
            "AR 차수 후보 (p)": p,
            "MA 차수 후보 (q)": q,
            "계절 성분": "존재" if result["seasonal"] else "없음",
            "유의한 계절 시차": ", ".join(map(str, seasonal_lags)) if seasonal_lags else "-",
        }], index=["ACF / PACF 판정"])

        display(rdf.T)

        if result["seasonal"]:
            display(Markdown(
                f"> 계절 주기 **s = {period}** 의 배수 시차에서 자기상관이 유의하다. "
                f"비계절 ARIMA가 아니라 **SARIMA**로 확장해야 한다."))

    return result


def plot_acf_pacf(data, y=None, lags=36, p=None, q=None, title=None,
                  width=1280, height=400, save_path=None):
    """ACF와 PACF를 위아래로 나란히 그려 차수 판정 근거를 시각화한다.

    파란 음영은 신뢰구간이다. 막대가 음영 **안**에 있으면 그 시차의 상관은 유의하지 않다.

    | 패턴 | 판정 |
    |---|---|
    | ACF 서서히 감소, PACF는 p 이후 절단 | AR(p) 모형 |
    | ACF는 q 이후 절단, PACF 서서히 감소 | MA(q) 모형 |
    | 둘 다 서서히 감소 | ARMA(p, q) 모형 |

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 정상성을 만족한 상태여야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        lags (int): 표시할 최대 시차 (기본값: 36).
        p (int): PACF 그래프에 표시할 AR 차수 후보. None이면 표시하지 않는다 (기본값: None).
        q (int): ACF 그래프에 표시할 MA 차수 후보. None이면 표시하지 않는다 (기본값: None).
        title (str): 그래프 전체 제목 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 한 칸의 높이 (기본값: 400).
        save_path (str): 그래프 저장 경로 (기본값: None).
    """
    # --- 1) 2단 그래프 초기화 ---
    s = _to_series(data, y).dropna()

    fig, ax = my_plot.init(rows=2, cols=1, width=width, height=height,
                           title=title if title else "ACF / PACF")

    # --- 2) ACF, PACF 그리기 (ax 파라미터 필수) ---
    plot_acf(s, lags=lags, ax=ax[0])
    ax[0].set_title(f"ACF - MA 차수(q) 판정용")
    ax[0].set_xlabel("Lag")

    plot_pacf(s, lags=lags, ax=ax[1])
    ax[1].set_title(f"PACF - AR 차수(p) 판정용")
    ax[1].set_xlabel("Lag")

    # --- 3) 차수 후보를 수직선으로 표시 ---
    for a, order, label in [(ax[0], q, "MA(q)"), (ax[1], p, "AR(p)")]:
        if order is None:
            continue

        a.axvline(x=order, linestyle="--", linewidth=1.5, alpha=0.8, color="red")
        a.text(order + 0.3, a.get_ylim()[1] * 0.85, f"{label} = {order}",
               verticalalignment="top", color="red")

    my_plot.show(save_path=save_path)


# ---------------------------------------------------------------------
# 4. ARIMA 모형 적합
# ---------------------------------------------------------------------

def train_test_split(data, test_size=0.2):
    """시계열을 **시간 순서를 유지한 채** 학습용과 검증용으로 분할한다.

    일반적인 무작위 분할을 쓰면 미래 값으로 과거를 예측하게 되어(data leakage)
    성능이 실제보다 좋게 나온다. 시계열은 반드시 앞뒤로 잘라야 한다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 시간 순으로 정렬되어 있어야 한다.
        test_size (float): 검증 데이터의 비율 (기본값: 0.2).
            뒤쪽 20%가 검증용이 된다.

    Returns:
        tuple: (train, test) 형태의 튜플. 입력과 같은 자료형으로 반환된다.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size는 0과 1 사이의 값이어야 합니다.")

    cut = int(len(data) * (1 - test_size))      # 분할 지점
    return data[:cut], data[cut:]


def fit_model(data, y=None, order=(1, 1, 1), seasonal_order=None, summary=False):
    """지정한 차수로 ARIMA(또는 SARIMA) 모형을 적합한다.

    차분은 `d`, `D` 파라미터가 모형 내부에서 자동으로 수행하므로
    **차분하지 않은 원본 시계열**을 그대로 전달해야 한다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 인덱스가 날짜형이어야 한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        order (tuple): 비계절 차수 (p, d, q) (기본값: (1, 1, 1)).
            - p: AR 차수 (과거 **값**을 몇 개 쓸 것인가)
            - d: 차분 횟수
            - q: MA 차수 (과거 **오차**를 몇 개 쓸 것인가)
        seasonal_order (tuple): 계절 차수 (P, D, Q, s). None이면 비계절 ARIMA (기본값: None).
        summary (bool): 적합 결과 요약표를 출력할지 여부 (기본값: False).

    Returns:
        적합이 완료된 ARIMA 분석 결과 객체.
    """
    # --- 1) 대상 시계열 확보 ---
    s = _to_series(data, y)

    # --- 2) 모형 생성 및 적합 ---
    if seasonal_order:
        model = ARIMA(s, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(s, order=order)

    # 수렴 경고는 그리드 탐색에서 따로 처리하므로 여기서는 표시하지 않는다
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit()

    if summary:
        print(fit.summary())

    return fit


def auto_arima(data, y=None, max_p=2, max_q=2, d=1, max_P=1, max_Q=1, D=0, s=None,
               criterion="aic", report=True, top=10):
    """차수 조합을 모두 시도해 정보량 기준이 가장 낮은 ARIMA 모형을 선택한다.

    비계절 차수 (p, q)와 계절 차수 (P, Q)를 **독립적으로** 탐색한다.
    두 값을 같은 값으로 묶으면 탐색 공간이 왜곡되므로 반드시 분리해야 한다.

    Args:
        data: 시계열 데이터프레임 또는 시리즈. 차분하지 않은 원본을 전달한다.
        y (str): `data`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        max_p (int): 비계절 AR 차수의 최대값 (기본값: 2).
        max_q (int): 비계절 MA 차수의 최대값 (기본값: 2).
        d (int): 비계절 차분 횟수. 정상성 진단 결과로 결정한다 (기본값: 1).
        max_P (int): 계절 AR 차수의 최대값 (기본값: 1).
        max_Q (int): 계절 MA 차수의 최대값 (기본값: 1).
        D (int): 계절 차분 횟수 (기본값: 0).
        s (int): 계절 주기. None이면 비계절 ARIMA만 탐색한다 (기본값: None).
        criterion (str): 모형 선택 기준. "aic" 또는 "bic" (기본값: "aic").
            - AIC: 예측 성능 중심. 상대적으로 복잡한 모형을 선택하는 경향
            - BIC: 간결성 중심. 표본이 클수록 더 단순한 모형을 선택
        report (bool): 탐색 결과표를 출력할지 여부 (기본값: True).
        top (int): 결과표에 표시할 상위 모형 수 (기본값: 10).

    Returns:
        최적 차수로 적합된 ARIMA 분석 결과 객체.
        탐색 결과표는 `.search_`, 선택 기준은 `.criterion_` 속성에 저장된다.

    Raises:
        ValueError: criterion이 "aic" 또는 "bic"가 아닌 경우.
        RuntimeError: 모든 조합이 적합에 실패한 경우.
    """
    # --- 1) 파라미터 검증 ---
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion은 'aic' 또는 'bic'이어야 합니다.")

    series = _to_series(data, y)

    # 계절 주기가 없으면 계절 차수 탐색을 생략한다
    if not s:
        max_P = max_Q = D = 0

    # --- 2) 모든 차수 조합을 적합하며 정보량 기준 수집 ---
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    order = (p, d, q)
                    seasonal = (P, D, Q, s) if s else None

                    try:
                        fit = fit_model(series, order=order, seasonal_order=seasonal)

                        # 수렴하지 않은 모형은 계수를 신뢰할 수 없으므로 제외한다
                        if not fit.mle_retvals.get("converged", True):
                            continue

                        rows.append({
                            "p": p, "d": d, "q": q,
                            "P": P, "D": D, "Q": Q, "s": s if s else 0,
                            "AIC": round(fit.aic, 3),
                            "BIC": round(fit.bic, 3),
                            "로그우도": round(fit.llf, 3),
                        })
                    except Exception:
                        # 적합 실패(특이행렬·발산 등)한 조합은 건너뛴다
                        continue

    if not rows:
        raise RuntimeError("적합에 성공한 차수 조합이 없습니다. d·D·s 값을 다시 확인하세요.")

    # --- 3) 정보량 기준으로 정렬하여 최적 모형 선택 ---
    key = criterion.upper()
    sdf = DataFrame(rows).sort_values(key).reset_index(drop=True)
    best = sdf.iloc[0]

    best_order = (int(best["p"]), int(best["d"]), int(best["q"]))
    best_seasonal = (int(best["P"]), int(best["D"]), int(best["Q"]), int(best["s"])) if s else None

    fit = fit_model(series, order=best_order, seasonal_order=best_seasonal)

    # 탐색 이력을 결과 객체에 붙여 둔다 (report_model에서 사용)
    fit.search_ = sdf
    fit.criterion_ = key

    # --- 4) 탐색 결과 보고 ---
    if report:
        display(Markdown(f"#### ▶︎ 차수 탐색 결과 (총 {len(sdf)}개 조합, {key} 오름차순 상위 {min(top, len(sdf))}개)"))
        display(sdf.head(top))

        label = f"ARIMA{best_order}"
        if best_seasonal:
            label += f"×{best_seasonal}"

        display(Markdown(f"> **선택된 모형: {label}** ({key} = {best[key]})"))

    return fit


# ---------------------------------------------------------------------
# 5. 결과 보고
# ---------------------------------------------------------------------

def _model_label(fit):
    """적합 결과 객체에서 모형 표기 문자열을 만든다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.

    Returns:
        str: "ARIMA(1, 1, 1)×(1, 1, 1, 12)" 형태의 표기.
    """
    order = fit.model.order
    seasonal = fit.model.seasonal_order

    # 계절 차수가 모두 0이면 비계절 ARIMA다
    if seasonal and any(seasonal[:3]):
        return f"ARIMA{order}×{seasonal}"

    return f"ARIMA{order}"


def report_fitness(fit):
    """적합된 ARIMA 모형의 구조와 적합도를 보고 문장으로 생성해 반환한다.

    Args:
        fit: `fit_model` 또는 `auto_arima`로 적합된 ARIMA 분석 결과 객체.

    Returns:
        str: 모형 구조·적합도 보고 문장 (markdown).
    """
    # --- 1) 모형 구조 정보 추출 ---
    p, d, q = fit.model.order
    seasonal = fit.model.seasonal_order
    label = _model_label(fit)

    # --- 2) 구조 설명 문장 구성 ---
    structure = f"{d}차 차분 후 AR({p}), MA({q})"

    if seasonal and any(seasonal[:3]):
        P, D, Q, s = seasonal
        structure += f", 주기 {s}의 계절 차분 {D}회와 계절 AR({P}), 계절 MA({Q})"

    # --- 3) 보고 문장 템플릿 ---
    template = (
        "**Note. n = {n}. "
        "Log Likelihood = {llf}, "
        "AIC = {aic}, "
        "BIC = {bic}, "
        "HQIC = {hqic}**\n\n"
        "{label} 모형을 적합하였다. 이는 {structure}를 결합한 구조다.\n\n"
        "> AIC = {aic}, BIC = {bic}\n\n"
        "AIC와 BIC는 값 자체에 절대적인 의미가 없으며, **같은 데이터에 적합한 다른 모형과 "
        "비교할 때만** 의미를 갖는다. 두 값 모두 작을수록 좋은 모형이다."
    )

    return template.format(
        n=int(fit.nobs),
        llf=round(fit.llf, 3),
        aic=round(fit.aic, 3),
        bic=round(fit.bic, 3),
        hqic=round(fit.hqic, 3),
        label=label,
        structure=structure,
    )


def report_variables(fit, alpha=0.05):
    """적합된 ARIMA 모형의 계수 보고표를 데이터프레임으로 생성해 반환한다.

    계수 이름의 의미:
        - `ar.L1` : 비계절 AR 1차 - "직전 시점의 **값**"이 현재에 미치는 영향
        - `ma.L1` : 비계절 MA 1차 - "직전 시점의 **예측 오차**"가 현재에 미치는 영향
        - `ar.S.L12` : 계절 AR - "12시점 전(작년 같은 달)의 값"이 미치는 영향
        - `ma.S.L12` : 계절 MA - "12시점 전의 예측 오차"가 미치는 영향
        - `sigma2` : 잔차의 분산 추정치

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        DataFrame: 변수·계수·표준오차·z·유의확률·신뢰구간 하한/상한·유의성 컬럼의 보고표.
    """
    # --- 1) 계수 통계량 추출 ---
    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    conf = fit.conf_int(alpha=alpha)

    # --- 2) 계수별 보고 내용 정리 ---
    rows = []
    for name in params.index:
        p = float(pvalues[name])

        # 유의수준별 별표 표기 (사회과학 논문 관행)
        if p < 0.001:   stars = "***"
        elif p < 0.01:  stars = "**"
        elif p < 0.05:  stars = "*"
        else:           stars = ""

        rows.append({
            "변수": name,                                       # 계수 이름
            "계수": round(float(params[name]), 4),               # 추정된 계수값
            "표준오차": round(float(bse[name]), 4),              # 계수의 표준오차
            "z": f"{float(params[name] / bse[name]):.3f}{stars}",  # z 통계량 + 유의성 별표
            "유의확률": round(p, 4),                             # p-value
            "CI 하한": round(float(conf.loc[name].iloc[0]), 4),   # 신뢰구간 하한
            "CI 상한": round(float(conf.loc[name].iloc[1]), 4),   # 신뢰구간 상한
            "유의성": p < alpha,                                 # 유의수준 기준 유의 여부
        })

    return DataFrame(rows)


def test_residual(fit, alpha=0.05, plot=True, width=1280, height=800, save_path=None):
    """적합된 ARIMA 모형의 잔차를 세 가지 관점에서 진단한다.

    **좋은 모형의 잔차는 백색잡음이어야 한다.** 즉 아무 패턴도 남아 있지 않아야 한다.
    패턴이 남아 있다면 모형이 아직 설명하지 못한 구조가 있다는 뜻이다.

    | 검정 | 귀무가설 | 통과 조건 |
    |---|---|---|
    | Ljung-Box | 잔차에 자기상관이 없다 | **p ≥ 0.05** (가장 중요) |
    | Jarque-Bera | 잔차가 정규분포를 따른다 | p ≥ 0.05 |
    | 이분산 검정 | 잔차의 분산이 일정하다 | p ≥ 0.05 |

    세 검정 모두 **p가 커야 통과**다. ADF 검정과 방향이 반대라는 점에 주의한다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.
        alpha (float): 유의수준 (기본값: 0.05).
        plot (bool): 잔차 진단 그래프(잔차·히스토그램·Q-Q·ACF)를 그릴지 여부 (기본값: True).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 800).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: 검정별 통계량·유의확률·통과 여부·해석 컬럼의 진단표.
    """
    # --- 1) 세 가지 잔차 검정 수행 ---
    # 각 검정은 (1, k) 또는 (1, 2, k) 형태의 배열을 반환하므로 마지막 값을 사용한다
    lb = np.asarray(fit.test_serial_correlation(method="ljungbox"))
    lb_stat, lb_p = float(lb[0][0][-1]), float(lb[0][1][-1])

    jb = np.asarray(fit.test_normality(method="jarquebera"))
    jb_stat, jb_p = float(jb[0][0]), float(jb[0][1])

    het = np.asarray(fit.test_heteroskedasticity(method="breakvar"))
    het_stat, het_p = float(het[0][0]), float(het[0][1])

    # --- 2) 진단표 구성 ---
    tests = [
        ("Ljung-Box (자기상관)", lb_stat, lb_p,
         "잔차에 시간적 패턴이 남아 있지 않다", "잔차에 자기상관이 남아 있어 모형이 불충분하다"),
        ("Jarque-Bera (정규성)", jb_stat, jb_p,
         "잔차가 정규분포를 따른다", "잔차가 정규분포에서 벗어난다 (예측구간 신뢰도 저하)"),
        ("이분산 (Break Variance)", het_stat, het_p,
         "잔차의 분산이 일정하다", "잔차의 분산이 일정하지 않다 (로그변환 검토)"),
    ]

    rows = []
    for name, stat, p, ok_text, ng_text in tests:
        passed = bool(p >= alpha)
        rows.append({
            "검정": name,                                   # 검정 이름
            "통계량": round(stat, 4),                        # 검정통계량
            "유의확률": round(p, 4),                         # p-value
            "통과": passed,                                  # 통과 여부 (p >= alpha)
            "해석": ok_text if passed else ng_text,          # 결과 해석 문장
        })

    rdf = DataFrame(rows)
    display(rdf)

    # --- 3) 종합 판정 ---
    if bool(rdf.loc[0, "통과"]):
        display(Markdown(
            "> ✅ **Ljung-Box 검정을 통과했다.** 잔차에 자기상관이 남아 있지 않으므로 "
            "모형이 시계열의 구조를 충분히 설명했다고 볼 수 있다."))
    else:
        display(Markdown(
            "> ❌ **Ljung-Box 검정을 통과하지 못했다.** 잔차에 아직 패턴이 남아 있다. "
            "차수(p, q)를 높이거나 계절 성분을 추가하는 것을 검토한다."))

    # --- 4) 잔차 진단 그래프 ---
    if plot:
        resid = fit.resid

        # 차분으로 소실되는 앞부분은 잔차가 의미 없는 값이므로 제외한다
        burn = _burn_in(fit)
        resid = resid[burn:]

        fig, ax = my_plot.init(rows=2, cols=2, width=width // 2, height=height // 2,
                               title="잔차 진단")

        # (1) 시간에 따른 잔차 - 패턴이 보이면 안 된다
        my_plot.lineplot(x=resid.index, y=resid, ax=ax[0])
        ax[0].axhline(0, color="red", linestyle="--", linewidth=1)
        ax[0].set_title("잔차 (0 주변에 무작위로 흩어져야 한다)")

        # (2) 잔차 분포 - 정규분포 모양이어야 한다
        sb.histplot(x=resid, kde=True, ax=ax[1])
        ax[1].set_title("잔차 분포 (정규분포에 가까워야 한다)")

        # (3) Q-Q 플롯 - 직선에 가까워야 한다
        from scipy.stats import probplot
        probplot(resid, dist="norm", plot=ax[2])
        ax[2].set_title("Q-Q 플롯 (직선에 가까워야 한다)")
        ax[2].get_lines()[0].set_markersize(3)

        # (4) 잔차의 ACF - 막대가 신뢰구간 안에 있어야 한다
        plot_acf(resid, ax=ax[3])
        ax[3].set_title("잔차의 ACF (모두 신뢰구간 안에 있어야 한다)")

        my_plot.show(save_path=save_path)

    return rdf


def _burn_in(fit):
    """차분으로 인해 의미가 없어지는 앞부분 관측치 개수를 계산한다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.

    Returns:
        int: 제외해야 할 앞부분 관측치 수 (d + D × s).
    """
    d = fit.model.order[1]
    seasonal = fit.model.seasonal_order

    if seasonal and len(seasonal) == 4:
        return d + seasonal[1] * seasonal[3]

    return d


def report_performance(fit, test=None, y=None, inverse=None, verbose=True):
    """예측 오차 지표(MAE·RMSE·MAPE)를 계산하여 모형의 예측 성능을 보고한다.

    `test`를 주면 **모형이 본 적 없는 구간**에 대한 성능(예측 성능)을,
    주지 않으면 학습 구간에 대한 성능(적합 성능)을 계산한다.
    **적합 성능은 항상 예측 성능보다 좋게 나오므로** 모형 평가의 근거로 쓸 수 없다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.
        test: 검증용 시계열(데이터프레임 또는 시리즈). None이면 학습 구간으로 평가한다 (기본값: None).
        y (str): `test`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        inverse (callable): 변환된 데이터로 적합한 경우의 역변환 함수 (기본값: None).
            로그변환한 시계열로 모형을 만들었다면 `np.exp`를 전달한다.
            지표를 **원래 단위로 환산**해야 실무적으로 해석할 수 있다.
            `test`도 같은 변환이 적용된 상태여야 한다.
        verbose (bool): 결과표를 화면에 출력할지 여부 (기본값: True).

    Returns:
        DataFrame: MAE·RMSE·MAPE·평가 구간 정보를 담은 성능표.
    """
    # --- 1) 실제값과 예측값 확보 ---
    if test is not None:
        actual = _to_series(test, y)
        # 검증 구간 길이만큼 미래를 예측한다
        predicted = fit.forecast(len(actual))
        scope = "검증 구간 (예측 성능)"
    else:
        burn = _burn_in(fit)
        actual = fit.model.endog[burn:]
        actual = Series(np.asarray(actual).ravel())
        predicted = Series(np.asarray(fit.fittedvalues)[burn:])
        scope = "학습 구간 (적합 성능)"

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # 변환된 스케일로 적합한 모형은 지표를 원래 단위로 되돌려야 해석할 수 있다
    if inverse is not None:
        actual = np.asarray(inverse(actual), dtype=float)
        predicted = np.asarray(inverse(predicted), dtype=float)
        scope += " · 역변환"

    # --- 2) 오차 지표 계산 ---
    error = actual - predicted

    mae = float(np.mean(np.abs(error)))                             # 평균절대오차
    rmse = float(np.sqrt(np.mean(error ** 2)))                      # 평균제곱근오차

    # MAPE는 실제값이 0이면 계산할 수 없으므로 0이 아닌 값만 사용한다
    nonzero = actual != 0
    mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan

    # --- 3) 성능표 구성 ---
    rdf = DataFrame([{
        "평가 구간": scope,                          # 어떤 구간을 평가했는가
        "관측치 수": len(actual),                    # 평가에 사용한 관측치 수
        "MAE": round(mae, 4),                       # 평균절대오차 (원래 단위)
        "RMSE": round(rmse, 4),                     # 평균제곱근오차 (큰 오차에 민감)
        "MAPE(%)": round(mape, 4),                  # 평균절대백분율오차 (단위 무관)
    }], index=[_model_label(fit)])

    if verbose:
        display(rdf.T)
        display(Markdown(
            f"> MAPE = {mape:.2f}% 는 예측값이 실제값에서 평균 {mape:.2f}% 만큼 "
            f"벗어난다는 뜻이다. 단위가 없으므로 **다른 데이터셋과도 비교할 수 있다.**"))

    return rdf


def forecast(fit, periods=12, alpha=0.05, inverse=None):
    """학습 구간 이후의 미래값을 예측하고 신뢰구간과 함께 반환한다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.
        periods (int): 예측할 기간 수. 월간 데이터에서 12는 1년을 의미한다 (기본값: 12).
        alpha (float): 신뢰구간의 유의수준. 0.05면 95% 신뢰구간 (기본값: 0.05).
        inverse (callable): 변환된 데이터로 적합한 경우의 역변환 함수 (기본값: None).
            로그변환한 시계열로 모형을 만들었다면 `np.exp`를 전달한다.

    Returns:
        DataFrame: 예측값·신뢰구간 하한/상한 컬럼을 담은 데이터프레임.
            인덱스는 예측 시점이다.
    """
    # --- 1) 예측 수행 ---
    fc = fit.get_forecast(periods)
    mean = fc.predicted_mean                    # 예측값
    conf = fc.conf_int(alpha=alpha)             # 신뢰구간

    # --- 2) 결과표 구성 ---
    level = int((1 - alpha) * 100)
    fdf = DataFrame({
        "예측값": mean.values,
        f"{level}% 하한": conf.iloc[:, 0].values,
        f"{level}% 상한": conf.iloc[:, 1].values,
    }, index=mean.index)

    # --- 3) 역변환 ---
    # 로그 등 단조증가 변환은 신뢰구간의 순서를 바꾸지 않으므로 그대로 되돌리면 된다
    if inverse is not None:
        fdf = fdf.apply(inverse)

    return fdf


def plot_forecast(fit, periods=12, test=None, y=None, alpha=0.05, inverse=None, title=None,
                  xlabel=None, ylabel=None, width=1600, height=520, save_path=None):
    """관측값·적합값·미래 예측값을 신뢰구간과 함께 한 그래프에 그린다.

    Args:
        fit: 적합된 ARIMA 분석 결과 객체.
        periods (int): 예측할 기간 수 (기본값: 12).
        test: 검증용 시계열. 주면 실제값을 함께 표시한다 (기본값: None).
        y (str): `test`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        alpha (float): 신뢰구간의 유의수준 (기본값: 0.05).
        inverse (callable): 변환된 데이터로 적합한 경우의 역변환 함수 (기본값: None).
            로그변환한 시계열로 모형을 만들었다면 `np.exp`를 전달한다.
            그래프 전체가 **원래 단위**로 그려진다.
        title (str): 그래프 제목 (기본값: None → 자동 생성).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1600).
        height (int): 그래프 높이 (기본값: 520).
        save_path (str): 그래프 저장 경로 (기본값: None).

    Returns:
        DataFrame: `forecast` 함수가 반환하는 예측 결과표.
    """
    # --- 1) 관측값·적합값·예측값 준비 ---
    # 적합값은 관측값과 인덱스가 같으므로 그 인덱스를 그대로 사용한다
    fitted = fit.fittedvalues
    observed = Series(np.asarray(fit.model.endog).ravel(), index=fitted.index)

    # 차분으로 소실되는 앞부분의 적합값은 의미가 없으므로 제외한다
    burn = _burn_in(fit)
    fitted = fitted[burn:]

    # 검증 데이터가 있으면 그 길이만큼, 없으면 periods 만큼 예측한다
    n = len(test) if test is not None else periods
    fdf = forecast(fit, periods=n, alpha=alpha, inverse=inverse)

    # 관측값과 적합값도 같은 단위로 맞춘다
    if inverse is not None:
        observed = observed.apply(inverse)
        fitted = fitted.apply(inverse)

    # --- 2) 그래프 그리기 ---
    fig, ax = my_plot.init(width=width, height=height,
                           title=title if title else f"{_model_label(fit)} 예측 결과",
                           xlabel=xlabel, ylabel=ylabel)

    my_plot.lineplot(x=observed.index, y=observed, label="관측값", ax=ax)
    my_plot.lineplot(x=fitted.index, y=fitted, label="적합값", linestyle="--", ax=ax)
    my_plot.lineplot(x=fdf.index, y=fdf["예측값"], label="예측값", linestyle="--", ax=ax)

    # 신뢰구간은 음영으로 표시한다
    ax.fill_between(fdf.index, fdf.iloc[:, 1], fdf.iloc[:, 2],
                    alpha=0.2, label=f"{int((1 - alpha) * 100)}% 신뢰구간")

    # 검증 데이터가 있으면 실제값을 함께 표시한다
    if test is not None:
        actual = _to_series(test, y)

        if inverse is not None:
            actual = actual.apply(inverse)

        my_plot.lineplot(x=actual.index, y=actual, label="실제값(검증)", ax=ax)

    ax.legend()
    my_plot.show(save_path=save_path)

    return fdf


def report_model(fit, periods=12, test=None, y=None, inverse=None, plot=True):
    """적합된 ARIMA 모형의 보고와 진단을 한 번에 출력한다.

    출력 구성:
        ### ▶︎ 성능 보고
            #### 1) 모형 적합도  2) 계수 보고표  3) 예측 성능  4) 예측 결과 시각화
        ---
        ### ▶︎ 잔차 진단

    Args:
        fit: `fit_model` 또는 `auto_arima`로 적합된 ARIMA 분석 결과 객체.
        periods (int): 미래 예측 기간 수 (기본값: 12).
        test: 검증용 시계열. 있으면 예측 성능을 함께 평가한다 (기본값: None).
        y (str): `test`가 데이터프레임일 때 분석 대상 컬럼명 (기본값: None).
        inverse (callable): 변환된 데이터로 적합한 경우의 역변환 함수 (기본값: None).
            로그변환한 시계열로 모형을 만들었다면 `np.exp`를 전달한다.
        plot (bool): 그래프를 함께 그릴지 여부 (기본값: True).
    """
    # --- 1) 제목 출력 함수 ---
    def heading(text):
        print()
        display(Markdown(text))

    heading(f"## 최종 모델: {_model_label(fit)}")

    # --- 2) 성능 보고 ---
    heading("### ▶︎ 성능 보고")

    heading("#### 1) 모형 적합도")
    display(Markdown(report_fitness(fit)))

    heading("#### 2) 계수 보고표")
    display(report_variables(fit))

    heading("#### 3) 예측 성능")
    report_performance(fit, test=test, y=y, inverse=inverse)

    if plot:
        heading("#### 4) 예측 결과 시각화")
        plot_forecast(fit, periods=periods, test=test, y=y, inverse=inverse)

    # --- 3) 성능 보고와 잔차 진단 사이 구분선 ---
    display(Markdown("---"))

    # --- 4) 잔차 진단 ---
    heading("### ▶︎ 잔차 진단")
    test_residual(fit, plot=plot)
