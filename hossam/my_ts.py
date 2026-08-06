import numpy as np
from IPython.display import display
from pandas import DataFrame, Series, Grouper, concat, to_datetime

# 정상성 검정 · 자기상관 계수 계산
from statsmodels.tsa.stattools import adfuller

# 시계열 분해
from statsmodels.tsa.seasonal import seasonal_decompose

# 자기상관 그래프
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from . import my_plot


# ===================================================================
# [1단원] 시계열 인덱스 설정
# ===================================================================
def set_index(data, column, freq=None, sort=True):
    """날짜 컬럼을 인덱스로 지정하고 관측 간격을 명시한다.

    Args:
        data (DataFrame): 날짜 컬럼을 포함하는 데이터프레임.
        column (str): 인덱스로 사용할 날짜 컬럼명.
        freq (str): 관측 간격 (예: "MS", "D", "W"). None이면 지정하지 않음 (기본값: None).
        sort (bool): 인덱스를 오름차순으로 정렬할지 여부 (기본값: True).

    Returns:
        DataFrame: 날짜 인덱스가 설정된 데이터프레임.
    """
    df = data.copy()

    # --- 1) 날짜 컬럼이 문자열이면 datetime으로 변환 ---
    if df[column].dtype == "object":
        df[column] = to_datetime(df[column])

    # --- 2) 인덱스로 지정 ---
    df = df.set_index(column)

    if sort:
        df = df.sort_index()

    # --- 3) 관측 간격 명시 ---
    # 간격을 명시해야 window·period 값이 "시간"을 뜻하게 된다.
    # 간격이 없으면 12는 "12개월"이 아니라 그냥 "관측치 12개"일 뿐이다.
    if freq is not None:
        df = df.asfreq(freq)

    return df


# ===================================================================
# [1단원] 구간별 통계 보고
# ===================================================================
def report_split(data, column=None, size=3):
    """시계열을 여러 구간으로 잘라 구간별 평균·표준편차·변동계수를 비교한다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        size (int): 나눌 구간의 개수 (기본값: 3).

    Returns:
        DataFrame: 구간별 시작·종료·관측치 수·평균·표준편차·변동계수.
    """
    if column is not None:          # data가 데이터프레임이면 대상 컬럼만 추출
        data = data[column]

    length = len(data) // size      # 구간 하나의 길이
    result = []                     # 결과를 담을 리스트

    for i in range(size):
        # i번째 구간을 잘라낸다
        part = data.iloc[i * length : (i + 1) * length]

        result.append({
            "구간": f"구간 {i + 1}",
            "시작": part.index[0].strftime("%Y-%m-%d"),
            "종료": part.index[-1].strftime("%Y-%m-%d"),
            "관측치 수": len(part),
            "평균": round(part.mean(), 3),
            "표준편차": round(part.std(), 3),
            "변동계수": round(part.std() / part.mean(), 4),
        })

    return DataFrame(result).set_index("구간")


# ===================================================================
# [1단원] 기간별 변동 보고
# ===================================================================
def report_variation(data, column=None, freq="YE"):
    """기간 단위로 묶어 변동폭이 수준에 비례해 커지는지 확인한다.

    표준편차는 커지는데 변동계수(표준편차/평균)가 일정하면 변동폭이 수준에 비례한다는 뜻이다.
    이 경우 로그변환이 필요하다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        freq (str): 묶을 기간 단위 (기본값: "YE").
            YE → 연말 기준 1년씩  /  QE → 분기씩  /  ME → 월말 기준 한 달씩, 
            WE → 주말 기준 1주일씩  /  D → 하루씩

    Returns:
        DataFrame: 기간별 평균·표준편차·변동계수.
    """
    # --- 0) 대상 컬럼 추출 ---
    if column is not None:
        data = data[column]

    # --- 1) 기간 단위로 묶어 평균과 표준편차를 계산 ---
    # groupby()는 "값이 같은 것끼리" 묶는데, 날짜는 값이 전부 달라서 그대로는 묶이지 않는다.
    # Grouper(freq=...)를 키로 주면 "이 시간 간격으로 잘라서 묶어라"라는 뜻이 된다.
    #   freq="YE" → 연말 기준 1년씩  /  "QE" → 분기씩  /  "ME" → 월말 기준 한 달씩
    # 인덱스는 각 구간의 끝 날짜가 된다. (1949년 묶음 → 1949-12-31)
    group = data.groupby(Grouper(freq=freq))

    result_df = DataFrame({
        "평균": group.mean().round(3),
        "표준편차": group.std().round(3),
    })

    # --- 2) 변동계수 = 표준편차 / 평균 ---
    result_df["변동계수"] = (result_df["표준편차"] / result_df["평균"]).round(4)
    result_df = result_df.dropna()
    result_df.index.name = "기간"

    # --- 3) 최대/최소 비율로 판정 ---
    std_ratio = result_df["표준편차"].max() / result_df["표준편차"].min()
    cv_ratio = result_df["변동계수"].max() / result_df["변동계수"].min()

    if cv_ratio < std_ratio:
        conclusion = "변동폭이 수준에 비례한다 → 로그변환 권장 / 승법(multiplicative) 모델"
    else:
        conclusion = "변동폭이 일정하다 → 원본 사용 가능 / 가법(additive) 모델"

    print(f"표준편차 최대/최소: {std_ratio:.2f}배")
    print(f"변동계수 최대/최소: {cv_ratio:.2f}배")
    print(f"판정: {conclusion}")

    return result_df


# ===================================================================
# [1단원] ADF 검정
# ===================================================================
def adf_test(data, column=None, name=None, alpha=0.05):
    """시계열 하나의 정상성을 ADF 검정으로 판정한다.

    여러 대상을 비교하는 adf_diff · adf_transform 도 이 함수를 반복 호출한다.
    반환값이 가로 1행 표이므로 concat() 으로 이어 붙이면 그대로 여러 행이 된다.

    Args:
        data (DataFrame | Series): 검정할 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        name (str): 결과표에 표시할 행 이름 (기본값: None).
            생략하면 컬럼명 → Series의 이름 → "시계열" 순으로 찾아 쓴다.
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        DataFrame: 대상을 인덱스로 하는 가로 1행 ADF 검정 결과표.
    """
    # --- 1) 기본 준비 및 정상성 검정 ---
    if column is not None:              # 컬럼명이 있다면
        data = data[column]             # 대상 데이터만 추출

    # 표의 행 이름 가져오기
    if name is None:                    # 이름을 직접 주지 않았다면
        if column is not None:
            name = column               # 컬럼명을 행 이름으로 사용
        else:
            name = data.name            # Series의 이름을 행 이름으로 사용

    if name is None:  name = "시계열"    # 그래도 이름이 없으면 그냥 "시계열"로 지정

    x = Series(data).dropna()           # 결측치 제거

    # adfuller()는 결과를 튜플로 돌려준다
    statistic, pvalue, usedlag, nobs, cvalues, icbest = adfuller(x)

    # --- 2) 결과 구성 및 반환 ---
    stationary = bool(pvalue < alpha)   # p-value가 작아야 정상성 확보

    # 결과표 반환
    return DataFrame([{
        "관측치 수": len(x),
        "검정통계량(ADF)": round(statistic, 3),
        "p-value": round(pvalue, 4),
        "사용 시차": usedlag,
        "1% 기각값": round(cvalues["1%"], 3),
        "5% 기각값": round(cvalues["5%"], 3),
        "10% 기각값": round(cvalues["10%"], 3),
        "표준편차": round(x.std(), 3),
        "정상성": stationary,
        "판정": "정상" if stationary else "비정상",
    }], index=[name])


# ===================================================================
# [1단원] 시계열 변환
# ===================================================================
def transform(data, column=None, log=False, diff=0, seasonal_diff=0, period=None):
    """로그변환 · 일반차분 · 계절차분을 순서대로 적용한다.

    Args:
        data (DataFrame | Series): 변환할 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        log (bool): 로그변환 적용 여부 (기본값: False).
        diff (int): 일반차분 횟수. ARIMA의 d에 해당 (기본값: 0).
        seasonal_diff (int): 계절차분 횟수. ARIMA의 D에 해당 (기본값: 0).
        period (int): 계절 주기. seasonal_diff가 1 이상이면 반드시 지정 (기본값: None).

    Returns:
        Series: 변환이 완료된 시계열. 차분으로 생긴 결측치는 제거된다.
    """
    # --- 0) 대상 컬럼 추출 ---
    if column is not None:
        data = data[column]

    result = data.copy()

    # --- 1) 로그변환 : 분산이 커지는 문제를 잡는다 ---
    if log:
        result = np.log(result)

    # --- 2) 일반차분 : 추세를 없앤다 ---
    for i in range(diff):
        result = result.diff()

    # --- 3) 계절차분 : 주기적 반복을 없앤다 ---
    for i in range(seasonal_diff):
        result = result.diff(period)

    return result.dropna()


# ===================================================================
# [1단원] 차분 차수별 ADF 검정
# ===================================================================
def adf_diff(data, column=None, max_diff=-1, alpha=0.05):
    """차분 차수를 늘려가며 ADF 검정을 수행하고 과대차분 여부를 판정한다.

    표준편차가 최소가 되는 차수를 넘기면 데이터에 없던 노이즈를 만들어낸다.
    p-value를 낮추는 것이 목적이 아니라 필요한 만큼만 차분하는 것이 목적이다.

    Args:
        data (DataFrame | Series): 검정할 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        max_diff (int): 최대 차분 횟수. -1이면 표준편차가 다시 커지는 지점,
            즉 과대차분이 시작되는 곳에서 스스로 멈춘다 (기본값: -1).
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        DataFrame: 차수별 ADF 검정 결과와 권장 표시.
    """
    # --- 0) 대상 컬럼 추출 ---
    if column is not None:      # 컬럼 이름이 있다면
        data = data[column]     # 대상 데이터만 추출

    result = []                 # 결과를 담을 리스트
    target = data.copy()        # 검정할 대상 시계열. 차분할 때마다 관측치가 하나씩 줄어든다.

    order = 0                   # 지금까지 차분한 횟수
    previous_std = None         # 직전 차수의 표준편차
    stopped = None              # 과대차분이 검출된 차수 이름

    # --- 1) 0차(원본)부터 차수를 늘려가며 검정 ---
    while True:
        # --- 1-1) ADF 검정 ---
        name = "원본" if order == 0 else f"{order}차 차분"  # 차분 회차별 이름 구성
        row = adf_test(target, name=name, alpha=alpha)     # 검정 수행
        result.append(row)                                 # 검정 결과를 리스트에 추가
        current_std = row["표준편차"].iloc[0]               # 검정 결과에서 표준편차만 가져오기

        # --- 1-2) 멈출 때가 되었는지 확인 ---
        if max_diff < 0:            # 최대 차분 횟수를 지정하지 않았다면
            # --- 표준편차가 직전보다 커졌다면 이미 한 번 지나친 것이므로 여기서 멈춘다
            if previous_std is not None and current_std > previous_std:
                stopped = name      # --- 과대차분이 시작된 차수 이름 기록
                break               # --- 반복을 멈춘다
        elif order >= max_diff:     # 최대 차분 횟수를 지정했고 그 횟수에 도달했다면
            break                   # --- 반복을 멈춘다

        # --- 1-3) 다음 차수 준비 (반복을 멈춘 경우 아래 코드는 실행 안됨) ---
        previous_std = current_std  # 직전 차수의 표준편차를 현재 차수의 표준편차로 갱신
        order = order + 1           # 차분 횟수 1 증가

        # 다음 차수를 위해 1차 차분 수행. 차분할 때마다 관측치가 하나씩 줄어든다.
        target = target.diff().dropna()

        # 관측치가 너무 적어지면 검정 자체가 의미를 잃으므로 멈춘다
        if len(target) < 10:
            break

    result_df = concat(result)            # 결과 리스트를 데이터프레임으로 합치기

    # --- 2) 표준편차가 최소인 차수 확인 --> 과대차분 시작 직전 지점 ---
    best = result_df["표준편차"].idxmin()                   # 표준편차가 최소인 차수 이름
    best_at = list(result_df.index).index(best)            # 그 차수가 몇 번째 행인지
    recommend = []                                          # 권장 표시를 담을 리스트

    for position, name in enumerate(result_df.index):       # 각 행을 순서대로 순회하며
        if position == best_at:                             # 표준편차가 최소인 차수라면
            recommend.append("★ 여기까지")                   # 여기서 멈추라는 표시
        elif position > best_at:                            # 최소점을 지난 차수라면
            recommend.append("과대차분")                     # 차분이 지나쳤다는 표시
        else:                                               # 최소점 이전이라면
            recommend.append("")                            # 빈 문자열을 넣는다

    result_df["권장"] = recommend           # 권장 표시를 결과표에 추가

    # --- 3) 판단 근거를 글로 알린다 ---
    if stopped is not None:                 # 자동으로 멈춘 경우라면
        print(f"{stopped}에서 표준편차가 다시 커져 중단했습니다. (과대차분 시작)")

    print(f"권장 차분 차수: {best} (표준편차가 최소인 지점)")

    # 권장 차수가 아직 비정상이면 차분을 더 하는 것이 답이 아님을 알린다.
    # 남은 원인은 대개 계절성과 분산이며, 그것은 차분 횟수로 풀 문제가 아니다.
    if not result_df.loc[best, "정상성"]:
        pvalue = result_df.loc[best, "p-value"]
        print(f"⚠️ {best}은 아직 비정상입니다(p-value {pvalue}). 차분을 더 늘리지 말고 "
              f"로그변환·계절차분으로 남은 구조를 처리하세요. → adf_transform()")

    return result_df                        # 결과표 반환


# ===================================================================
# [1단원] 전처리 조합별 ADF 검정
# ===================================================================
def adf_transform(data, column=None, period=None, max_diff=2, log=True, alpha=0.05):
    """로그변환 · 일반차분 · 계절차분의 조합을 만들어 전부 ADF 검정한다.

    정상성을 만족하는 조합 중 표준편차가 가장 작은 것이 맨 위에 온다.

    Args:
        data (DataFrame | Series): 검정할 시계열 데이터.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        period (int): 계절 주기. None이면 계절차분 조합을 만들지 않음 (기본값: None).
        max_diff (int): 조합에 포함할 최대 일반차분 횟수 (기본값: 2).
        log (bool): 로그변환 조합을 포함할지 여부 (기본값: True).
        alpha (float): 유의수준 (기본값: 0.05).

    Returns:
        Series: 권장 조합을 적용해 변환이 끝난 시계열. 조합 이름이 Series의 이름으로 붙는다.
            조합별 검정 결과표는 함수 안에서 화면에 출력한다.
    """
    # --- 1) 조합 후보 정의 ---
    if column is not None:          # 컬럼 이름이 있다면
        data = data[column]         # 대상 데이터만 추출

    log_options = [False]           # 로그변환 여부 후보. False만 넣어두고 True를 추가할지 결정

    if log:                         # 사용자가 로그변환 조합을 포함하라고 했다면
        log_options.append(True)    # True를 후보에 추가

    seasonal_options = [0]          # 주기를 모르면 계절차분은 시도하지 않는다

    if period is not None:          # 주기를 지정했다면
        seasonal_options.append(1)  # 계절차분 1회 조합을 후보에 추가

    result = []                     # 검정 결과를 담을 리스트
    changed = {}                    # 조합 이름별 변환 결과를 담을 딕셔너리

    # --- 2) 모든 조합을 순회하며 변환 후 검정 ---
    for use_log in log_options:                     # 로그변환 여부 후보 순회
        for diff in range(max_diff + 1):            # 일반차분 횟수 후보 순회
            for seasonal_diff in seasonal_options:  # 계절차분 여부 후보 순회

                # --- 2-1) 조합 이름 구성 ---
                parts = []
                if use_log:                     # 로그변환을 적용했다면 조합이름에 추가
                    parts.append("로그변환")

                if diff > 0:                    # 일반차분을 적용했다면 조합이름에 추가
                    parts.append(f"{diff}차 차분") 

                if seasonal_diff > 0:           # 계절차분을 적용했다면 조합이름에 추가
                    parts.append(f"계절차분({period})")

                if len(parts) == 0:             # 조합이름이 비어있다면 원본임
                    name = "원본"
                else:                           # 조합이름이 있다면 +로 이어붙인다
                    name = " + ".join(parts)

                # --- 2-2) 조합대로 변환 후 ADF 검정 ---
                target = data.copy()            # 조합대로 변환을 수행하기 위해 원본 복사

                if use_log:                     # 로그변환을 적용했다면 처리한다.
                    target = np.log(target)

                for i in range(diff):           # 일반차분을 적용했다면 처리한다.
                    target = target.diff()

                for i in range(seasonal_diff):  # 계절차분을 적용했다면 처리한다.
                    target = target.diff(period)

                target = target.dropna()        # 차분으로 생긴 결측치 제거
                target.name = name              # 변환 결과에 조합 이름을 붙인다

                changed[name] = target          # 나중에 골라 돌려주기 위해 보관
                result.append(adf_test(target, name=name, alpha=alpha))

    # --- 3) 정상성 우선, 같으면 표준편차가 작은 순으로 정렬 ---
    result_df = concat(result)
    result_df = result_df.sort_values(["정상성", "표준편차"], ascending=[False, True])

    best = result_df.index[0]
    best_std = result_df["표준편차"].iloc[0]

    print(f"조합 {len(result_df)}개 중 정상 {int(result_df['정상성'].sum())}개")
    print(f"권장 전처리: {best}")

    # --- 4) 과대차분 경고 ---
    # 권장한 조합보다 표준편차가 더 작은데 정상성만 못 갖춘 조합이 있다면,
    # 차분을 한 번 더 해서 정상성을 얻는 대신 없던 노이즈를 만들었을 수 있다.
    smaller = result_df[result_df["표준편차"] < best_std]

    if len(smaller) > 0:
        name = smaller["표준편차"].idxmin()
        std = smaller["표준편차"].min()
        print(f"⚠️ {name}의 표준편차({std})가 더 작습니다. "
              f"과대차분일 수 있으니 계절차분·로그변환을 함께 검토하세요.")

    display(result_df)      # 결과 표 출력

    return changed[best]    # 최적의 전처리 결과 반환


# ===================================================================
# [2단원] 이동평균 · 이동표준편차 시각화
# ===================================================================
def plot_rolling(data, window, column=None, title=None, xlabel=None, ylabel=None,
                 width=1280, height=480, save_path=None):
    """원본과 이동평균·이동표준편차를 한 그래프에 그려 정상성을 눈으로 확인한다.

    정상 시계열이라면 이동평균과 이동표준편차가 모두 수평선에 가까워야 한다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        window (int): 이동 계산에 사용할 창 크기. 계절 주기로 잡는다.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None).
        ylabel (str): y축 레이블 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀 (기본값: 480).
        save_path (str): 이미지 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본·이동평균·이동표준편차.
    """
    # --- 1) 대상 데이터 준비 ---
    if column is not None:      # 컬럼명이 전달되었다면
        data = data[column]     # 대상 컬럼만 추출

    # --- 2) 이동평균과 이동표준편차 계산 ---
    result_df = DataFrame({
        "원본": data,
        "이동평균": data.rolling(window).mean(),
        "이동표준편차": data.rolling(window).std(),
    })

    # --- 3) 그래프 그리기 및 결과 반환 ---
    if title is None:
        title = f"원본 · 이동평균 · 이동표준편차 (window={window})"

    fig, ax = my_plot.init(width=width, height=height, title=title,
                           xlabel=xlabel, ylabel=ylabel)

    for col in result_df.columns:
        my_plot.lineplot(x=result_df.index, y=result_df[col], label=col, ax=ax)

    ax.legend()
    my_plot.show(save_path=save_path)

    return result_df


# ===================================================================
# [2단원] 평활 비교
# ===================================================================
def compare_smoothing(data, sizes, column=None, method="ma", overlay=False,
                      plot=True, title=None, xlabel=None, ylabel=None,
                      width=960, height=400, save_path=None):
    """창 크기(또는 span)를 바꿔가며 평활 결과를 비교한다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        sizes (list | tuple): 비교할 창 크기(또는 span) 목록.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        method (str): "ma"(이동평균) · "ewm"(지수평활) · "both"(둘 다) (기본값: "ma").
        overlay (bool): True면 한 축에 겹쳐 그리고, False면 격자에 나눠 그린다 (기본값: False).
        plot (bool): 그래프를 그릴지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None).
        ylabel (str): y축 레이블 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 960).
        height (int): 캔버스 세로 픽셀 (기본값: 400).
        save_path (str): 이미지 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본과 평활 결과.
    """
    # --- 1) 창 크기별 평활 계산 ---
    if column is not None:                  # 컬럼명이 전달되었다면
        data = data[column]                 # 대상 컬럼만 추출

    result_df = DataFrame({"원본": data})   # 결과를 담을 데이터프레임 준비

    for size in sizes:                      # 각 창 크기별로 평활 계산
        # --- 1-1) 이동평균 ---
        if method == "ma" or method == "both":
            result_df[f"이동평균({size})"] = data.rolling(size).mean()

        # --- 1-2) 지수평활 ---
        if method == "ewm" or method == "both":
            # ewm()의 첫 인자는 span이 아니라 com이므로 span을 키워드로 명시한다
            result_df[f"지수평활({size})"] = data.ewm(span=size).mean()

    # 그래프를 그리지 않으면 결과표만 반환하고 여기서 중단
    if not plot:
        return result_df

    # --- 2) 한 축에 겹쳐 그리기 ---
    if overlay:
        fig, ax = my_plot.init(width=width, height=height, title=title,
                               xlabel=xlabel, ylabel=ylabel)

        for col in result_df.columns:
            my_plot.lineplot(x=result_df.index, y=result_df[col], label=col, ax=ax)

        ax.legend()
        my_plot.show(save_path=save_path)

        # 겹쳐 그린 경우에도 결과표를 반환하고 여기서 중단
        return result_df

    # --- 3) 격자에 한 칸씩 그리기 ---
    cols = 2
    rows = int(np.ceil(len(result_df.columns) / cols))

    fig, ax = my_plot.init(rows=rows, cols=cols, width=width, height=height, title=title)

    for i, col in enumerate(result_df.columns):
        my_plot.lineplot(x=result_df.index, y=result_df[col], ax=ax[i])
        ax[i].set_title(col)

    # 컬럼 수가 홀수면 마지막 칸이 남으므로 숨긴다
    for i in range(len(result_df.columns), rows * cols):
        ax[i].set_visible(False)

    my_plot.show(save_path=save_path)

    return result_df


# ===================================================================
# [3단원] 시계열 분해
# ===================================================================
def decompose(data, period, column=None, model=None, plot=True, title=None,
              width=1280, height=250, save_path=None):
    """시계열을 추세 · 계절 · 잔차 세 성분으로 분해한다.

    분해는 차분하기 전의 원본으로 수행한다. 차분은 추세와 계절성을 지우는 작업인데
    분해는 그 추세와 계절성을 보려는 것이기 때문이다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        period (int): 계절 주기.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        model (str): "additive" 또는 "multiplicative". None이면 자동 판정 (기본값: None).
        plot (bool): 성분별 그래프를 그릴지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 한 단의 세로 픽셀 (기본값: 250).
        save_path (str): 이미지 저장 경로 (기본값: None).

    Returns:
        DataFrame: 원본·추세·계절·잔차 네 성분.
    """
    if column is not None:
        data = data[column]

    # --- 1) 결합 방식 자동 판정 ---
    # 주기 단위로 묶어, 표준편차보다 변동계수가 더 일정하면 승법으로 본다
    if model is None:
        block = np.arange(len(data)) // period
        block_std = data.groupby(block).std()
        block_mean = data.groupby(block).mean()
        block_cv = block_std / block_mean

        if (block_cv.max() / block_cv.min()) < (block_std.max() / block_std.min()):
            model = "multiplicative"
        else:
            model = "additive"

        print(f"결합 방식 자동 판정: {model}")
    else:
        print(f"결합 방식 지정: {model}")

    # --- 2) 분해 수행 ---
    result = seasonal_decompose(data, model=model, period=period)

    result_df = DataFrame({
        "원본": result.observed,
        "추세": result.trend,
        "계절": result.seasonal,
        "잔차": result.resid,
    })

    print(f"관측치 {len(result_df)}개 중 추세·잔차가 계산된 구간 {len(result_df.dropna())}개")

    if not plot:
        return result_df

    # --- 3) 성분별로 한 단씩 그리기 ---
    if title is None:
        title = f"시계열 분해 ({model}, period={period})"

    fig, ax = my_plot.init(rows=4, cols=1, width=width, height=height, title=title)

    for i, col in enumerate(result_df.columns):
        if col == "잔차":
            # 잔차는 불규칙하므로 점으로 그려야 남은 패턴이 보인다
            my_plot.scatterplot(data=None, x=result_df.index, y=result_df[col],
                                size=8, linewidth=0.5, ax=ax[i])
        else:
            my_plot.lineplot(x=result_df.index, y=result_df[col], ax=ax[i])

        ax[i].set_title(col)
        ax[i].set_ylabel("")

    # 잔차의 기준선은 승법이면 1, 가법이면 0이다
    if model == "multiplicative":
        base = 1
    else:
        base = 0

    ax[3].axhline(base, color="red", linestyle="--", linewidth=1)

    my_plot.show(save_path=save_path)

    return result_df


# ===================================================================
# [3단원] 계절지수 보고
# ===================================================================
def report_seasonal(data, period, column=None, model=None, plot=True,
                    title=None, xlabel=None, ylabel=None,
                    width=1000, height=400, save_path=None):
    """계절 성분을 주기 안의 위치별로 정리해 계절지수표를 만든다.

    승법 모델의 계절지수는 배수로 읽는다. 1.227은 추세선보다 22.7% 높다는 뜻이다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        period (int): 계절 주기.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        model (str): "additive" 또는 "multiplicative". None이면 자동 판정 (기본값: None).
        plot (bool): 계절지수 그래프를 그릴지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None).
        ylabel (str): y축 레이블 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 1000).
        height (int): 캔버스 세로 픽셀 (기본값: 400).
        save_path (str): 이미지 저장 경로 (기본값: None).

    Returns:
        DataFrame: 주기 내 위치(1~period)를 인덱스로 하는 계절지수표.
    """
    if column is not None:
        data = data[column]

    # --- 1) 결합 방식 자동 판정 ---
    if model is None:
        block = np.arange(len(data)) // period
        block_std = data.groupby(block).std()
        block_mean = data.groupby(block).mean()
        block_cv = block_std / block_mean

        if (block_cv.max() / block_cv.min()) < (block_std.max() / block_std.min()):
            model = "multiplicative"
        else:
            model = "additive"

    # --- 2) 분해 후 계절 성분만 사용 ---
    result = seasonal_decompose(data, model=model, period=period)
    season = result.seasonal

    # 계절 성분은 매 주기 똑같이 반복되므로 한 주기만 보면 전체를 알 수 있다.
    # 위치는 데이터 시작 시점을 1로 하여 주기 안에서 센다.
    position = np.arange(len(season)) % period + 1
    index = season.groupby(position).first()

    result_df = DataFrame({"계절지수": index.round(4)})
    result_df.index.name = "주기 내 위치"

    # --- 3) 기준 대비 크기 ---
    # 승법은 1이 기준이므로 백분율로, 가법은 0이 기준이므로 절대값으로 읽는다
    if model == "multiplicative":
        base = 1
        result_df["기준 대비(%)"] = ((index - 1) * 100).round(1)
    else:
        base = 0
        result_df["기준 대비"] = index.round(3)

    print(f"결합 방식: {model}")
    print(f"최고 위치: {index.idxmax()} ({index.max():.4f})")
    print(f"최저 위치: {index.idxmin()} ({index.min():.4f})")

    if plot:
        if title is None:
            title = f"주기 내 위치별 계절지수 ({model}, period={period})"

        fig, ax = my_plot.init(width=width, height=height, title=title,
                               xlabel=xlabel, ylabel=ylabel)

        my_plot.lineplot(x=index.index, y=index.values, marker="o", ax=ax)
        ax.axhline(base, color="red", linestyle="--", linewidth=1)

        my_plot.show(save_path=save_path)

    return result_df


# ===================================================================
# [3단원] 계절조정
# ===================================================================
def adjust_seasonal(data, period, column=None, model=None, plot=True, title=None,
                    xlabel=None, ylabel=None, width=1280, height=480, save_path=None):
    """원본에서 계절 성분을 걷어내 계절조정 시계열을 만든다.

    계절조정 결과에서 갑자기 튀는 지점은 계절성으로 설명되지 않는 사건이 있었다는 뜻이다.

    Args:
        data (DataFrame | Series): 날짜 인덱스를 가진 시계열 데이터.
        period (int): 계절 주기.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        model (str): "additive" 또는 "multiplicative". None이면 자동 판정 (기본값: None).
        plot (bool): 원본과 겹쳐 그릴지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None).
        ylabel (str): y축 레이블 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀 (기본값: 480).
        save_path (str): 이미지 저장 경로 (기본값: None).

    Returns:
        Series: 계절 요인이 제거된 시계열.
    """
    if column is not None:
        data = data[column]

    # --- 1) 결합 방식 자동 판정 ---
    if model is None:
        block = np.arange(len(data)) // period
        block_std = data.groupby(block).std()
        block_mean = data.groupby(block).mean()
        block_cv = block_std / block_mean

        if (block_cv.max() / block_cv.min()) < (block_std.max() / block_std.min()):
            model = "multiplicative"
        else:
            model = "additive"

    # --- 2) 계절 성분 제거 ---
    result = seasonal_decompose(data, model=model, period=period)

    # 승법이면 나누고, 가법이면 뺀다
    if model == "multiplicative":
        adjusted = result.observed / result.seasonal
        print(f"결합 방식: {model} (원본 ÷ 계절)")
    else:
        adjusted = result.observed - result.seasonal
        print(f"결합 방식: {model} (원본 − 계절)")

    if plot:
        if title is None:
            title = f"원본 vs 계절조정 ({model}, period={period})"

        fig, ax = my_plot.init(width=width, height=height, title=title,
                               xlabel=xlabel, ylabel=ylabel)

        my_plot.lineplot(x=data.index, y=data, label="원본", ax=ax)
        my_plot.lineplot(x=adjusted.index, y=adjusted, label="계절조정", ax=ax)

        ax.legend()
        my_plot.show(save_path=save_path)

    return adjusted


# ===================================================================
# [4단원] 지연 산점도
# ===================================================================
def plot_lag(data, lag=1, column=None, title=None, xlabel=None, ylabel=None,
             width=640, height=560, save_path=None, ax=None):
    """현재 값과 lag 시점 전의 값을 산점도로 그려 자기상관을 눈으로 확인한다.

    비정상 시계열의 상관계수는 추세 때문에 자동으로 높게 나오므로 의미가 없다.

    Args:
        data (DataFrame | Series): 시계열 데이터.
        lag (int): 몇 시점 전의 값과 비교할지 (기본값: 1).
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None).
        ylabel (str): y축 레이블 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 640).
        height (int): 캔버스 세로 픽셀 (기본값: 560).
        save_path (str): 이미지 저장 경로 (기본값: None).
        ax: 그래프를 그릴 Axes 객체. None이면 새로 생성 (기본값: None).

    Returns:
        float: 현재 값과 lag 시점 전 값의 상관계수.
    """
    if column is not None:
        data = data[column]

    # lag 시점 전의 값
    past = data.shift(lag)
    corr = data.corr(past)

    if title is None:
        title = f"지연 산점도 (lag={lag}, 상관계수={corr:.3f})"

    if xlabel is None:
        xlabel = f"{lag}시점 전 값"

    if ylabel is None:
        ylabel = "현재 값"

    if ax is None:
        my_plot.scatterplot(data=None, x=past, y=data, size=20, linewidth=0.5,
                            title=title, xlabel=xlabel, ylabel=ylabel,
                            width=width, height=height, save_path=save_path)
    else:
        my_plot.scatterplot(data=None, x=past, y=data, size=20, linewidth=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    return round(corr, 3)


# ===================================================================
# [4단원] 자기상관 계수 보고
# ===================================================================
def report_correlation(data, lags, column=None, kind="acf"):
    """ACF 또는 PACF 계수를 구하고 유의성과 절단 지점을 판정한다.

    반드시 정상 시계열에 적용한다. 비정상 시계열은 모든 시차가 유의하게 나와
    어디서 끊기는지 판단할 수 없다.

    Args:
        data (DataFrame | Series): 정상성을 만족하는 시계열 데이터.
        lags (int): 확인할 최대 시차. 계절 주기의 2~3배로 잡는다.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        kind (str): "acf"(자기상관) 또는 "pacf"(부분자기상관) (기본값: "acf").

    Returns:
        DataFrame: 시차를 인덱스로 하는 계수·절댓값·기준값·유의성·절단후보.
    """
    if column is not None:
        data = data[column]

    data = data.dropna()

    if kind == "acf":
        values = acf(data, nlags=lags)
    else:
        values = pacf(data, nlags=lags)

    # lag 0은 자기 자신(항상 1)이므로 판정 대상에서 제외한다
    values = values[1:]

    # 유의성 기준값. 관측치가 많을수록 작아진다.
    threshold = 2 / np.sqrt(len(data))

    result_df = DataFrame({
        "lag": range(1, lags + 1),
        kind: values.round(3),
        "절댓값": np.abs(values).round(3),
        "기준값": round(threshold, 3),
    })

    # 계수의 절댓값이 기준값을 넘으면 그 시차의 자기상관은 0이 아니다
    result_df["유의성"] = np.abs(values) > threshold

    # 절단후보: 직전 시차는 유의한데 현재 시차는 유의하지 않게 된 지점
    result_df["절단후보"] = result_df["유의성"].shift(1, fill_value=False) & ~result_df["유의성"]

    result_df = result_df.set_index("lag")

    significant = result_df[result_df["유의성"]].index.tolist()
    candidates = result_df[result_df["절단후보"]].index.tolist()

    print(f"[{kind}] 관측치 {len(data)}개 | 유의성 기준값 {threshold:.3f}")
    print(f"유의한 시차: {significant}")

    if len(candidates) > 0:
        print(f"첫 절단 지점: lag {candidates[0]} → 차수 후보 {candidates[0] - 1}")
    else:
        print("절단 지점 없음 → 차수 후보 0")

    return result_df


# ===================================================================
# [4단원] 자기상관 그래프
# ===================================================================
def plot_correlation(data, lags, column=None, p=None, q=None, title=None,
                     width=1280, height=380, save_path=None):
    """ACF와 PACF를 위아래로 그리고, 차수를 주면 판정 위치에 세로선을 표시한다.

    Args:
        data (DataFrame | Series): 정상성을 만족하는 시계열 데이터.
        lags (int): 확인할 최대 시차. 계절 주기의 2~3배로 잡는다.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        p (int): PACF에 표시할 AR 차수 (기본값: None).
        q (int): ACF에 표시할 MA 차수 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 한 단의 세로 픽셀 (기본값: 380).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    if column is not None:
        data = data[column]

    data = data.dropna()

    fig, ax = my_plot.init(rows=2, cols=1, width=width, height=height, title=title)

    # --- 1) ACF : MA 차수(q)의 근거 ---
    plot_acf(data, lags=lags, ax=ax[0])
    ax[0].set_xlabel("Lag")

    if q is None:
        ax[0].set_title("ACF (자기상관)")
    else:
        ax[0].axvline(q, color="red", linestyle="--", linewidth=1)
        ax[0].set_title(f"ACF — lag {q} 이후 절단 → q = {q}")

    # --- 2) PACF : AR 차수(p)의 근거 ---
    plot_pacf(data, lags=lags, ax=ax[1])
    ax[1].set_xlabel("Lag")

    if p is None:
        ax[1].set_title("PACF (부분자기상관)")
    else:
        ax[1].axvline(p, color="red", linestyle="--", linewidth=1)
        ax[1].set_title(f"PACF — lag {p} 이후 절단 → p = {p}")

    my_plot.show(save_path=save_path)


# ===================================================================
# [4단원] 차수 판정
# ===================================================================
def report_order(data, lags, column=None, period=None):
    """ACF와 PACF의 절단 지점에서 AR·MA 차수 후보와 계절 성분을 판정한다.

    ACF/PACF는 후보를 좁히는 도구이지 정답을 주는 도구가 아니다.
    최종 차수는 실제로 적합해 보고 AIC/BIC로 고른다.
    일반차분(d)과 계절차분(D) 횟수는 adf_diff · adf_transform이 담당한다.

    Args:
        data (DataFrame | Series): 정상성을 만족하는 시계열 데이터.
        lags (int): 확인할 최대 시차. 계절 주기의 2~3배로 잡는다.
        column (str): data가 데이터프레임인 경우 대상 컬럼명 (기본값: None).
        period (int): 계절 주기. None이면 계절 성분을 판정하지 않음 (기본값: None).

    Returns:
        dict: p · q · period · seasonal · seasonal_lags.
    """
    if column is not None:
        data = data[column]

    data = data.dropna()

    # lag 0은 자기 자신(항상 1)이므로 제외한다
    acf_values = acf(data, nlags=lags)[1:]
    pacf_values = pacf(data, nlags=lags)[1:]

    threshold = 2 / np.sqrt(len(data))

    # --- 1) ACF의 첫 절단 지점에서 MA 차수(q)를 읽는다 ---
    # 절단후보는 "직전 시차는 유의, 현재 시차는 비유의"인 지점이므로
    # 그 직전 시차까지가 유의한 구간이다. 그래서 lag - 1이 차수가 된다.
    q = 0

    for lag in range(2, lags + 1):
        previous = abs(acf_values[lag - 2])
        current = abs(acf_values[lag - 1])

        if previous > threshold and current <= threshold:
            q = lag - 1
            break

    # --- 2) PACF의 첫 절단 지점에서 AR 차수(p)를 읽는다 ---
    p = 0

    for lag in range(2, lags + 1):
        previous = abs(pacf_values[lag - 2])
        current = abs(pacf_values[lag - 1])

        if previous > threshold and current <= threshold:
            p = lag - 1
            break

    # --- 3) 주기의 배수 중 유의한 시차를 찾아 계절 성분을 판정한다 ---
    seasonal_lags = []

    if period is not None:
        for lag in range(period, lags + 1, period):
            if abs(acf_values[lag - 1]) > threshold:
                seasonal_lags.append(lag)

    order = {
        "p": p,
        "q": q,
        "period": period,
        "seasonal": len(seasonal_lags) > 0,
        "seasonal_lags": seasonal_lags,
    }

    print(f"AR 차수 후보 (p): {p}   ← PACF가 lag {p} 이후 절단")
    print(f"MA 차수 후보 (q): {q}   ← ACF가 lag {q} 이후 절단")

    if order["seasonal"]:
        print(f"계절 성분: 존재 (주기 {period}, 유의한 계절 시차 {seasonal_lags})")
    else:
        print("계절 성분: 없음")

    return order
