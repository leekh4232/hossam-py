import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame
from statsmodels.api import add_constant, Logit
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

from . import my_plot
from . import my_stats


def fit_model(data, y, summary=False):
    """statsmodels의 Logit을 이용해 이항 로지스틱 회귀 모델을 적합한다.

    종속변수 `y`를 제외한 나머지 모든 컬럼을 독립변수로 사용하며,
    절편(상수항)을 자동으로 추가한 뒤 최대우도추정(MLE)으로 회귀계수를 추정한다.
    종속변수는 0/1의 두 값만 가지는 이분형이어야 한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명. `data`에 반드시 존재해야 하며 0/1의 이분형이어야 한다.
        summary: True로 설정하면 적합된 모델의 요약 통계량을 출력한다.
                  Defaults to False.

    Returns:
        적합이 완료된 로지스틱 회귀분석 결과 객체.
    """
    x = data.drop(columns=[y])          # 독립변수 데이터프레임 생성
    y_series = data[y]                  # 종속변수 시리즈 생성
    x_input = add_constant(x)           # 독립변수에 절편(상수항) 추가
    model = Logit(y_series, x_input)    # Logit 모델 객체 생성
    fit = model.fit(disp=0)             # 모델 적합. disp=0 -> 수렴 메시지 출력 안함

    if summary:
        print(fit.summary())            # 적합된 모델의 요약 통계량 출력 여부 확인

    return fit                          # 적합된 모델 객체(분석 결과) 반환


def predict(fit, new_data, threshold=0.5):
    """적합된 로지스틱 모델로 새로운 데이터의 예측 확률과 예측 범주를 계산한다.

    로지스틱 회귀의 예측값은 `1`(사건 발생)일 확률이므로, 임계값(threshold)을
    초과하면 1, 그렇지 않으면 0으로 분류한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        new_data: 예측에 사용할 새로운 데이터프레임. 독립변수 컬럼만 포함해야 한다.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).

    Returns:
        DataFrame: 예측 확률('proba')과 예측값('pred')을 담은 데이터프레임.
    """
    # 새로운 데이터에 절편(상수항) 추가
    new_data_with_const = add_constant(new_data)

    # 사건 발생(=1) 확률 예측
    proba = fit.predict(new_data_with_const)

    # 예측 확률과 임계값 기준 예측 범주를 DataFrame으로 반환
    return DataFrame({
        "proba": proba,                             # 1이 될 확률
        "pred": (proba > threshold).astype(int),    # 예측값
    })


def plot_sigmoid(fit, data, x, threshold=0.5, palette=None, title=None,
                 xlabel=None, ylabel=None, width=1280, height=640, save_path=None):
    """독립변수에 따른 사건 발생 확률의 S자 곡선(시그모이드)을 그린다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. (독립변수와 종속변수를 모두 포함)
        x (str): 곡선의 x축으로 사용할 독립변수명.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).
        palette (str): 그래프 색상에 사용할 팔레트 이름. (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None → 독립변수명).
        ylabel (str): y축 레이블 (기본값: None → "P(종속변수=1)").
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 그릴 종속변수 결정 ---
    yname = fit.model.endog_names

    # --- 2) 곡선을 그릴 x값 격자 생성 ---
    # 관측된 x의 최솟값~최댓값을 200등분해 촘촘한 곡선을 만든다
    grid = np.linspace(data[x].min(), data[x].max(), 200)

    # 곡선 계산용 입력 데이터
    curve_input = DataFrame({x: grid})

    # 예측에 사용할 수 있도록 상수항을 추가한 뒤 사건 발생(=1) 확률 계산
    proba = fit.predict(add_constant(curve_input))

    # --- 3) 그래프 초기화 ---
    line_color = sb.color_palette(palette)[0] if palette else "#328CC1"

    fig, ax = my_plot.init(width=width, height=height, title=title,
                           xlabel=xlabel if xlabel else x,
                           ylabel=ylabel if ylabel else f"P({yname}=1)")

    # --- 4) 실제 관측치(0/1) 산점도 ---
    # 같은 높이(0 또는 1)에 점이 겹쳐 보이므로 반투명하게 처리한다
    my_plot.scatterplot(data=data, x=x, y=yname, color="#888888",
                        alpha=0.4, palette=None, ax=ax)

    # --- 5) 예측 확률의 S자 곡선 ---
    my_plot.lineplot(x=grid, y=proba, color=line_color, ax=ax)

    # --- 6) 임계값 가로선과 분류 경계 세로선 ---
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7)

    # 곡선이 임계값을 넘어서는 지점(= 분류 경계)을 찾아 세로선으로 표시한다.
    # 확률이 단조 증가/감소하므로 부호가 바뀌는 첫 지점을 경계로 본다.
    crossed = np.where(np.diff(np.sign(np.asarray(proba) - threshold)) != 0)[0]
    if crossed.size > 0:
        boundary = grid[crossed[0]]
        ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.7)
        ax.text(x=boundary, y=threshold, s=f" 분류 경계: {boundary:.2f}",
                color="red", va="bottom", ha="left")

    # 확률은 0~1 범위이므로 여백을 조금 두고 축을 고정한다
    ax.set_ylim(-0.1, 1.1)

    # --- 7) 그래프 표시 ---
    my_plot.show(save_path=save_path)


def report_fitness(fit, log_x=None, log1p_x=None, reflect_x=None):
    """적합된 로지스틱 모델의 모형 적합도(model fit) 보고 문장을 생성해 반환한다.

    summary() 결과표의 문자열을 파싱하지 않고, `fit` 객체가 이미 갖고 있는 속성에서
    지표를 직접 읽어와 문장을 구성한다.
    로그변환한 독립변수는 log(...)/log1p(...)/log1p(max-...)로 표기해 실제 적합한 모형을 그대로 드러낸다.

    ※ 로지스틱 회귀의 종속변수는 0/1 이분형이므로 변환 대상이 아니다.
      따라서 my_ols의 동명 함수와 달리 log_y·log1p_y·reflect_y 인수를 받지 않는다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        log_x (list): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_x (list): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        reflect_x (list): 반사 후 log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).

    Returns:
        str: 모형 적합도 보고 문장. `IPython.display.Markdown`으로 감싸 출력하면 좋다.
    """
    # --- 1) 변수 라벨 구성 (상수항 제외) ---
    # log_x, log1p_x, reflect_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []
    reflect_x = reflect_x or []

    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != "const"]

    # 변환이 적용된 변수는 문장에 log(...)/log1p(...)로 표기한다.
    # 반사 변환은 log1p 와 식이 다르므로(대소가 뒤집힌다) 라벨도 구분해 적는다.
    xlabels = []   # 독립변수별 표기 라벨
    for x in xnames:
        if x in reflect_x:  xlabels.append(f"log1p(max-{x})")
        elif x in log1p_x:  xlabels.append(f"log1p({x})")
        elif x in log_x:    xlabels.append(f"log({x})")
        else:               xlabels.append(x)

    xlabel = ", ".join(xlabels)

    # --- 2) 유의확률 구간 표기 변환 ---
    p = fit.llr_pvalue
    if p < 0.001:   alpha = "< 0.001"
    elif p < 0.01:  alpha = "< 0.01"
    elif p < 0.05:  alpha = "< 0.05"
    else:           alpha = "≥ 0.05"

    # 유의수준(0.05) 기준 모형의 통계적 유의성 판정
    result = "유의하였다" if p < 0.05 else "유의하지 않았다"

    # --- 3) 유사결정계수의 적합 수준 해석 ---
    # (fit.prsquared 관례: 0.2~0.4 매우 우수 / 0.1~0.2 양호 / 그 미만 다소 낮음)
    prsq = fit.prsquared
    if prsq >= 0.2:     fit_level = "매우 우수한"
    elif prsq >= 0.1:   fit_level = "양호한"
    else:               fit_level = "다소 낮은"

    # --- 4) 문장 템플릿 구성 ---
    template = (
        "**Note. n = {n}. "
        "LL = {llf}, LL-Null = {llnull}, "
        "LLR χ²({df_model}) = {llr}, p {alpha}, "
        "Pseudo R² = {prsq}**\n\n"
        "{Y}를 종속변수로, {X}(을)를 독립변수로 한 이항 로지스틱 회귀분석 결과, "
        "모형은 통계적으로 {result}.\n\n"
        "> LLR χ²({df_model}) = {llr}, p {alpha}, Pseudo R² = {prsq}.\n\n"
        "즉, 모형의 유사결정계수는 {prsq}로 {fit_level} 적합 수준을 보였다.\n\n"
        "> ※ Pseudo R²는 선형회귀의 R²처럼 '분산 설명 비율'로 해석하지 않는다. "
        "일반적으로 **0.2~0.4** 구간이면 매우 우수한 적합으로 본다."
    )

    # --- 5) 문장 템플릿 값 치환 ---
    report = template.format(
        n=int(fit.nobs),
        llf=round(fit.llf, 3),
        llnull=round(fit.llnull, 3),
        df_model=int(fit.df_model),
        llr=round(fit.llr, 3),
        alpha=alpha,
        prsq=round(prsq, 3),
        Y=yname,
        X=xlabel,
        result=result,
        fit_level=fit_level,
    )

    # --- 6) 결과 리턴 ---
    return report


def report_variables(fit, data):
    """적합된 로지스틱 모델의 독립변수별 회귀계수·오즈비 보고표를 데이터프레임으로 생성해 반환한다.

    오즈비와 그 95% 신뢰구간을 함께 제공한다.
    다중공선성 점검을 위한 VIF 계산에 원본 데이터가 필요하므로 `data`를 함께 받는다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야 한다.

    Returns:
        DataFrame: 독립변수별 보고표. 종속변수·독립변수·B·표준오차·z·유의확률·
            오즈비(OR)·OR 95% 신뢰구간·공차·VIF 컬럼을 가진다.
            |B|(=|log OR|) 내림차순으로 정렬되어 영향력이 큰 변수가 위로 온다.
    """
    # --- 1) 대상 변수 확인 및 VIF 계산 ---
    yname = fit.model.endog_names                # 종속변수 이름
    exog_names = list(fit.model.exog_names)      # 상수항(const)을 포함한 전체 변수 이름 순서
    xnames = [name for name in exog_names if name != "const"]  # 상수항 제외 독립변수

    # 독립변수 전체를 대상으로 VIF를 한 번에 계산 (상수항 제외한 결과가 반환된다)
    vif = my_stats.compute_vif(data, columns=xnames)

    # 통계량을 위치 인덱스로 접근하기 위해 배열로 변환
    params = np.asarray(fit.params)             # 비표준화 회귀계수(B)
    bse = np.asarray(fit.bse)                   # 계수 표준오차
    zvalues = np.asarray(fit.tvalues)           # z-통계량 (로지스틱은 t가 아니라 z)
    pvalues = np.asarray(fit.pvalues)           # 계수 유의확률
    conf = np.asarray(fit.conf_int())           # 계수의 95% 신뢰구간 [하한, 상한]

    # --- 2) 독립변수별 계수·오즈비 정리 ---
    variables = []   # 독립변수별 보고 내용을 저장할 빈 리스트
    for x in xnames:
        i = exog_names.index(x)                 # 상수항을 포함한 전체 순서에서의 위치
        b = float(params[i])                    # 비표준화 회귀계수(B)
        vif_value = vif.loc[x, "VIF"]           # 미리 계산해 둔 VIF 값 조회

        # 표준화 회귀계수
        # --> (βstd) = B × 독립변수 표준편차 = "독립변수 1 SD 변화당 log(오즈비) 변화".
        # 로지스틱은 종속변수가 0/1이라 (OLS와 달리) y로는 표준화하지 않고 독립변수만 표준화한다.
        beta_std = b * float(data[x].std(ddof=1))

        row = {
            "종속변수": yname,                          # 종속변수 이름
            "독립변수": x,                              # 독립변수 이름
            "B": b,                                    # 비표준화 회귀계수(B)
            "βstd": beta_std,                          # 표준화 회귀계수
            "표준오차": bse[i],                         # 계수 표준오차
            "z": zvalues[i],                           # z-통계량
            "유의확률": pvalues[i],                     # 계수 유의확률
            "오즈비(OR)": float(np.exp(b)),             # 오즈비 = exp(B)
            "OR 95% 하한": float(np.exp(conf[i, 0])),  # 오즈비 신뢰구간 하한
            "OR 95% 상한": float(np.exp(conf[i, 1])),  # 오즈비 신뢰구간 상한
            "공차": 1 / vif_value,                     # 공차(Tolerance = 1/VIF)
            "VIF": vif_value,                          # 분산팽창계수
        }
        variables.append(row)

    # --- 3) 독립변수별 보고표 생성 및 반환 ---
    vdf = DataFrame(variables)

    # |B|(=|log OR|)의 절대값으로 내림차순 정렬 후 리턴 (영향력이 큰 변수가 위로 오도록)
    vdf = vdf.sort_values("B", key=abs, ascending=False).reset_index(drop=True)
    return vdf


def report_variables_text(fit, data=None, alpha=0.05,
                          log_x=None, log1p_x=None, reflect_x=None):
    """독립변수별 오즈비 해석 문장을 markdown 불릿 리스트로 생성해 반환한다.

    각 독립변수에 대해 계수(B)·오즈비(OR)·z·유의확률을 문장으로 풀어 쓰고,
    오즈비를 백분율 변화로 환산하여 "오즈가 약 몇 % 증가/감소" 형태로 해석한다.
    `data`가 주어지면 이분형(더미) 변수와 연속형 변수를 구분해 해석 표현을 달리한다.

    로그변환한 변수는 '1 증가'가 원래 단위의 1이 아니므로(예: log(Age)가 1 증가 = 나이가 약 2.72배)
    "1% 증가할 때"를 기준으로 해석을 바꾼다. 이때의 오즈비는 exp(B) 가 아니라 **1.01^B** 이다.
    반사 변환(log1p(max-x))은 % 해석의 기준이 반사값 (1+max-x)이고 대소가 뒤집히므로,
    원 변수 기준의 방향을 문장 끝에 따로 덧붙인다.

    ※ 로지스틱 회귀의 종속변수는 0/1 이분형이므로 변환 대상이 아니다.
      따라서 my_ols의 동명 함수와 달리 log_y·log1p_y·reflect_y 인수를 받지 않는다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 원본 데이터프레임 (기본값: None). 주어지면 이분형/연속형을 구분해 표현한다.
        alpha (float): 유의성 판정에 사용할 유의수준 (기본값: 0.05).
        log_x (list): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_x (list): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        reflect_x (list): 반사 후 log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).

    Returns:
        str: 독립변수별 해석 문장 불릿 리스트. `IPython.display.Markdown`으로 감싸 출력하면 좋다.
    """
    # --- 1) 해석 대상 결정 (상수항 제외) ---
    # log_x, log1p_x, reflect_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []
    reflect_x = reflect_x or []

    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != "const"]

    # 독립변수의 변환 종류를 하나의 값으로 판별한다 (구체적인 변환부터 확인한다)
    def kind_of(name):
        if name in reflect_x:   return "reflect"
        if name in log1p_x:     return "log1p"
        if name in log_x:       return "log"
        return "none"

    # --- 2) 문장 템플릿 구성 (독립변수마다 반복 적용) ---
    line_template = (
        "- **{x}**의 회귀계수는 **B = {B}**, 오즈비는 **OR = {OR}**로 나타났으며, "
        "이는 **{y}**에 {sig} 요인임을 의미한다. "
        "(**z = {z}**, **{p}**) "
        "즉, {change} {y}가 1(사건 발생)이 될 오즈는 평균적으로 약 **{pct}% {direction}**하는 것으로 해석된다.{note}"
    )
    # 반사 변환이 끼면 위 문장은 반사값 기준이므로, 원 변수 기준의 방향을 짧게 덧붙인다
    note_template = " (원 변수 기준: **{x}가 클수록 오즈 {orig_direction}**)"
    opposite = {"증가": "감소", "감소": "증가"}   # 반사로 뒤집힌 방향을 되돌릴 때 쓴다

    # --- 3) 독립변수별 해석 문장 생성 ---
    lines = []   # 독립변수별 문장(불릿)을 저장할 빈 리스트
    for x in xnames:
        x_kind = kind_of(x)             # none / log / log1p / reflect
        B = fit.params[x]               # 비표준화 회귀계수(B)
        z = fit.tvalues[x]              # z-통계량
        p = fit.pvalues[x]              # 계수 유의확률
        OR = np.exp(B)                  # 오즈비 = exp(B)

        # 유의성 판정 (유의수준 기준)
        sig_word = "유의한" if p < alpha else "유의하지 않은"

        # p값 APA 표기 (앞자리 0 생략)
        if p < 0.001:   p_text = "p < .001"
        else:           p_text = f"p = {p:.3f}".replace("0.", ".")

        # 계수 부호로 증가/감소 방향 결정
        # (문장의 주어가 반사값이면 이 방향은 반사값 기준의 방향이다)
        direction = "증가" if B > 0 else "감소"

        # 변화 표현: 로그 계열은 % 해석의 기준이 되는 값이 무엇인지가 핵심이다.
        # 변환하지 않은 변수만 이분형(더미)/연속형을 구분한다.
        is_binary = x_kind == "none" and data is not None and data[x].nunique() <= 2
        if is_binary:   change = f"**{x}**에 해당하는 경우(기준 범주 대비)"
        else:
            change = {
                "reflect": f"**(1+max-{x})가 1% 증가**할 때",
                "log1p":   f"**(1+{x})가 1% 증가**할 때",
                "log":     f"**{x}가 1% 증가**할 때",
                "none":    f"**{x}**가 1 증가할 때",
            }[x_kind]

        # 오즈의 백분율 변화 환산
        # 변환 안 함: 1 증가당 오즈비 = exp(B) / 로그 계열: 1% 증가당 오즈비 = 1.01^B
        # 1% 증가는 변화폭이 작아 소수점 첫째 자리에서 0이 되어버리므로 자리수를 더 남긴다
        if x_kind == "none":    pct = round(abs((OR - 1) * 100), 1)
        else:                   pct = round(abs((1.01 ** B - 1) * 100), 3)

        # 반사 변환은 대소 관계를 뒤집으므로 원 변수 기준의 방향은 반대가 된다
        if x_kind != "reflect":     note = ""
        else:                       note = note_template.format(x=x, orig_direction=opposite[direction])

        # 하나의 독립변수 → 하나의 불릿 문장
        lines.append(line_template.format(
            x=x, B=round(B, 4), OR=round(OR, 4), y=yname, sig=sig_word,
            z=round(z, 2), p=p_text, change=change,
            pct=pct, direction=direction, note=note,
        ))

    # --- 4) 해석 주의 각주 첨부 ---
    report = "\n".join(lines)
    report += (
        "\n\n> ※ 오즈비(OR)가 1보다 크면 사건 발생 오즈가 증가, 1보다 작으면 감소함을 뜻한다. "
        "유의확률이 유의수준보다 큰(=유의하지 않은) 변수는 효과가 통계적으로 확인되지 않았으므로 "
        "오즈비 해석에 주의한다. 더미변수의 오즈비는 '기준(drop_first로 제외된) 범주' 대비 값이다."
    )

    # --- 5) 로그·반사 변환 사용 시 주의 각주 ---
    if log_x or log1p_x or reflect_x:
        report += (
            "\n\n> ※ **로그변환한 변수**는 '1 증가'가 원래 단위의 1이 아니라 "
            "**변수가 약 2.72배(e배)가 되는 것**을 뜻한다.      \n"
            "그래서 위 문장은 **1% 증가** 기준으로 적었고, 이때의 오즈비는 표의 OR(=exp(B))이 아니라 "
            "**1.01^B**다. (변수가 2배가 되면 오즈는 **2^B배**)      \n"
            "즉 **표의 OR은 로그값이 1 증가했을 때의 값**이므로 원 단위로 그대로 읽으면 안 된다."
        )

    if log1p_x:
        report += (
            "\n\n> ※ **log1p**(=ln(1+·))의 % 해석은 변수 자체가 아니라 **(1+변수)** 기준이며, "
            "값이 클 때만 위 근사가 성립한다.      \n(0·작은 값 구간에서는 원본처럼 동작해 부정확)      \n"
            "이 구간에서는 부호·유의성 중심으로 해석한다."
        )

    if reflect_x:
        report += (
            "\n\n> ※ **반사 후 log1p**(=ln(1+max-·))는 값의 대소가 뒤집힌 변환이다. "
            "위 %·증감은 **(1+max-변수)** 기준이고,      \n"
            "원 변수 기준 방향은 각 문장 끝 괄호에 적었다.      \n"
            "변수가 **최댓값에 가까운 구간**에서는 위 근사가 부정확하므로 부호·유의성 중심으로 읽는다."
        )

    return report


def plot_odds(fit, data, palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=None, save_path=None):
    """오즈비(Odds Ratio)를 가로 막대그래프로 시각화해 독립변수의 영향력을 보여준다.

    막대는 `report_variables`가 정렬해 둔 |B|(=|log OR|) 내림차순 그대로 위에서
    아래로 배치되며, 오즈비가 1보다 큰지(사건 발생 오즈 증가) 작은지(감소)에 따라
    색을 달리하고 막대 끝에 오즈비 값을 표기한다. OR=1(영향 없음) 위치에 기준선을 둔다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        palette (dict): 부호별 막대 색상. None이면 {'+': 파랑, '-': 빨강} (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None → "오즈비(Odds Ratio)").
        ylabel (str): y축 레이블 (기본값: None → "독립변수").
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀. None이면 독립변수 수 × 80으로 자동 계산 (기본값: None).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 시각화용 데이터 전처리 ---
    vdf = report_variables(fit, data)
    rdf = vdf[["독립변수", "오즈비(OR)"]].copy()
    # OR>1이면 증가(+), OR<1이면 감소(-)로 색상 구분
    rdf["부호"] = np.where(rdf["오즈비(OR)"] > 1, "+", "-")

    # 독립변수가 많을수록 막대가 촘촘해지므로, 변수 하나당 80px씩 세로 공간을 확보한다
    if height is None:
        height = len(rdf) * 80

    if height < 200:
        height = 200   # 최소 높이 200px

    # 부호별 기본 색상: 증가(+)은 파랑, 감소(-)은 빨강
    if palette is None:
        palette = {"+": "#0066ff", "-": "#ff3333"}

    # --- 2) 그래프 초기화 ---
    fig, ax = my_plot.init(width=width, height=height, title=title,
                           xlabel=xlabel if xlabel else "오즈비(Odds Ratio)",
                           ylabel=ylabel if ylabel else "독립변수")

    # --- 3) 가로 막대그래프 ---
    my_plot.barplot(rdf, x="오즈비(OR)", y="독립변수", hue="부호", palette=palette, ax=ax)

    # OR=1(영향 없음) 기준선
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.7)

    # --- 4) 막대 끝에 오즈비 값 표기 ---
    for i in rdf.index:
        orv = rdf.loc[i, "오즈비(OR)"]
        ax.text(x=orv, y=i, s=f"{orv:.2f}", va="center",
                ha="left" if orv >= 1 else "right", color="black")

    # --- 5) 그래프 표시 ---
    my_plot.show(save_path=save_path)
