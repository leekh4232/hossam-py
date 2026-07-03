"""품질검사·EDA 반자동화 파이프라인 모듈

두 개의 파이프라인을 제공한다.

  1) qtcheck_pipeline : 원본 데이터를 받아 자료형 점검 → (체크박스로) 명목형
     변수 선택·변환 → (버튼으로) 중복 처리 결정 → 결측치 점검 → 기술통계 →
     인사이트(결론)까지 대화형으로 수행하는 품질검사 파이프라인.
  2) eda_pipeline     : 품질검사가 끝난 데이터와 종속변수 이름을 받아 시각화·
     추론통계·변수 선별표를 탭으로 종합하는 EDA 파이프라인.

원본 데이터프레임과 종속변수 이름만 전달하면, 필요한 기술통계량을
my_qtcheck 로 스스로 산출하고, my_stats 와 중복되지 않는 시각화를 일괄
수행한 뒤, 추론통계 검정을 단계적으로 실행하여 최종 '변수 선별표'까지
만들어내는 파이프라인을 제공한다.

산출물은 파이캐럿(PyCaret)처럼 처리 단계별 탭으로 구분되어 표시된다
(ipywidgets 미설치 시 순차 출력으로 자동 대체).

    기술통계 → 종속변수 시각화 → 독립변수 시각화
             → 연속×연속 / 연속×명목(2) / 연속×명목(3↑) / 명목×명목
             → 변수 선별표

각 탭은 하나의 공개 함수로 분리되어 있어, 노트북에서 개별적으로 호출하여
독립적으로 결과를 확인할 수 있다. eda_pipeline 은 이 개별 함수들을 호출해
결과가 있는 경우에만 탭으로 종합한다.

    show_descriptive   ── 기술통계
    viz_dependent      ── 종속변수 시각화
    viz_independent    ── 독립변수 시각화
    infer_cont_cont    ── 연속 × 연속       (상관분석)
    infer_cont_nom2    ── 연속 × 명목(2집단) (T검정)
    infer_cont_nom3    ── 연속 × 명목(3집단↑)(분산분석)
    infer_nom_nom      ── 명목 × 명목       (교차분석)
    selection_table    ── 변수 선별표
    eda_pipeline       ── 위 함수들을 탭으로 종합

검정 선택 규칙 (독립변수 유형 × 종속변수 유형)

    ┌───────────────┬───────────────────────┬────────────────────────┐
    │               │ 연속형 종속변수(예측) │ 범주형 종속변수(분류)  │
    ├───────────────┼───────────────────────┼────────────────────────┤
    │ 연속형 독립변수 │ 상관분석              │ 집단비교(T검정/ANOVA)  │
    │ 명목형 독립변수 │ 집단비교(T검정/ANOVA) │ 교차분석(카이제곱)     │
    └───────────────┴───────────────────────┴────────────────────────┘
"""

import os
import numpy as np
from itertools import combinations
from pandas import DataFrame
from IPython.display import display, Markdown

from . import my_qtcheck   # 기술통계량 산출
from . import my_plot      # 시각화
from . import my_stats     # 추론통계 검정
from . import my_prep      # long/wide 변환 등 전처리

# ipywidgets 가 있으면 탭 + (드롭다운으로 넘기는) 페이지로, 없으면 순차 출력으로 대체
try:
    from ipywidgets import (Tab as _Tab, Output as _Output, Stack as _Stack,
                            Dropdown as _Dropdown, VBox as _VBox, HBox as _HBox,
                            HTML as _HTML, Checkbox as _Checkbox, Button as _Button,
                            Label as _Label, jslink as _jslink)
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False


# 효과크기·강도 라벨 한글 변환
_STRENGTH_KR = {
    "Strong": "강함", "Moderate": "중간", "Weak": "약함", "None": "없음",
    "Large": "큼", "Medium": "중간", "Small": "작음", "Negligible": "미미",
}


def _kr(label):
    """효과크기 강도 라벨을 한글로 변환 (매핑에 없으면 원문 유지)"""
    return _STRENGTH_KR.get(label, label)


def _blank():
    """출력 사이에 빈 줄(여백)을 하나 넣는다."""
    display(Markdown("<br>"))


# 단계 사이 구분선 스타일 (여백 포함)
_DIVIDER_STYLE = "margin:20px 0; border:none; border-top:2px solid #d0d0d0;"


def _step_divider_widget():
    """대화형 단계 사이에 넣을 구분선(여백 포함) 위젯을 만든다."""
    return _HTML(f"<hr style='{_DIVIDER_STYLE}'>")


def _step_divider_md():
    """비대화형(순차 출력)에서 단계 사이에 구분선(여백)을 표시한다."""
    display(Markdown(f"<hr style='{_DIVIDER_STYLE}'>"))


def _save_df(df, path, index=True):
    """데이터프레임을 확장자에 맞춰 저장한다.

    지원 형식: 엑셀(.xlsx/.xls), CSV(.csv, utf-8 인코딩), Parquet(.parquet).
    path 가 비어 있으면 저장하지 않는다.
    """
    if not path:
        return
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=index)
    elif ext == ".csv":
        df.to_csv(path, index=index, encoding="utf-8")
    elif ext == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: '{ext}' "
            f"(지원: .xlsx/.xls, .csv, .parquet)")


# ===================================================================
# 내부 유틸리티
# ===================================================================
def _resolve_task(task):
    """모델링 유형 문자열을 분류 여부(bool)로 정규화"""
    return str(task).lower() in ("classification", "분류", "범주형", "clf")


def _classify(data, target):
    """my_qtcheck 로 변수 유형을 분류하여 반환한다.

    Returns:
        (number_cols, category_cols, feature_continuous, feature_nominal)
    """
    if target not in data.columns:
        raise KeyError(f"종속변수 '{target}' 가 데이터프레임에 없습니다.")
    number_cols = my_qtcheck.get_number_column_names(data)
    category_cols = my_qtcheck.get_categorical_column_names(data)
    feature_continuous = [c for c in number_cols if c != target]
    feature_nominal = [c for c in category_cols if c != target]
    return number_cols, category_cols, feature_continuous, feature_nominal


def _nlevels(data, col):
    """결측 제외 후 고유값(집단) 개수"""
    return int(data[col].dropna().nunique())


def _compute_vif(data, feature_continuous):
    """연속형 독립변수 간 다중공선성(VIF) 계산 (2개 미만이면 None)"""
    if len(feature_continuous) < 2:
        return None
    try:
        return my_stats.compute_vif(data[feature_continuous].dropna())
    except Exception:
        return None


def _fmt_p(p):
    """p-value 를 표기 규칙에 맞춰 문자열로 변환 (매우 작으면 <0.001)"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _asymmetry_signal(desc, var):
    """비대칭(왜도) 신호 문자열 생성

    numerical_summary(desc)의 skew_interpret / log_need 를 한글로 변환한다.
    명목형 등 desc 에 없는 변수는 '해당 없음'을 반환한다.
    """
    if desc is None or var not in desc.index:
        return "해당 없음"
    skew_map = {"right tail": "우편향", "left tail": "좌편향", "symmetric": "대칭"}
    log_map = {"log1p": "log1p 필요", "reverse_log1p": "reverse_log1p 필요", "none": "불필요"}
    skew = skew_map.get(desc.loc[var, "skew_interpret"], "대칭")
    log = log_map.get(desc.loc[var, "log_need"], "불필요")
    return f"{skew} / {log}"


def _collinearity_map(data, feature_continuous, alpha, threshold):
    """연속형 독립변수 간 상관분석 결과로부터 공선성 신호 맵을 만든다.

    (이 단계에서는 VIF를 사용하지 않으며, 오직 독립변수 쌍의 상관분석 결과만
    사용한다.) 각 변수에 대해 다른 연속형 독립변수와의 |상관계수|가 threshold
    이상인 가장 강한 상대를 (상대변수, 계수)로 기록한다. 없으면 None.

    Returns:
        dict: {변수명: (상대변수, 계수) 또는 None}
    """
    strongest = {v: None for v in feature_continuous}
    for a, b in combinations(feature_continuous, 2):
        try:
            res = my_stats.correlation(data, a, b, alpha=alpha, plot=False)
        except Exception:
            continue
        coef = float(res.iloc[0]["coef"])
        if abs(coef) >= threshold:
            for v, other in ((a, b), (b, a)):
                cur = strongest[v]
                if cur is None or abs(coef) > abs(cur[1]):
                    strongest[v] = (other, coef)
    return strongest


def _collinearity_signal(collin, var):
    """공선성 신호를 색상 아이콘(O/X)으로 생성 (상관분석 결과 기반).

    🅾️ = 공선성 신호 있음(다른 연속형 독립변수와 강한 상관), ❎ = 없음.
    """
    if not collin or var not in collin or collin[var] is None:
        return "❎"      # X (공선성 신호 없음)
    return "🅾️"          # O (공선성 신호 있음)


def _decide(family, significant, alpha, corr_threshold,
            coef=None, np2=None, cramers=None):
    """검정 결과로부터 채택여부(verdict)와 근거(reason)를 산출한다.

    - 유의하지 않으면 → ❌ 제외
    - 유의 + 효과 충분 → ✅ 채택
    - 유의 + 효과 약함 → 🟡 후보
    """
    if not significant:
        if family == "corr":
            return "❌ 제외", f"유의한 상관 없음 (p≥{alpha})"
        if family == "chi2":
            return "❌ 제외", f"유의한 연관 없음 (p≥{alpha})"
        return "❌ 제외", f"집단 간 차이 없음 (p≥{alpha})"

    if family == "corr":
        if abs(coef) >= corr_threshold:
            return "✅ 채택", f"유의미한 상관 (|coef|≥{corr_threshold})"
        return "🟡 후보", f"유의하나 관계 약함 (|coef|<{corr_threshold})"

    if family == "anova":
        if np2 is not None and np2 >= 0.06:      # Cohen: 중간 이상
            return "✅ 채택", "집단 간 차이 유의 + 효과크기 충분"
        return "🟡 후보", "차이는 유의하나 효과크기 약함"

    if family == "chi2":
        if cramers is not None and cramers >= 0.1:  # Cohen: 약함 이상
            return "✅ 채택", "유의한 연관 (V≥0.1)"
        return "🟡 후보", "유의하나 연관 약함 (V<0.1)"

    # 2집단 T검정: 효과크기 미제공 → 유의성으로 판단
    return "✅ 채택", "집단 간 차이 유의함"


# ===================================================================
# 저수준 검정 실행 (독립변수 1개 → (결과행 dict, 결과표 DataFrame))
# ===================================================================
def _run_correlation(data, feature, target, alpha, corr_threshold, plot, palette):
    """연속형 ↔ 연속형 : 상관분석"""
    res = my_stats.correlation(data, x=feature, y=target, alpha=alpha,
                               plot=plot, palette=palette,
                               title=f"{feature} ↔ {target}")
    row = res.iloc[0]
    coef = float(row["coef"])
    p = float(row["p-value"])
    sig = p < alpha
    sym = "r" if row["method"] == "Pearson" else "ρ"

    # 가정점검 문구 구성
    #  - '이상치 계수왜곡'은 이상치가 상관계수 값을 왜곡하는 '영향점'인지를 뜻한다
    #    (단순한 이상치 존재/치우침은 비대칭신호 컬럼의 왜도로 표시된다).
    #  - '고왜도'는 |왜도|>1 로, 이 경우 Pearson 대신 Spearman 이 선택되므로
    #    Spearman 채택 사유를 함께 드러내기 위해 표기한다.
    checks = []
    checks.append("정규성 충족" if (row["normality_x"] and row["normality_y"]) else "정규성 위배")
    checks.append("선형성 충족" if row["linearity"] else "선형성 위배")
    checks.append("이상치 계수왜곡 있음" if row["influential_outlier"] else "이상치 계수왜곡 없음")
    if row["high_skew"]:
        checks.append("고왜도(→Spearman)")

    verdict, reason = _decide("corr", sig, alpha, corr_threshold, coef=coef)
    return {
        "채택여부": verdict, "변수": feature, "검정방법": row["method"],
        "유의수준 (p)": _fmt_p(p), "효과크기": f"{sym}={coef:+.3f} ({_kr(row['strength'])})",
        "가정점검": " / ".join(checks),
        "_family": "corr", "_p": p, "_effect": abs(coef), "근거": reason,
    }, res


def _run_ttest(data, cat, cont, feature_name, alpha, corr_threshold, plot, palette, title=None):
    """범주(2집단) ↔ 연속 : 독립표본 T검정 (정규성/등분산에 따라 자동 분기)"""
    pair = data[[cat, cont]].dropna()
    wide = my_prep.long2wide(pair, hue=cat, values=cont)
    if wide.shape[1] < 2:
        return None, None
    g1, g2 = wide.columns[0], wide.columns[1]
    res = my_stats.test_independent(wide, g1, g2, alpha=alpha, plot=plot,
                                    palette=palette,
                                    title=title if title is not None else f"{feature_name} ↔ {cont}")
    two = res.xs("two-sided", level="alternative")
    test_name = two.index[0]
    p = float(two["p-value"].iloc[0])
    stat = float(two["statistic"].iloc[0])
    sig = p < alpha

    stat_label = "U" if "Mann-Whitney" in test_name else "t"
    assume = {
        "Mann-Whitney U test": "정규성 위배",
        "Welch t-test": "정규성 충족 / 등분산 위배",
        "Student t-test": "정규성·등분산 충족",
    }.get(test_name, "-")

    verdict, reason = _decide("ttest", sig, alpha, corr_threshold)
    return {
        "채택여부": verdict, "변수": feature_name, "검정방법": test_name,
        "유의수준 (p)": _fmt_p(p), "효과크기": f"미제공 ({stat_label}={stat:g})",
        "가정점검": assume,
        "_family": "ttest", "_p": p, "_effect": 0.0, "근거": reason,
    }, res


def _run_anova(data, cat, cont, feature_name, alpha, corr_threshold):
    """범주(3집단↑) ↔ 연속 : 일원분산분석 (등분산에 따라 ANOVA/Welch)"""
    aov = my_stats.anova_oneway(data, y=cont, between=cat, alpha=alpha)
    frow = aov[aov["Source"] == cat].iloc[0]
    pcol = "p-unc" if "p-unc" in aov.columns else "p_unc"
    p = float(frow[pcol])
    np2 = float(frow["np2"])
    sig = p < alpha
    test_name = "Welch ANOVA" if frow["test"] == "welch_anova" else "One-way ANOVA"
    assume = "등분산 위배" if frow["test"] == "welch_anova" else "등분산 충족"

    verdict, reason = _decide("anova", sig, alpha, corr_threshold, np2=np2)
    return {
        "채택여부": verdict, "변수": feature_name, "검정방법": test_name,
        "유의수준 (p)": _fmt_p(p), "효과크기": f"η²={np2:.3f} ({_kr(frow['effect_size'])})",
        "가정점검": assume,
        "_family": "anova", "_p": p, "_effect": np2, "근거": reason,
    }, aov


def _run_chi2(data, feature, target, alpha, corr_threshold, plot, palette):
    """명목 ↔ 범주 : 교차분석 (카이제곱/피셔)"""
    res = my_stats.chi2_independence(data, x=feature, y=target, alpha=alpha,
                                     plot=plot, palette=palette,
                                     title=f"{feature} ↔ {target}")
    row = res.iloc[0]
    p = float(row["p-value"])
    cramers = float(row["effect(V)"])
    sig = p < alpha
    assume = "기대빈도 충족" if row["assumption"] else "기대빈도 위배"

    verdict, reason = _decide("chi2", sig, alpha, corr_threshold, cramers=cramers)
    return {
        "채택여부": verdict, "변수": feature, "검정방법": row["test"],
        "유의수준 (p)": _fmt_p(p), "효과크기": f"V={cramers:.3f} ({_kr(row['strength'])})",
        "가정점검": assume,
        "_family": "chi2", "_p": p, "_effect": cramers, "근거": reason,
    }, res


# ===================================================================
# 탭 안의 페이지 구분 헬퍼 (드롭다운으로 넘기는 방식)
# -------------------------------------------------------------------
# "탭 안의 탭"은 ipywidgets 에서 Output 안에 컨테이너 위젯을 display 하면
# 렌더링되지 않는 문제가 있어, 대신 하나의 Stack 을 Dropdown 으로 넘기는
# 방식으로 페이지를 구분한다. 각 페이지의 내용은 Output 에 담기고(Stack 의
# 말단 자식), Dropdown↔Stack 은 client-side(jslink)로 연결된다. 이렇게
# 만들어진 위젯은 상위 Tab 의 자식으로 '직접' 넣어야 정상 렌더링된다.
# ===================================================================
def _pages_widget(sections):
    """(제목, 렌더함수) 목록을 '드롭다운으로 넘기는 페이지' 위젯으로 만든다.

    각 섹션의 렌더함수를 즉시 실행해 Output 에 담고, 항목이 2개 이상이면
    Dropdown + Stack 으로 묶어 반환한다. 항목이 하나면 Output 하나만,
    없으면 None 을 반환한다. (ipywidgets 가 있을 때만 사용)
    """
    if not sections:
        return None

    outs = []
    for _, fn in sections:
        out = _Output()
        with out:
            fn()
        outs.append(out)

    if len(outs) == 1:
        return outs[0]

    labels = [title for title, _ in sections]
    stack = _Stack(children=outs, selected_index=0)
    dropdown = _Dropdown(options=labels, index=0, description="항목:")
    _jslink((dropdown, "index"), (stack, "selected_index"))  # 클릭 시 즉시 전환

    # 드롭다운과 컨텐츠 사이에 여백 + 구분선을 두어 시각적으로 분리
    divider = _HTML("<hr style='margin:10px 0 16px 0; border:none; "
                    "border-top:1px solid #d0d0d0;'>")
    return _VBox([dropdown, divider, stack])


def _pages_display(sections):
    """페이지 섹션을 그 자리에서 표시한다 (단독 호출·순차 출력 폴백용)."""
    if not sections:
        return
    if _HAS_WIDGETS:
        widget = _pages_widget(sections)
        if widget is not None:
            display(widget)
    else:
        for title, fn in sections:
            display(Markdown(f"##### ▸ {title}"))
            fn()


def _emit_sections(items, work, verbose, as_widget=False):
    """검정 항목들을 실행해 결과행을 모으고, 표시 방식은 인자로 제어한다.

    Args:
        items (list): (페이지 제목, 인자) 튜플 리스트
        work (callable): work(arg, plot) → 결과행 dict 또는 None.
            plot=True 면 결과표·그래프를 직접 출력한다.
        verbose (bool): True 면 페이지로 렌더링, False 면 조용히 수집만.
        as_widget (bool): True 면 렌더링 대신 (rows, 위젯) 을 반환(파이프라인용).

    Returns:
        list[dict] 또는 (list[dict], widget): 수집된 결과행 (as_widget 이면 위젯 동반)
    """
    rows = []

    # 조용히 수집만 (변수 선별표 조립용)
    if not verbose:
        for _, arg in items:
            row = work(arg, plot=False)
            if row is not None:
                rows.append(row)
        return (rows, None) if as_widget else rows

    # 페이지별 렌더 함수 구성 (렌더 중 결과행을 함께 수집)
    sections = []
    for title, arg in items:
        def render(arg=arg):
            row = work(arg, plot=True)
            if row is not None:
                rows.append(row)
        sections.append((title, render))

    # 파이프라인용: 위젯을 만들어 반환(상위 Tab 자식으로 직접 사용)
    if as_widget:
        return rows, _pages_widget(sections)

    # 단독 호출용: 그 자리에서 표시
    _pages_display(sections)
    return rows


# ===================================================================
# 탭 ① 기술통계
# ===================================================================
def show_descriptive(data, target, task="regression", _as_widget=False):
    """기술통계 탭: 페이지(기본정보 · 연속형 기술통계량 · 명목형 기술통계량)로 구분해 출력한다.

    노트북에서 단독 호출 시 드롭다운으로 페이지를 넘길 수 있다.
    (다중공선성 VIF 는 '독립변수 상관분석' 탭으로 옮겨졌다.)
    _as_widget=True 면 표시 대신 위젯을 반환한다(파이프라인용).
    """
    is_clf = _resolve_task(task)
    number_cols, category_cols, feat_cont, feat_nom = _classify(data, target)

    # ── 기본정보: 종속변수·모델링 유형·변수 분류 ──
    def basic():
        display(Markdown(f"**종속변수** : `{target}`  |  **모델링 유형** : "
                         f"{'분류(범주형)' if is_clf else '예측(연속형)'}"))
        display(Markdown(f"- 연속형 독립변수 : {feat_cont}"))
        display(Markdown(f"- 명목형 독립변수 : {feat_nom}"))

    # ── 연속형 기술통계량 ──
    def numeric():
        display(my_qtcheck.numerical_summary(data, columns=number_cols))

    # ── 명목형 기술통계량 ──
    def category():
        display(my_qtcheck.categorical_summary(data, columns=category_cols, value_counts=False))

    sections = [("기본정보", basic)]
    if number_cols:
        sections.append(("연속형 기술통계량", numeric))
    if category_cols:
        sections.append(("명목형 기술통계량", category))

    if _as_widget:
        return _pages_widget(sections)
    _pages_display(sections)


# ===================================================================
# 단변량 시각화 아래에 표시할 변수별 기술통계량 표
# ===================================================================
def _var_stats(data, var, continuous):
    """단변량 그래프 아래에 해당 변수의 기술통계량을 표로 표시한다.

    - 연속형: numerical_summary 를 전치(변수 1개 → 세로 표)해서 표시
    - 명목형: 범주별 빈도·비율 표시
    """
    display(Markdown("**기술통계량**"))
    if continuous:
        display(my_qtcheck.numerical_summary(data, columns=[var]).T)
    else:
        vc = data[var].value_counts().sort_index()
        freq = vc.rename("빈도").to_frame()
        freq["비율(%)"] = (vc / vc.sum() * 100).round(2)
        display(freq)


# ===================================================================
# 탭 ② 종속변수 시각화 (my_stats 와 중복되지 않는 단변량 분포)
# ===================================================================
def viz_dependent(data, target, task="regression", palette=None, _as_widget=False):
    """종속변수 시각화 탭: 종속변수의 단변량 분포 + 기술통계량 표를 표시한다.

    - 연속형: 히스토그램 + 상자그림
    - 범주형: 빈도 막대그래프

    _as_widget=True 면 표시 대신 위젯을 반환한다(파이프라인용).
    """
    is_clf = _resolve_task(task)

    def render():
        display(Markdown(f"#### 종속변수 `{target}` 분포"))
        if is_clf:
            my_plot.countplot(data=data, x=target, palette=palette,
                              title=f"종속변수 {target} 분포", width=800, height=500)
        else:
            fig, ax = my_plot.init(rows=1, cols=2, title=f"종속변수 {target} 분포",
                                   width=800, height=500)
            my_plot.histplot(data=data, x=target, kde=True, ax=ax[0])
            my_plot.boxplot(data=data, x=target, ax=ax[1])
            my_plot.show()
        # 그래프 아래에 기술통계량 표 함께 표시
        _var_stats(data, target, continuous=not is_clf)

    if _as_widget:
        return _pages_widget([("종속변수 분포", render)])
    render()


# ===================================================================
# 탭 ③ 독립변수 시각화 (my_stats 와 중복되지 않는 단변량 분포)
# ===================================================================
def viz_independent(data, target, task="regression", palette=None, _as_widget=False):
    """독립변수 시각화 탭: 독립변수별 페이지로 단변량 분포를 그린다.

    - 연속형: 히스토그램 + 상자그림
    - 명목형: 빈도 막대그래프

    _as_widget=True 면 위젯(없으면 None)을 반환하고, 아니면 그 자리에서 표시하고
    표시할 내용이 있었는지 여부(bool)를 반환한다.
    """
    _, _, feat_cont, feat_nom = _classify(data, target)

    # 독립변수별로 페이지 구성 (연속형: 히스토그램+상자그림 / 명목형: 빈도)
    # 각 그래프 아래에 해당 변수의 기술통계량 표를 함께 표시한다.
    def _cont(f):
        def render():
            fig, ax = my_plot.init(rows=1, cols=2, title=f"독립변수 {f} 분포",
                                   width=800, height=500)
            my_plot.histplot(data=data, x=f, kde=True, ax=ax[0])
            my_plot.boxplot(data=data, x=f, ax=ax[1])
            my_plot.show()
            _var_stats(data, f, continuous=True)
        return render

    def _nom(n):
        def render():
            my_plot.countplot(data=data, x=n, palette=palette,
                              title=f"독립변수 {n} 분포", width=800, height=500)
            _var_stats(data, n, continuous=False)
        return render

    sections = [(f, _cont(f)) for f in feat_cont] + [(n, _nom(n)) for n in feat_nom]

    if _as_widget:
        return _pages_widget(sections)
    _pages_display(sections)
    return bool(sections)


# ===================================================================
# 탭 ④ 추론통계 - 연속 × 연속 (상관분석)
# ===================================================================
def infer_cont_cont(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None, _as_widget=False):
    """연속 × 연속 상관분석 탭 (변수 쌍별 페이지).

    연속형 종속변수(예측)일 때만 수행되며, 각 연속형 독립변수와 종속변수의
    상관분석을 실행한다. 대상이 없으면 빈 리스트를 반환한다.

    Args:
        verbose (bool): True 면 결과표·산점도를 표시, False 면 결과만 수집.
        _as_widget (bool): True 면 (rows, 위젯) 을 반환한다(파이프라인용).

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트
    """
    if _resolve_task(task):
        return ([], None) if _as_widget else []   # 연속형 종속변수 아님 → 대상 없음
    _, _, feat_cont, _ = _classify(data, target)

    def work(f, plot):
        row, res = _run_correlation(data, f, target, alpha, corr_threshold,
                                    plot=plot, palette=palette)
        if plot and row is not None:
            display(res)
        return row

    items = [(f"{f} ↔ {target}", f) for f in feat_cont]
    return _emit_sections(items, work, verbose, as_widget=_as_widget)


# ===================================================================
# 탭 ⑤ 추론통계 - 연속 × 명목(2집단) (T검정)
# ===================================================================
def infer_cont_nom2(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None, _as_widget=False):
    """연속 × 명목(2집단) 독립표본 T검정 탭 (변수 쌍별 페이지).

    - 예측: 2수준 명목형 독립변수 ↔ 연속형 종속변수
    - 분류: 연속형 독립변수 ↔ 2범주 종속변수

    _as_widget=True 면 (rows, 위젯) 을 반환한다(파이프라인용).

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트 (대상 없으면 빈 리스트)
    """
    is_clf = _resolve_task(task)
    _, _, feat_cont, feat_nom = _classify(data, target)

    # (cat, cont, 표시명) 쌍 구성
    if is_clf:
        if _nlevels(data, target) != 2:
            return ([], None) if _as_widget else []
        pairs = [(target, f, f) for f in feat_cont]
    else:
        pairs = [(n, target, n) for n in feat_nom if _nlevels(data, n) == 2]

    def work(pair, plot):
        cat, cont, name = pair
        row, res = _run_ttest(data, cat, cont, name, alpha, corr_threshold,
                              plot=plot, palette=palette, title=f"{name} ↔ {target}")
        if plot and row is not None:
            display(res)
        return row

    items = [(f"{name} ↔ {target}", (cat, cont, name)) for cat, cont, name in pairs]
    return _emit_sections(items, work, verbose, as_widget=_as_widget)


# ===================================================================
# 탭 ⑥ 추론통계 - 연속 × 명목(3집단↑) (분산분석)
# ===================================================================
def infer_cont_nom3(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None, _as_widget=False):
    """연속 × 명목(3집단↑) 일원분산분석 + 사후검정 탭 (변수 쌍별 페이지).

    - 예측: 3수준 이상 명목형 독립변수 ↔ 연속형 종속변수
    - 분류: 연속형 독립변수 ↔ 3범주 이상 종속변수

    _as_widget=True 면 (rows, 위젯) 을 반환한다(파이프라인용).

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트 (대상 없으면 빈 리스트)
    """
    is_clf = _resolve_task(task)
    _, _, feat_cont, feat_nom = _classify(data, target)

    if is_clf:
        if _nlevels(data, target) < 3:
            return ([], None) if _as_widget else []
        pairs = [(target, f, f) for f in feat_cont]
    else:
        pairs = [(n, target, n) for n in feat_nom if _nlevels(data, n) >= 3]

    def work(pair, plot):
        cat, cont, name = pair
        row, aov = _run_anova(data, cat, cont, name, alpha, corr_threshold)
        if plot and row is not None:
            display(aov)
            # 사후검정(그룹 쌍별 비교)을 시각화와 함께 표시
            ph = my_stats.posthoc_oneway(data, y=cont, between=cat, alpha=alpha,
                                         plot=True, palette=palette,
                                         title=f"{name} 사후검정")
            display(ph)
        return row

    items = [(f"{name} ↔ {target}", (cat, cont, name)) for cat, cont, name in pairs]
    return _emit_sections(items, work, verbose, as_widget=_as_widget)


# ===================================================================
# 탭 ⑦ 추론통계 - 명목 × 명목 (교차분석)
# ===================================================================
def infer_nom_nom(data, target, task="regression", alpha=0.05,
                  corr_threshold=0.3, verbose=True, palette=None, _as_widget=False):
    """명목 × 명목 교차분석(카이제곱/피셔) 탭 (변수 쌍별 페이지).

    범주형 종속변수(분류)일 때만 수행되며, 각 명목형 독립변수와 종속변수의
    독립성 검정을 실행한다. 대상이 없으면 빈 리스트를 반환한다.

    _as_widget=True 면 (rows, 위젯) 을 반환한다(파이프라인용).

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트
    """
    if not _resolve_task(task):
        return ([], None) if _as_widget else []   # 범주형 종속변수 아님 → 대상 없음
    _, _, _, feat_nom = _classify(data, target)

    def work(n, plot):
        row, res = _run_chi2(data, n, target, alpha, corr_threshold,
                             plot=plot, palette=palette)
        if plot and row is not None:
            display(res)
        return row

    items = [(f"{n} ↔ {target}", n) for n in feat_nom]
    return _emit_sections(items, work, verbose, as_widget=_as_widget)


# ===================================================================
# 탭 ⑧ 독립변수 간 상관분석 (공선성 신호의 근거)
# ===================================================================
def corr_independent(data, target, alpha=0.05, collinearity_threshold=0.7,
                     palette=None, _as_widget=False):
    """독립변수 간 상관분석 탭.

    연속형 독립변수들 사이의 상관행렬과 강한 상관 쌍(공선성 후보)을 산출한다.
    이는 변수 선별표의 공선성 신호(🅾️/❎)의 근거가 된다. 연속형 독립변수가
    2개 미만이면 대상이 없으므로 위젯 모드에서는 None 을 반환한다.

    _as_widget=True 면 위젯(없으면 None)을 반환하고, 아니면 그 자리에서 표시한다.
    """
    _, _, feat_cont, _ = _classify(data, target)
    if len(feat_cont) < 2:
        return None if _as_widget else None

    # ── 상관행렬 · 강한 상관 쌍 ──
    def corr_page():
        display(Markdown("#### 연속형 독립변수 간 상관행렬"))
        _blank()
        display(Markdown("색이 진할수록 강한 상관 → 서로 중복(공선성) 후보"))
        _blank()
        # multi_correlation: 스타일된 상관행렬을 표시하고 쌍별 결과표를 반환
        corr_df = my_stats.multi_correlation(data, columns=feat_cont,
                                             plot=False, palette=palette)
        _blank()
        # 강한 상관 쌍(공선성 신호로 이어지는 쌍)을 강조해 표시
        display(Markdown(f"**강한 상관 쌍 (|coef|≥{collinearity_threshold} → 공선성 🅾️)**"))
        _blank()
        strong = corr_df[corr_df["coef"].abs() >= collinearity_threshold]
        if len(strong):
            display(strong.sort_values("coef", key=abs, ascending=False))
        else:
            display(Markdown("> 강한 상관 쌍 없음 (모든 독립변수 공선성 ❎)"))

    # ── 다중공선성(VIF): 기술통계 탭에서 이동해 옴 (독립변수 간 관계로 묶음) ──
    def vif_page():
        display(Markdown("#### 다중공선성(VIF)"))
        _blank()
        vif = _compute_vif(data, feat_cont)
        if vif is not None:
            display(vif)
        else:
            display(Markdown("> VIF 를 계산할 수 없습니다."))

    sections = [("독립변수 상관", corr_page), ("다중공선성(VIF)", vif_page)]
    if _as_widget:
        return _pages_widget(sections)
    _pages_display(sections)


# ===================================================================
# 탭 ⑨ 변수 선별표
# ===================================================================
_SELECT_COLUMNS = ["채택여부", "변수", "검정방법", "유의수준 (p)", "효과크기",
                   "가정점검", "비대칭신호", "공선성신호", "근거"]


def _build_selection_frame(rows, desc, collin):
    """결과행 리스트에 비대칭·공선성 신호를 결합하고 정렬하여 표로 만든다.

    공선성 신호는 VIF가 아니라 연속형 독립변수 간 상관분석 결과(collin)로만 산출한다.
    """
    if not rows:
        return DataFrame(columns=_SELECT_COLUMNS)

    for row in rows:
        var = row["변수"]
        row["비대칭신호"] = _asymmetry_signal(desc, var)
        sig = _collinearity_signal(collin, var)
        row["공선성신호"] = sig
        # 공선성 신호(🅾️)가 있는 채택 변수는 바로 채택하지 않고 '후보'로 분류한다.
        # (공선성 그룹 중 대표변수를 이후 단계에서 선택해야 하므로)
        if sig == "🅾️" and row["채택여부"] == "✅ 채택":
            row["채택여부"] = "🟡 후보"
            row["근거"] = f"{row['근거']} · 공선성 존재 → 후보(대표변수 선택 필요)"

    table = DataFrame(rows)
    # 정렬: 채택(0)→후보(1)→제외(2), 그 안에서 효과크기↓ · p값↑
    rank = {"✅ 채택": 0, "🟡 후보": 1, "❌ 제외": 2}
    table["_rank"] = table["채택여부"].map(rank)
    table = table.sort_values(by=["_rank", "_effect", "_p"],
                              ascending=[True, False, True])
    return table[_SELECT_COLUMNS].reset_index(drop=True)


def _display_selection(table, alpha, corr_threshold, collinearity_threshold):
    """변수 선별표를 채택여부별 배경색과 함께 출력한다."""
    display(Markdown("### 📋 변수 선별표"))
    display(Markdown(f"기준 : 유의수준 α={alpha}, 상관 채택 |coef|≥{corr_threshold}, "
                     f"공선성 신호 |r|≥{collinearity_threshold} (상관분석 기반 · VIF 미적용)  ·  "
                     f"공선성 🅾️=있음 / ❎=없음  ·  "
                     f"분류 ✅ 채택 / 🟡 후보 / ❌ 제외 (공선성 있는 채택 변수는 후보로 분류)"))
    if not len(table):
        display(Markdown("> 검정 가능한 독립변수가 없습니다."))
        return

    def _highlight(r):
        color = {"✅ 채택": "#e8f5e9", "🟡 후보": "#fffde7",
                 "❌ 제외": "#ffebee"}.get(r["채택여부"], "")
        return [f"background-color: {color}"] * len(r)

    display(table.style.apply(_highlight, axis=1))


def selection_table(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, collinearity_threshold=0.7, palette=None):
    """변수 선별표 탭: 4종 추론통계를 조용히 수행하여 최종 선별표를 출력한다.

    노트북에서 단독 호출하면 개별 검정 과정 출력 없이 선별표만 확인할 수 있다.
    공선성 신호는 연속형 독립변수 간 상관분석 결과로만 산출한다(VIF 미적용).
    """
    number_cols, _, feat_cont, _ = _classify(data, target)
    desc = my_qtcheck.numerical_summary(data, columns=number_cols) if number_cols else None
    collin = _collinearity_map(data, feat_cont, alpha, collinearity_threshold)

    rows = []
    for fn in (infer_cont_cont, infer_cont_nom2, infer_cont_nom3, infer_nom_nom):
        rows.extend(fn(data, target, task=task, alpha=alpha,
                       corr_threshold=corr_threshold, verbose=False, palette=palette))

    table = _build_selection_frame(rows, desc, collin)
    _display_selection(table, alpha, corr_threshold, collinearity_threshold)


# ===================================================================
# 파이프라인 본체 (개별 탭 함수를 호출해 결과가 있는 것만 탭으로 종합)
# ===================================================================
# (함수, 예측(연속형 종속) 탭 제목, 분류(범주형 종속) 탭 제목)
# 탭 제목은 "독립변수 유형 × 종속" 형태이며, 같은 검정이라도 task 에 따라
# 독립변수 유형이 달라지므로(예: T검정은 예측에선 명목독립, 분류에선 연속독립)
# task 에 맞는 제목을 사용한다.
_INFER_DEFS = [
    (infer_cont_cont, "연속 × 종속", "연속 × 종속"),
    (infer_cont_nom2, "명목 × 종속(2집단)", "연속 × 종속(2집단)"),
    (infer_cont_nom3, "명목 × 종속(3집단↑)", "연속 × 종속(3집단↑)"),
    (infer_nom_nom, "명목 × 명목", "명목 × 종속"),
]


def eda_pipeline(data, target, task="regression", alpha=0.05,
                 corr_threshold=0.3, collinearity_threshold=0.7, palette=None):
    """추론통계 기반 EDA 를 단계별 탭으로 종합 실행하는 파이프라인.

    원본 데이터프레임과 종속변수 이름만 전달하면, 기술통계량은 my_qtcheck 로
    스스로 산출하고, 개별 탭 함수를 차례로 호출하여 결과가 있는 탭만 모아
    파이캐럿처럼 탭으로 표시한다. 별도의 반환값은 없다.

    구성 탭:
        기술통계 · 종속변수 시각화 · 독립변수 시각화 ·
        (연속×연속 / 연속×명목(2) / 연속×명목(3↑) / 명목×명목 중 수행된 것만) ·
        변수 선별표

    Args:
        data (DataFrame): 품질점검·타입변환이 완료된 원본 데이터프레임
            (명목형은 category 타입으로 지정되어 있어야 자동 분류된다.)
        target (str): 종속변수(목표변수) 컬럼명
        task (str): 모델링 유형. 'regression'/'예측'/'연속형'(기본) 또는
            'classification'/'분류'/'범주형'
        alpha (float): 유의수준 (기본값: 0.05)
        corr_threshold (float): 상관계수 채택 기준 |coef| (기본값: 0.3)
        collinearity_threshold (float): 공선성 신호 기준 |r| (연속형 독립변수 간
            상관분석 기반, 기본값: 0.7). 이 단계에서 VIF는 사용하지 않는다.
        palette (str or list): 시각화 색상 팔레트 (기본값: None)
    """
    # 변수 선별표에 필요한 기술통계량/공선성 신호는 한 번만 계산해 재사용
    # (공선성 신호는 VIF가 아니라 연속형 독립변수 간 상관분석 결과로만 산출)
    is_clf = _resolve_task(task)
    _, _, feat_cont, _ = _classify(data, target)
    number_cols = my_qtcheck.get_number_column_names(data)
    desc = my_qtcheck.numerical_summary(data, columns=number_cols) if number_cols else None
    collin = _collinearity_map(data, feat_cont, alpha, collinearity_threshold)

    # -----------------------------------------------------------------
    # 폴백: ipywidgets 가 없으면 탭 없이 순차 출력
    # -----------------------------------------------------------------
    if not _HAS_WIDGETS:
        display(Markdown("## 기술통계"))
        show_descriptive(data, target, task)
        display(Markdown("## 종속변수 시각화"))
        viz_dependent(data, target, task, palette=palette)
        display(Markdown("## 독립변수 시각화"))
        viz_independent(data, target, task, palette=palette)

        all_rows = []
        for fn, reg_title, clf_title in _INFER_DEFS:
            rows = fn(data, target, task=task, alpha=alpha,
                      corr_threshold=corr_threshold, verbose=True, palette=palette)
            if rows:
                display(Markdown(f"## {clf_title if is_clf else reg_title}"))
                all_rows.extend(rows)

        if len(feat_cont) >= 2:
            display(Markdown("## 독립변수 상관분석"))
            corr_independent(data, target, alpha=alpha,
                             collinearity_threshold=collinearity_threshold, palette=palette)

        display(Markdown("## 변수 선별표"))
        table = _build_selection_frame(all_rows, desc, collin)
        _display_selection(table, alpha, corr_threshold, collinearity_threshold)
        return

    # -----------------------------------------------------------------
    # 위젯 모드: 각 탭 위젯을 '직접' 구성해 상위 Tab 의 자식으로 넣는다.
    # (Output 안에서 컨테이너 위젯을 display 하는 중첩을 피해야 정상 렌더링됨)
    # -----------------------------------------------------------------
    tab_items = []   # (제목, 위젯)

    # 기술통계 · 종속변수 시각화 (항상 표시)
    tab_items.append(("기술통계", show_descriptive(data, target, task, _as_widget=True)))
    tab_items.append(("종속변수 시각화",
                      viz_dependent(data, target, task, palette=palette, _as_widget=True)))

    # 독립변수 시각화 (독립변수가 있을 때만)
    w = viz_independent(data, target, task, palette=palette, _as_widget=True)
    if w is not None:
        tab_items.append(("독립변수 시각화", w))

    # 추론통계 4종 (수행되어 결과가 있는 것만)
    all_rows = []
    for fn, reg_title, clf_title in _INFER_DEFS:
        rows, w = fn(data, target, task=task, alpha=alpha, corr_threshold=corr_threshold,
                     verbose=True, palette=palette, _as_widget=True)
        if rows:
            all_rows.extend(rows)
            tab_items.append((clf_title if is_clf else reg_title, w))

    # 독립변수 간 상관분석 (연속형 독립변수 2개 이상일 때만) — 공선성 신호의 근거
    w = corr_independent(data, target, alpha=alpha,
                         collinearity_threshold=collinearity_threshold,
                         palette=palette, _as_widget=True)
    if w is not None:
        tab_items.append(("독립변수 상관분석", w))

    # 변수 선별표 (항상 표시) — 표 하나뿐이라 Output 말단으로 담는다
    table = _build_selection_frame(all_rows, desc, collin)
    sel_out = _Output()
    with sel_out:
        _display_selection(table, alpha, corr_threshold, collinearity_threshold)
    tab_items.append(("변수 선별표", sel_out))

    # 상위 Tab 구성 후 한 번만 표시
    tab = _Tab(children=[w for _, w in tab_items])
    for i, (t, _) in enumerate(tab_items):
        tab.set_title(i, t)
    display(tab)


# ===================================================================
# ===================================================================
# 품질검사 파이프라인 (qtcheck_pipeline)
# -------------------------------------------------------------------
# 원본 데이터를 받아 자료형 점검 → (체크박스) 명목형 변수 선택·변환 →
# (버튼) 중복 처리 결정 → 결측치 점검 → 기술통계 → 인사이트(결론)까지
# 대화형으로 수행한다. 사용자 의사결정이 필요한 단계는 위젯으로 물어보고,
# 나머지는 자동 진행하며, 최종 결과는 탭(마지막이 인사이트 결론)으로 종합한다.
# ===================================================================
# ===================================================================
def _nonnumeric_columns(data):
    """수치형이 아닌 컬럼(명목형 변환 후보) 목록을 반환한다."""
    from pandas.api.types import is_numeric_dtype
    return [c for c in data.columns if not is_numeric_dtype(data[c])]


def _qtcheck_insight(df, dup_count, dup_removed):
    """품질검사 결과로부터 인사이트(결론) Q1~Q4 를 자동 생성해 표시한다.

    a.ipynb 의 '인사이트' 결론 형식을 따르며, 수치는 데이터로부터 계산한다.
    """
    number_cols = my_qtcheck.get_number_column_names(df)
    category_cols = my_qtcheck.get_categorical_column_names(df)
    n_num, n_cat = len(number_cols), len(category_cols)
    total = n_num + n_cat

    # 연속형 기술통계량은 Q3·Q4 에서 공용으로 사용하므로 한 번만 계산
    desc = my_qtcheck.numerical_summary(df, columns=number_cols) if number_cols else None
    skew_kr = {"right tail": "우편향", "left tail": "좌편향", "symmetric": "대칭"}

    display(Markdown("## 💡 인사이트 (결론)"))
    _blank()

    # Q1. 수치/범주 비율
    display(Markdown("### Q1. 수치/범주 비율은?"))
    if total:
        display(Markdown(
            f"- 전체 변수 **{total}개** — 수치형 **{n_num}개({n_num/total*100:.1f}%)**, "
            f"범주형 **{n_cat}개({n_cat/total*100:.1f}%)**"))
        display(Markdown(f"- 수치형: {number_cols}"))
        display(Markdown(f"- 범주형: {category_cols}"))
    _blank()

    # Q2. 결측·중복
    display(Markdown("### Q2. 결측·중복 있나?"))
    na = df.isna().sum()
    total_na = int(na.sum())
    if total_na == 0:
        na_txt = "- 결측치: **0건** (모든 컬럼 결측 없음)"
    else:
        detail = ", ".join(f"`{k}`={int(v)}" for k, v in na[na > 0].items())
        na_txt = f"- 결측치: 총 **{total_na}건** ({detail})"
    if dup_count == 0:
        dup_txt = "- 중복: **0건** (처리 불필요)"
    else:
        dup_txt = f"- 중복: **{dup_count}건** → " + ("**제거함**" if dup_removed else "**유지함**")
    display(Markdown(na_txt))
    display(Markdown(dup_txt))
    _blank()

    # Q3. 분포 형태(왜도) / 로그변환 후보
    display(Markdown("### Q3. 분포 형태(왜도)와 로그변환 후보는?"))
    if desc is not None:
        for c in number_cols:
            r = desc.loc[c]
            display(Markdown(
                f"- `{c}`: 평균 {r['mean']:,.1f} / 중앙값 {r['50%']:,.1f} / "
                f"왜도 {r['skew']:.3f}({skew_kr.get(r['skew_interpret'], '')}) → **{r['log_need']}**"))
        _blank()
        log_cands = list(desc.index[desc["log_need"] != "none"])
        display(Markdown(f"➡ **로그변환 우선 검토 대상: {log_cands if log_cands else '없음'}**"))
    _blank()

    # Q4. 이상치 경계
    display(Markdown("### Q4. 이상치 경계는?"))
    if desc is not None:
        for c in number_cols:
            r = desc.loc[c]
            display(Markdown(
                f"- `{c}`: 하한 **{r['lower_bound']:,.1f}** / 상한 **{r['upper_bound']:,.1f}** — "
                f"상한 초과 {int(r['upper_outliers'])}건({r['upper_outliers_ratio']*100:.2f}%), "
                f"하한 미만 {int(r['lower_outliers'])}건({r['lower_outliers_ratio']*100:.2f}%)"))


def _qtcheck_result_tabs(df, dup_count, dup_removed,
                         numeric_save_path=None, category_save_path=None):
    """결측치·기술통계·인사이트(결론)를 탭 위젯으로 구성해 '반환'한다.

    numeric_save_path / category_save_path 가 주어지면 각 기술통계량 표를
    해당 경로에 저장한다(index 유지).

    (Output 안에서 컨테이너 위젯을 display 하는 중첩을 피하기 위해 표시하지
    않고 Tab 위젯을 반환하며, 호출부에서 상위 컨테이너의 자식으로 직접 넣는다.)
    """
    number_cols = my_qtcheck.get_number_column_names(df)
    category_cols = my_qtcheck.get_categorical_column_names(df)
    tab_items = []

    # 결측치
    o_na = _Output()
    with o_na:
        display(Markdown("#### 결측치 점검"))
        _blank()
        display(my_qtcheck.check_missing_values(df))
    tab_items.append(("결측치", o_na))

    # 기술통계 (명목형 / 연속형 드롭다운 페이지) — 계산 후 필요 시 저장
    secs = []
    if category_cols:
        cat_desc = my_qtcheck.categorical_summary(df, columns=category_cols, value_counts=False)
        _save_df(cat_desc, category_save_path, index=True)

        def cat_page(cd=cat_desc, path=category_save_path):
            if path:
                display(Markdown(f"- 저장됨: `{path}`"))
                _blank()
            display(cd)
        secs.append(("명목형 기술통계량", cat_page))
    if number_cols:
        num_desc = my_qtcheck.numerical_summary(df, columns=number_cols)
        _save_df(num_desc, numeric_save_path, index=True)

        def num_page(nd=num_desc, path=numeric_save_path):
            if path:
                display(Markdown(f"- 저장됨: `{path}`"))
                _blank()
            display(nd.T)
        secs.append(("연속형 기술통계량", num_page))
    if secs:
        tab_items.append(("기술통계", _pages_widget(secs)))

    # 인사이트 (결론) — 마지막 탭
    o_ins = _Output()
    with o_ins:
        _qtcheck_insight(df, dup_count, dup_removed)
    tab_items.append(("💡 인사이트(결론)", o_ins))

    tab = _Tab(children=[w for _, w in tab_items])
    for i, (t, _) in enumerate(tab_items):
        tab.set_title(i, t)
    return tab


def _qtcheck_noninteractive(data, save_path=None,
                            numeric_save_path=None, category_save_path=None):
    """ipywidgets 가 없을 때의 비대화형 품질검사 (기본값으로 자동 수행)."""
    display(Markdown("## ① 자료형 확인"))
    data.info()
    _step_divider_md()

    selected = _nonnumeric_columns(data)
    display(Markdown(f"## ② 명목형 변수 변환 (기본 선택: {selected})"))
    df = my_qtcheck.set_type(data, as_category=selected)
    _step_divider_md()

    display(Markdown("## ③ 데이터 중복 점검"))
    dup_count = int(df.duplicated().sum())
    df = my_qtcheck.check_duplicates(df, drop=True)
    _save_df(df, save_path, index=False)   # 품질검사 데이터 저장
    if save_path:
        display(Markdown(f"- 품질검사 데이터 저장: `{save_path}`"))
    _step_divider_md()

    display(Markdown("## ④ 결측치 점검"))
    display(my_qtcheck.check_missing_values(df))
    _step_divider_md()

    display(Markdown("## ⑤ 기술 통계량"))
    cats = my_qtcheck.get_categorical_column_names(df)
    nums = my_qtcheck.get_number_column_names(df)
    if cats:
        cat_desc = my_qtcheck.categorical_summary(df, columns=cats, value_counts=False)
        _save_df(cat_desc, category_save_path, index=True)
        display(cat_desc)
    if nums:
        num_desc = my_qtcheck.numerical_summary(df, columns=nums)
        _save_df(num_desc, numeric_save_path, index=True)
        display(num_desc.T)
    _step_divider_md()

    _qtcheck_insight(df, dup_count, dup_removed=(dup_count > 0))


def qtcheck_pipeline(data, save_path=None,
                     numeric_save_path=None, category_save_path=None):
    """데이터 품질검사를 대화형으로 수행하는 파이프라인.

    자료형 점검 → (체크박스) 명목형 변수 선택·변환 → (버튼) 중복 처리 결정 →
    결측치 점검 → 기술통계 → 인사이트(결론) 순으로 진행한다. 사용자의
    의사결정이 필요한 단계(명목형 선택·중복 처리)는 위젯으로 물어보고,
    나머지는 자동 진행하며, 결측치·기술통계·인사이트는 탭(마지막이 인사이트
    결론)으로 종합한다. 별도의 반환값은 없다.

    Args:
        data (DataFrame): 품질검사 대상 원본 데이터프레임
        save_path (str): 품질검사 완료 데이터프레임을 저장할 파일 경로
            (지원: .xlsx/.xls, .csv[utf-8], .parquet / 기본값: None → 저장 안 함)
        numeric_save_path (str): 연속형 기술통계량 표를 저장할 파일 경로
        category_save_path (str): 명목형 기술통계량 표를 저장할 파일 경로

    Note:
        ipywidgets 가 없으면 기본값(비수치형→범주형 변환, 중복 제거)으로
        비대화형 수행한다.
    """
    # 폴백: 비대화형 (기본값 사용)
    if not _HAS_WIDGETS:
        _qtcheck_noninteractive(data, save_path=save_path,
                                numeric_save_path=numeric_save_path,
                                category_save_path=category_save_path)
        return

    state = {"df": data.copy(), "dup_count": 0, "dup_removed": False}
    root = _VBox([])

    def append(*widgets):
        root.children = root.children + tuple(widgets)

    # ── ① 자료형 확인 ──
    o1 = _Output()
    with o1:
        display(Markdown("## ① 자료형 확인"))
        data.info()
    append(o1)

    # ── ② 명목형(범주형) 변수 선택 (체크박스) ──
    o2 = _Output()
    with o2:
        display(Markdown("## ② 명목형(범주형) 변수 선택"))
        display(Markdown("`category` 타입으로 변환할 컬럼을 체크하세요. "
                         "(비수치형 컬럼은 기본 선택됨)"))
    default_nom = set(_nonnumeric_columns(data))
    checks = [_Checkbox(value=(c in default_nom), description=c, indent=False)
              for c in data.columns]
    convert_btn = _Button(description="자료형 변환 ▶", button_style="primary")
    step2_out = _Output()
    append(_step_divider_widget(), o2, _VBox(checks), convert_btn, step2_out)

    # ── ③ 중복 점검 (버튼) ──
    def step3_duplicates():
        o3 = _Output()
        dup_count = int(state["df"].duplicated().sum())
        state["dup_count"] = dup_count
        with o3:
            display(Markdown("## ③ 데이터 중복 점검"))
            display(Markdown(f"중복 행: **{dup_count}건**"))
        append(_step_divider_widget(), o3)

        if dup_count > 0:
            with o3:
                display(Markdown("중복을 어떻게 처리할까요?"))
            drop_btn = _Button(description="🗑 중복 제거", button_style="warning")
            keep_btn = _Button(description="유지")
            dup_out = _Output()
            append(_HBox([drop_btn, keep_btn]), dup_out)

            def finish(removed):
                drop_btn.disabled = keep_btn.disabled = True
                state["dup_removed"] = removed
                with dup_out:
                    if removed:
                        state["df"] = my_qtcheck.check_duplicates(state["df"], drop=True)
                    else:
                        display(Markdown("중복을 **유지**합니다."))
                step_final()

            drop_btn.on_click(lambda _: finish(True))
            keep_btn.on_click(lambda _: finish(False))
        else:
            state["dup_removed"] = False
            step_final()

    # ── ④ 결측치·기술통계·인사이트(결론)를 탭으로 종합 ──
    def step_final():
        _save_df(state["df"], save_path, index=False)   # 품질검사 데이터 저장
        head = _Output()
        with head:
            display(Markdown("## ④ 품질검사 결과"))
            if save_path:
                display(Markdown(f"- 품질검사 데이터 저장: `{save_path}`"))
        # Tab 위젯을 Output 에 담지 않고 root 의 자식으로 '직접' 넣어야 렌더링됨
        tab = _qtcheck_result_tabs(state["df"], state["dup_count"], state["dup_removed"],
                                   numeric_save_path=numeric_save_path,
                                   category_save_path=category_save_path)
        append(_step_divider_widget(), head, tab)

    def on_convert(_):
        convert_btn.disabled = True
        for cb in checks:
            cb.disabled = True
        selected = [cb.description for cb in checks if cb.value]
        with step2_out:
            display(Markdown(f"**선택된 명목형 변수:** {selected}"))
            state["df"] = my_qtcheck.set_type(data, as_category=selected)
        step3_duplicates()

    convert_btn.on_click(on_convert)

    display(root)
