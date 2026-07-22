"""품질검사·EDA 반자동화 파이프라인 모듈

두 개의 파이프라인을 제공한다.

  1) qtcheck : 원본 데이터를 받아 자료형 점검 → (체크박스로) 명목형 변수
     선택·변환 → (버튼으로) 중복 처리 결정 → 결측치 점검 → 기술통계 →
     인사이트(결론)까지 대화형으로 수행하는 품질검사 파이프라인.
  2) eda     : 품질검사가 끝난 데이터와 종속변수 이름을 받아, 노트북의 EDA
     분석 흐름(#01~#05)을 그대로 자동 수행하는 EDA 파이프라인.

eda 는 노트북에서 손으로 진행하던 분석 흐름과 장(章) 구분을 1:1로 따른다.
각 장이 하나의 탭이 되고, 장 안의 분석 단위(변수·변수쌍)가 드롭다운으로
넘기는 페이지가 된다 (ipywidgets 미설치 시 순차 출력으로 자동 대체).

    #01 준비작업      변수 유형 분류 · 연속형/명목형 기술통계량 · EDA 범위 정리
    #02 단변량 분석    종속변수 → 연속형 독립변수 → 명목형 독립변수
    #03 이변량 분석    상관분석 → T검정 → ANOVA → 교차분석
    #04 다변량 분석    독립변수 간 상관행렬·중복 집계 · 다중공선성(VIF)
    #05 최종 변수 선택 변수 선별표

노트북과 맞춘 점

  - 컬럼 의미 딕셔너리(column_means)를 받아 그래프·페이지 제목에
    `carat(중량)` 처럼 의미를 함께 표기한다.
  - 품질점검 단계에서 미리 산출해 둔 기술통계량(num_desc / cat_desc)을 그대로
    재사용한다 (전달하지 않으면 my_qtcheck 로 직접 산출).
  - 종속변수 유형과 실제 변수 구성으로부터 'EDA 범위 정리' 표를 자동 생성해,
    어떤 검정이 적용/미적용인지 근거와 함께 보여준다.
  - 상관분석은 산점도를 먼저 그린 뒤 결과표를 표시하고, ANOVA 사후검정은
    기본적으로 결과표만 표시한다(posthoc_plot=True 로 시각화 추가 가능).
  - 서열척도(범주 순서)는 전달받은 데이터의 category 순서를 그대로 사용한다.
  - 노트북에서 손으로 적던 '💡 인사이트' 표를 변수·변수쌍마다 자동 생성해
    그래프·결과표 아래에 함께 붙인다. '관찰값'(측정치)과 '판정'(라벨)을 나눠,
    어떤 수치를 보고 그렇게 판정했는지 되짚을 수 있게 한다.

      단변량 연속형  분포 모양 · 봉우리 수 · 이상치 · 이산점(군집) · 판단·조치
      단변량 명목형  범주 수 · 최빈 범주 · 최소 범주 · 균형 여부 · 판단·조치
      상관분석      선형/비선형 · 단조성 · 산포 · 이상치 · 채택 여부
      T검정         집단 차이 · 가정 점검 · 효과크기 · 채택 여부
      분산분석      집단 차이 · 효과크기 · 사후검정 구분력 · 채택 여부
      교차분석      연관성 · 효과크기 · 기대빈도 가정 · 채택 여부

    변수 선정은 이 흐름을 따른다. 단변량에서 각 변수의 전처리 방침(판단·조치)을
    정하고, 이변량에서 종속변수와의 관계로 1차 채택 여부를 가른 뒤, 다변량에서
    공선성 그룹을 찾아 그룹마다 효과크기 1위 하나만 대표로 남기고 나머지를
    후보로 돌린다. 최종 판정은 #05 변수 선별표에서 확정된다.

각 장은 하나의 공개 함수로 분리되어 있어, 노트북에서 개별적으로 호출하여
독립적으로 결과를 확인할 수 있다. eda 는 이 함수들의 페이지를 장 단위로 모아
탭으로 종합한다.

    show_descriptive   ── #01 준비작업 (변수 분류 · 기술통계량 · EDA 범위)
    viz_dependent      ── #02 종속변수 단변량
    viz_independent    ── #02 독립변수 단변량
    infer_cont_cont    ── #03 상관분석   (연속 × 연속)
    infer_cont_nom2    ── #03 T검정      (연속 × 명목(2집단))
    infer_cont_nom3    ── #03 ANOVA     (연속 × 명목(3집단↑))
    infer_nom_nom      ── #03 교차분석   (명목 × 명목)
    viz_cont_target_hue── #03 보조: 명목형을 색으로 얹은 산점도(상호작용 후보)
    corr_independent   ── #04 다변량 분석
    selection_table    ── #05 최종 변수 선택
    eda                ── 위 함수들을 장(章) 탭으로 종합

검정 선택 규칙 (독립변수 유형 × 종속변수 유형)

    ┌───────────────┬───────────────────────┬────────────────────────┐
    │               │ 연속형 종속변수(예측) │ 범주형 종속변수(분류)  │
    ├───────────────┼───────────────────────┼────────────────────────┤
    │ 연속형 독립변수 │ 상관분석              │ 집단비교(T검정/ANOVA)  │
    │ 명목형 독립변수 │ 집단비교(T검정/ANOVA) │ 교차분석(카이제곱)     │
    └───────────────┴───────────────────────┴────────────────────────┘
"""

import os
import base64
import html as _htmlmod
import numpy as np
from pandas import DataFrame
from IPython.display import display, Markdown
from IPython.utils.capture import capture_output
from statsmodels.stats.api import het_breuschpagan   # 등분산성(Breusch-Pagan) 검정

# 마크다운 → HTML 변환기 (정적 스냅샷 렌더링용).
# report_fitness 등이 만드는 마크다운(굵게·인용문·목록·표)을 HTML로 옮기려면 변환기가 필요하다.
# 어느 하나라도 설치돼 있으면 그것을 쓰고(markdown → mistune → markdown-it 순),
# 하나도 없을 때만 원문을 그대로 보여준다. mistune·markdown-it 은 nbconvert 가 함께 설치하므로
# 주피터 환경이라면 대개 셋 중 하나는 존재한다.
def _make_md_renderer():
    try:
        import markdown as _markdown
        return lambda text: _markdown.markdown(text, extensions=["tables"])
    except Exception:
        pass

    try:
        import mistune as _mistune
        _md = _mistune.create_markdown(plugins=["table"])
        return lambda text: _md(text)
    except Exception:
        pass

    try:
        from markdown_it import MarkdownIt as _MarkdownIt
        _mdit = _MarkdownIt("commonmark").enable("table")
        return lambda text: _mdit.render(text)
    except Exception:
        pass

    # 최후의 수단: 변환은 못 하더라도 줄바꿈만은 살려서 한 줄로 뭉치지 않게 한다.
    def _plain(text):
        return f'<div>{_htmlmod.escape(text).replace(chr(10), "<br>")}</div>'

    return _plain


_render_md = _make_md_renderer()

from . import my_qtcheck   # 기술통계량 산출
from . import my_plot      # 시각화
from . import my_stats     # 추론통계 검정
from . import my_prep      # long/wide 변환 등 전처리
from . import my_ols       # 회귀분석

# ipywidgets 가 있으면 탭 + (드롭다운으로 넘기는) 페이지로, 없으면 순차 출력으로 대체
try:
    from ipywidgets import (Tab as _Tab, Output as _Output, Stack as _Stack,
                            Dropdown as _Dropdown, VBox as _VBox, HBox as _HBox,
                            HTML as _HTML, Checkbox as _Checkbox, Button as _Button,
                            Label as _Label, IntProgress as _IntProgress,
                            jslink as _jslink)
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


# 출력 폰트 크기 (필요 시 이 값만 바꾸면 됨)
_CONTENT_FONT_SIZE = "16px"   # 표 + 일반 텍스트
_HEADING_FONT_SIZE = "24px"   # 제목(h1~h6)

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


def _capture_to_html(render_fn):
    """render_fn 의 출력(표·그래프·마크다운·텍스트)을 정적 HTML 로 캡처해
    HTML 위젯으로 반환한다.

    탭/페이지 내용을 Output 위젯 대신 값이 문자열 하나뿐인 HTML 위젯으로 담는다.
    Output 위젯은 캡처한 출력 목록을 프론트엔드 재렌더링(스크롤·comm 재동기화)
    때 중복 재생해 '이중출력'이 생기는데, HTML 위젯은 이 문제가 없다.
    """
    with capture_output() as cap:
        render_fn()

    parts = []
    for o in cap.outputs:
        d = getattr(o, "data", None) or {}
        if "text/html" in d:                         # 표(DataFrame/Styler) 등
            parts.append(d["text/html"])
        elif "image/svg+xml" in d:                   # SVG 그래프
            parts.append(d["image/svg+xml"])
        elif "image/png" in d:                       # PNG 그래프
            img = d["image/png"]
            if isinstance(img, bytes):
                img = base64.b64encode(img).decode("ascii")
            parts.append(f'<img style="max-width:100%;height:auto" '
                         f'src="data:image/png;base64,{img}">')
        elif "text/markdown" in d:                   # 마크다운 텍스트
            parts.append(_render_md(d["text/markdown"]))
        elif "text/plain" in d:                      # 그 외 텍스트
            parts.append(f'<pre style="white-space:pre-wrap;margin:0">'
                         f'{_htmlmod.escape(d["text/plain"])}</pre>')
    if cap.stdout:                                   # print/info 등 표준출력
        parts.append(f'<pre style="white-space:pre-wrap">'
                     f'{_htmlmod.escape(cap.stdout)}</pre>')

    # 표·일반 텍스트는 본문 크기, 제목(h1~h6)은 제목 크기로 지정
    body = "\n".join(parts)
    style = (
        "<style>"
        f".mp-content, .mp-content table {{ font-size: {_CONTENT_FONT_SIZE}; }}"
        f".mp-content h1, .mp-content h2, .mp-content h3,"
        f".mp-content h4, .mp-content h5, .mp-content h6"
        f" {{ font-size: {_HEADING_FONT_SIZE}; }}"
        "</style>"
    )
    html = f'{style}<div class="mp-content">{body}</div>'
    return _HTML(value=html)


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


def _compute_corr_pairs(data, feature_continuous, alpha):
    """연속형 독립변수들의 쌍별 상관분석 결과표를 한 번만 산출한다.

    이 결과표는 '#04 다변량 분석'의 상관행렬·중복 집계와 변수 선별표의
    공선성 신호에서 함께 사용되므로, 중복 계산을 피하기 위해 공유한다.
    (multi_correlation 은 호출 즉시 상관행렬을 출력하므로 여기서는 출력을 억제한다.)
    """
    if len(feature_continuous) < 2:
        return None
    try:
        with capture_output():
            return my_stats.multi_correlation(data, columns=feature_continuous,
                                              alpha=alpha, plot=False)
    except Exception:
        return None


def _corr_matrix(corr_pairs, columns):
    """쌍별 상관분석 결과표(corr_pairs)를 대칭 상관행렬로 재배치한다."""
    matrix = DataFrame(1.0, index=columns, columns=columns)
    for (a, b), coef in corr_pairs["coef"].items():
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] = coef
            matrix.loc[b, a] = coef
    return matrix


def _collinearity_map(corr_pairs, feature_continuous, threshold):
    """연속형 독립변수 간 상관분석 결과로부터 공선성 신호 맵을 만든다.

    (이 단계에서는 VIF를 사용하지 않으며, 오직 독립변수 쌍의 상관분석 결과만
    사용한다.) 각 변수마다 |상관계수|가 threshold 이상인 상대를 계수가 큰 순서로
    모두 기록해, 어떤 변수와 중복인지까지 근거로 남긴다.

    Returns:
        dict: {변수명: [(상대변수, 계수), ...]}  (중복 없으면 빈 리스트)
    """
    strong = {v: [] for v in feature_continuous}
    if corr_pairs is None:
        return strong

    for (a, b), coef in corr_pairs["coef"].items():
        if a not in strong or b not in strong:
            continue
        coef = float(coef)
        if abs(coef) >= threshold:
            strong[a].append((b, coef))
            strong[b].append((a, coef))

    # 가장 강하게 중복된 상대가 앞에 오도록 정렬
    for v in strong:
        strong[v].sort(key=lambda pair: abs(pair[1]), reverse=True)
    return strong


def _collinearity_groups(collin):
    """공선성 신호로 이어진 변수들을 하나의 그룹(연결 성분)으로 묶는다.

    `carat–x`, `x–y` 처럼 사슬로 이어진 변수들은 결국 같은 정보를 담고 있으므로
    하나의 그룹으로 본다. 그룹마다 대표변수 하나만 채택하면 된다.

    Returns:
        dict: {변수명: 그룹번호(1부터)}  (그룹에 속하지 않는 변수는 키가 없음)
    """
    # 공선성 신호가 있는 변수만 인접 목록으로 구성
    adjacent = {v: [other for other, _ in partners]
                for v, partners in collin.items() if partners}

    seen, groups = set(), []
    for start in adjacent:                       # dict 순서 = 변수 순서 → 결과가 항상 같다
        if start in seen:
            continue
        # 너비/깊이 우선 탐색으로 사슬로 이어진 변수를 모두 모은다
        stack, member = [start], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            member.append(cur)
            stack.extend(o for o in adjacent.get(cur, []) if o not in seen)
        groups.append(member)

    return {var: i + 1 for i, member in enumerate(groups) for var in member}


# 공선성 신호에 함께 표기할 중복 상대 변수의 최대 개수
_MAX_COLLIN_DETAIL = 3


def _collinearity_signal(collin, var, groups=None):
    """공선성 신호를 아이콘 + 그룹번호 + 중복 상대로 생성 (상관분석 결과 기반).

    🅾️ = 공선성 신호 있음(다른 연속형 독립변수와 강한 상관), ❎ = 없음.
    신호가 있으면 어느 그룹에 속하고 어떤 변수와 얼마나 중복인지
    `🅾️ G1 (x 0.996 / y 0.996)` 로 덧붙인다.
    """
    if not collin or not collin.get(var):
        return "❎"      # X (공선성 신호 없음)
    detail = " / ".join(f"{other} {coef:.3f}"
                        for other, coef in collin[var][:_MAX_COLLIN_DETAIL])
    gid = (groups or {}).get(var)
    return f"🅾️ G{gid} ({detail})" if gid else f"🅾️ ({detail})"


def _select_group_representative(rows, groups):
    """공선성 그룹마다 대표변수 하나만 채택으로 남기고 나머지는 후보로 내린다.

    같은 그룹의 변수들은 사실상 같은 정보이므로 하나만 모델에 넣으면 된다.
    대표는 **종속변수에 대한 효과크기가 가장 큰 변수**가 가져간다.
    (효과크기가 같으면 p값이 작은 변수 → 변수명 순으로 결정한다.)

    1·2위 격차(Δ)가 아무리 작아도 순위대로 대표를 정하되, 얼마나 근소한 차이로
    갈렸는지 판단할 수 있도록 Δ를 근거에 함께 남긴다. Δ가 매우 작다면 통계적으로는
    동률이므로, 최종 선택은 도메인 지식으로 다시 확인하는 것이 좋다.
    """
    # 그룹별 구성원 모으기 (이미 제외된 변수는 대표 후보가 될 수 없다)
    member_of = {}
    for row in rows:
        gid = groups.get(row["변수"])
        if gid is not None and row["채택여부"] != "❌ 제외":
            member_of.setdefault(gid, []).append(row)

    for gid, members in member_of.items():
        # 효과크기 내림차순 → p값 오름차순 → 변수명 순 (항상 같은 결과가 나오도록)
        members.sort(key=lambda r: (-r["_effect"], r["_p"], r["변수"]))
        names = ", ".join(m["변수"] for m in members)

        rep, rest = members[0], members[1:]
        if not rest:                       # 그룹에 살아남은 변수가 하나뿐이면 그대로 채택
            continue

        # 대표는 자신이 받은 판정을 그대로 유지한다. 1위라고 해서 효과크기가
        # 기준에 못 미치는 변수를 채택으로 올리지는 않는다.
        gap = rep["_effect"] - rest[0]["_effect"]
        verdict = "대표 채택" if rep["채택여부"] == "✅ 채택" else "그룹 대표"
        rep["근거"] = (f"{rep['근거']} · 공선성 그룹 G{gid}({names}) 중 효과크기 1위 "
                      f"→ {verdict} (2위 {rest[0]['변수']}와 Δ={gap:.4f})")

        for m in rest:
            m["채택여부"] = "🟡 후보"
            m["근거"] = (f"{m['근거']} · 공선성 그룹 G{gid} 대표를 {rep['변수']}에 양보 "
                        f"(Δ={rep['_effect'] - m['_effect']:.4f})")


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
def _run_correlation(data, feature, target, alpha, corr_threshold):
    """연속형 ↔ 연속형 : 상관분석

    시각화(산점도)는 노트북 흐름과 동일하게 호출부에서 별도로 그리므로
    여기서는 검정만 수행한다.
    """
    res = my_stats.correlation(data, x=feature, y=target, alpha=alpha, plot=False)
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


# 사후검정 상자그림 기본 높이(픽셀) — my_stats.posthoc_oneway 의 기본값과 동일
_POSTHOC_BASE_HEIGHT = 640
# 그룹이 3집단(3쌍)을 넘어설 때, '유의성 주석 막대 1개'당 더해줄 높이(픽셀)
_POSTHOC_HEIGHT_PER_PAIR = 30


def _posthoc_height(nlevels, base=_POSTHOC_BASE_HEIGHT):
    """사후검정 상자그림 높이를 집단 수에 따라 동적으로 키워 반환한다.

    사후검정 시각화는 그룹 쌍마다 유의성 주석 막대를 상자그림 '안쪽'에 세로로
    쌓는다. 막대 수는 집단 수 k 에 대해 k(k-1)/2 개로 늘어나므로, 집단이 많으면
    막대가 상자를 눌러 답답해진다. 기본 3집단(3쌍, base px)을 기준으로 그보다
    많은 쌍의 수만큼 높이를 늘려 주석이 들어갈 세로 공간을 확보한다.

    Args:
        nlevels (int): 명목형 변수의 집단(범주) 수
        base (int): 기준 높이(픽셀). 3집단일 때의 높이 (기본값: 640)

    Returns:
        int: 동적으로 계산한 그래프 높이(픽셀)
    """
    k = int(nlevels)
    n_pairs = k * (k - 1) // 2               # 세로로 쌓이는 유의성 막대 수
    extra_pairs = max(0, n_pairs - 3)        # 기본 3집단(3쌍) 초과분
    return base + extra_pairs * _POSTHOC_HEIGHT_PER_PAIR


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

    각 섹션의 렌더함수를 즉시 실행해 '정적 HTML 위젯'으로 담고, 항목이 2개
    이상이면 Dropdown + Stack 으로 묶어 반환한다. 항목이 하나면 HTML 위젯
    하나만, 없으면 None 을 반환한다. (ipywidgets 가 있을 때만 사용)

    Output 위젯 대신 HTML 위젯을 쓰는 이유는 프론트엔드 재렌더링 시 Output 이
    출력 목록을 중복 재생하는 이중출력 버그를 피하기 위함이다.
    """
    if not sections:
        return None

    outs = [_capture_to_html(fn) for _, fn in sections]

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


def _emit(sections, mode):
    """페이지 섹션 목록을 mode 에 맞는 형태로 내보낸다.

    모든 탭 함수가 공유하는 출력 규약이다.

    Args:
        sections (list): (페이지 제목, 렌더함수) 목록
        mode (str): "sections" → 목록 그대로 (장 단위로 합칠 때)
                    "widget"   → 페이지 위젯 (섹션이 없으면 None)
                    그 외       → 그 자리에서 표시

    Returns:
        mode 에 따라 섹션 목록 / 위젯 / 표시 여부(bool)
    """
    if mode == "sections":
        return sections
    if mode == "widget":
        return _pages_widget(sections)
    _pages_display(sections)
    return bool(sections)


def _emit_with_rows(rows, sections, mode):
    """검정 결과행을 함께 돌려주는 _emit (추론통계 탭 전용).

    결과행은 섹션이 렌더링될 때 채워지므로, 호출부는 위젯·섹션을 만든 뒤에
    rows 를 읽어야 한다.
    """
    if mode == "sections":
        return rows, sections
    if mode == "widget":
        return rows, _pages_widget(sections)
    _pages_display(sections)
    return rows


def _emit_sections(items, work, verbose, mode="display"):
    """검정 항목들을 실행해 결과행을 모으고, 표시 방식은 인자로 제어한다.

    Args:
        items (list): (페이지 제목, 인자) 튜플 리스트
        work (callable): work(arg, plot) → 결과행 dict 또는 None.
            plot=True 면 결과표·그래프를 직접 출력한다.
        verbose (bool): True 면 페이지로 렌더링, False 면 조용히 수집만.
        mode (str): 결과 형태
            - "display" : 그 자리에서 표시하고 결과행만 반환 (단독 호출용)
            - "widget"  : (결과행, 페이지 위젯) 반환
            - "sections": (결과행, [(제목, 렌더함수)]) 반환. 여러 검정의 페이지를
              하나의 탭으로 합칠 때 사용하며, 결과행은 렌더 시점에 채워진다.

    Returns:
        list[dict] 또는 (list[dict], widget|sections)
    """
    rows = []

    # 조용히 수집만 (변수 선별표 조립용) — 표시할 페이지가 없다
    if not verbose:
        for _, arg in items:
            row = work(arg, plot=False)
            if row is not None:
                rows.append(row)
        return _emit_with_rows(rows, [], mode)

    # 페이지별 렌더 함수 구성 (렌더 중 결과행을 함께 수집)
    sections = []
    for title, arg in items:
        def render(arg=arg):
            row = work(arg, plot=True)
            if row is not None:
                rows.append(row)
        sections.append((title, render))

    return _emit_with_rows(rows, sections, mode)


# ===================================================================
# 인사이트 판정표
# -------------------------------------------------------------------
# 노트북에서 사람이 손으로 적던 '💡 인사이트' 표를 데이터로부터 자동 생성한다.
# '관찰값'(측정치)과 '판정'(라벨)을 따로 두어, 어떤 수치를 보고 그렇게 판정했는지
# 학습자가 되짚을 수 있게 한다. 판정 기준은 아래 상수만 고치면 일괄로 바뀐다.
# ===================================================================
_INSIGHT_COLUMNS = ["점검 항목", "관찰값", "판정"]

# 이상치 비율 판정 경계 (미만 기준) — 초과 시 "많음"
_OUTLIER_BOUNDS = [(0.01, "적음"), (0.03, "소수"), (0.05, "다소 많음")]

# 범주 균형 판정 경계: 최빈 범주 ÷ 최소 범주 배수 (미만 기준) — 초과 시 "불균형"
_BALANCE_BOUNDS = [(2, "균형"), (5, "비교적 균형"), (15, "다소 불균형")]

# 사후검정 구분력 판정 경계: '실질 구분 쌍'의 비율 (미만 기준) — 초과 시 "높음"
#   실질 구분 쌍 = 유의하면서 효과크기가 Negligible 이 아닌 쌍.
#   표본이 크면 거의 모든 쌍이 유의해지므로(대표본 포화), 유의성만으로는 변별되지
#   않는다. 효과크기를 함께 봐야 '어느 등급쌍이 실제로 갈리는지'가 드러난다.
_POSTHOC_BOUNDS = [(0.3, "낮음"), (0.6, "보통")]

# 단조 관계가 이보다 약하면 산포(등분산) 판정은 해석 의미가 낮다고 본다
_WEAK_MONOTONE = 0.1

# 봉우리 탐지 설정
#   정수 격자처럼 값이 촘촘히 반복되는 변수는 KDE 대역폭이 지나치게 좁아져
#   격자마다 봉우리가 잡힌다. 대역폭을 넓혀 분포의 '큰 모양'만 남긴다.
_PEAK_BW_MULT = 2.0        # KDE 대역폭 배수
_PEAK_PROMINENCE = 0.05    # 봉우리로 인정할 최소 융기 (최대 밀도 대비 비율)
_PEAK_GRID = 512           # 밀도 곡선을 평가할 격자 수
_PEAK_MAX_SAMPLE = 20000   # 이보다 크면 표본추출 (결과 고정을 위해 시드 지정)

# 이산점(군집) 판정 기준
_INTEGER_RATIO = 0.9       # 정수 비율이 이 이상이면 '정수 격자형'
_SPARSE_RATIO = 0.01       # 고유값/행수가 이 이하이면 '값이 성김'


def _grade(value, bounds, over):
    """구간 목록으로 라벨을 정한다. bounds=[(상한, 라벨), ...] (미만 기준)"""
    for limit, label in bounds:
        if value < limit:
            return label
    return over


def _insight_frame(rows):
    """(점검 항목, 관찰값, 판정) 목록을 판정표로 만든다."""
    return DataFrame(rows, columns=_INSIGHT_COLUMNS)


def _show_insight(frame):
    """판정표를 표시한다 (결론 행인 '판단·조치'/'채택 여부'를 강조)."""
    display(Markdown("**💡 인사이트 (자동 판정)**"))

    def _mark(r):
        conclusion = str(r["점검 항목"]) in ("판단·조치", "채택 여부")
        return [f"background-color: {'#e3f2fd' if conclusion else ''}"] * len(r)

    display(frame.style.apply(_mark, axis=1).hide(axis="index"))


# -------------------------------------------------------------------
# 단변량 : 연속형
# -------------------------------------------------------------------
def _skew_label(skew, interpret):
    """왜도를 '강도 + 방향' 라벨로 변환한다 (|왜도| 0.5 / 1 / 2 를 경계로 사용)."""
    if abs(skew) < 0.5:
        return "거의 대칭"
    level = "중간" if abs(skew) < 1.0 else ("강한" if abs(skew) < 2.0 else "매우 강한")
    side = {"right tail": "우편향", "left tail": "좌편향"}.get(interpret, "대칭")
    return f"{level} {side}"


def _count_peaks(series):
    """분포 곡선(KDE)의 봉우리 수를 세어 (개수, 위치목록) 을 반환한다.

    산출할 수 없으면 (None, []) 을 반환한다.
    """
    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
    except Exception:
        return None, []

    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.ptp(values) == 0:
        return None, []
    if len(values) > _PEAK_MAX_SAMPLE:
        values = np.random.default_rng(0).choice(values, _PEAK_MAX_SAMPLE, replace=False)

    try:
        kde = gaussian_kde(values)
        kde.set_bandwidth(kde.factor * _PEAK_BW_MULT)
        grid = np.linspace(values.min(), values.max(), _PEAK_GRID)
        density = kde(grid)
        peaks, _ = find_peaks(density, prominence=_PEAK_PROMINENCE * density.max())
    except Exception:
        return None, []
    return len(peaks), [float(grid[i]) for i in peaks]


def _insight_continuous(data, var, desc=None):
    """연속형 변수의 단변량 판정표.

    노트북의 '분포 모양 · 봉우리 수 · 이상치 · 이산점(군집) · 판단·조치' 를 그대로 따른다.
    """
    stat = _desc_of(data, var, desc).loc[var]
    values = data[var].dropna()
    total = len(values)
    rows = []

    # ── 분포 모양: 왜도(치우침) + 첨도(뾰족함) ──
    skew = float(stat["skew"])
    kurt_label = {"leptokurtic": "뾰족함", "platykurtic": "완만함"}.get(
        stat["kurt_interpret"], "정규에 가까움")
    rows.append(("분포 모양",
                 f"왜도 {skew:+.3f} · 첨도 {float(stat['kurt']):+.3f}",
                 f"{_skew_label(skew, stat['skew_interpret'])} · {kurt_label}"))

    # ── 봉우리 수: 단봉이면 하나의 집단, 다봉이면 이질적인 집단이 섞여 있다는 신호 ──
    n_peaks, positions = _count_peaks(values)
    if n_peaks is None:
        rows.append(("봉우리 수", "산출 불가", "-"))
    else:
        shown = ", ".join(f"{p:,.2f}" for p in positions[:5])
        rows.append(("봉우리 수", f"{n_peaks}개" + (f" ({shown})" if shown else ""),
                     "단봉" if n_peaks <= 1 else "다봉"))

    # ── 이상치: IQR 울타리 기준 상·하단 건수와 비율 ──
    ratio = float(stat["outliers_ratio"])
    rows.append(("이상치",
                 f"상단 {int(stat['upper_outliers']):,}건 / "
                 f"하단 {int(stat['lower_outliers']):,}건 ({ratio:.1%})",
                 _grade(ratio, _OUTLIER_BOUNDS, "많음")))

    # ── 이산점(군집): 값이 특정 지점에 몰려 있는지 (정수 격자 · 성긴 고유값 · 0값) ──
    unique = int(values.nunique())
    integer_ratio = float((values == values.round()).mean()) if total else 0.0
    zeros = int((values == 0).sum())
    top_value, top_ratio = (None, 0.0)
    if total:
        counts = values.value_counts(normalize=True)
        top_value, top_ratio = counts.index[0], float(counts.iloc[0])

    observed = (f"고유값 {unique:,}종({unique / total:.1%}) · "
                f"최빈 {top_value:,.2f}({top_ratio:.1%}) · "
                f"정수 {integer_ratio:.0%} · 0값 {zeros:,}건")
    if zeros:
        cluster = f"0값 {zeros:,}건 — 확인 필요"
    elif integer_ratio >= _INTEGER_RATIO:
        cluster = "정수 격자형(사실상 이산)"
    elif unique / total <= _SPARSE_RATIO:
        cluster = "값이 성김(이산적)"
    else:
        cluster = "연속적"
    rows.append(("이산점(군집)", observed, cluster))

    # ── 판단·조치: 위 신호를 모아 전처리 방침을 제시 ──
    actions = []
    if zeros:
        actions.append(f"0값 {zeros:,}건 결측 처리 검토")
    log_need = str(stat["log_need"])
    if log_need == "log1p":
        actions.append("로그 변환(log1p) 검토")
    elif log_need == "reverse_log1p":
        actions.append("reverse_log1p 검토")
    if ratio >= _OUTLIER_BOUNDS[-1][0]:
        actions.append("이상치 처리 검토")
    rows.append(("판단·조치",
                 f"log_need={log_need} · 이상치 {ratio:.1%}",
                 " · ".join(actions) if actions else "변환 불필요 — 그대로 투입"))

    return _insight_frame(rows)


# -------------------------------------------------------------------
# 단변량 : 명목형
# -------------------------------------------------------------------
def _insight_nominal(data, var, is_target=False, is_clf=False):
    """명목형 변수의 단변량 판정표.

    노트북의 '범주 수 · 최빈 범주 · 최소 범주 · 균형 여부 · 판단·조치' 를 그대로 따른다.
    """
    counts = data[var].value_counts()
    counts = counts[counts > 0]              # 사용되지 않는 범주는 제외
    total = int(counts.sum())
    levels = int(len(counts))
    rows = []

    # ── 범주 수: 어떤 검정을 쓸지 결정하는 값 ──
    rows.append(("범주 수", f"{levels}개",
                 "2집단" if levels == 2 else ("3집단 이상" if levels >= 3 else "단일 범주")))

    # ── 최빈·최소 범주 ──
    top, bottom = counts.index[0], counts.index[-1]
    top_n, bottom_n = int(counts.iloc[0]), int(counts.iloc[-1])
    top_ratio, bottom_ratio = top_n / total, bottom_n / total
    rows.append(("최빈 범주", f"{top} {top_n:,}건 ({top_ratio:.1%})", "-"))
    rows.append(("최소 범주", f"{bottom} {bottom_n:,}건 ({bottom_ratio:.1%})",
                 "희소 범주" if bottom_ratio < 0.01 else "-"))

    # ── 균형 여부: 최빈 ÷ 최소 배수로 판정 ──
    fold = (top_n / bottom_n) if bottom_n else float("inf")
    rows.append(("균형 여부", f"최빈/최소 = {fold:.1f}배",
                 _grade(fold, _BALANCE_BOUNDS, "불균형")))

    # ── 판단·조치: 이 변수로 수행할 검정 + 희소 범주 처리 ──
    actions = []
    if is_target:
        actions.append("종속변수(분류)")
        if fold >= _BALANCE_BOUNDS[-1][0]:
            actions.append("클래스 불균형 → 가중치·리샘플링 검토")
    elif is_clf:
        actions.append("교차분석 대상")
    else:
        actions.append("T검정 대상" if levels == 2 else
                       ("ANOVA 대상" if levels >= 3 else "검정 불가(단일 범주)"))
    if bottom_ratio < 0.01:
        actions.append(f"희소 범주 '{bottom}' 병합 검토")
    rows.append(("판단·조치", f"범주 {levels}개 · 최소 {bottom_ratio:.1%}", " · ".join(actions)))

    return _insight_frame(rows)


# -------------------------------------------------------------------
# 이변량 : 공통 (채택 여부 행)
# -------------------------------------------------------------------
def _verdict_row(row):
    """검정 결과행에서 '채택 여부' 판정 행을 만든다.

    공선성 그룹의 대표변수 선정은 모든 검정이 끝난 뒤 #05 에서 확정되므로,
    여기서의 판정은 1차 판정임을 함께 표기한다.
    """
    return ("채택 여부", f"{row['근거']} (1차 판정)", row["채택여부"])


def _hetero_signal(data, x, y, alpha):
    """Breusch-Pagan 검정으로 잔차의 등분산 여부를 판정한다."""
    try:
        import statsmodels.api as sm
        pair = data[[x, y]].dropna()
        exog = sm.add_constant(pair[x].astype(float))
        resid = sm.OLS(pair[y].astype(float), exog).fit().resid
        p = float(het_breuschpagan(resid, exog)[1])
        return f"Breusch-Pagan p={_fmt_p(p)}", ("이분산" if p < alpha else "등분산")
    except Exception:
        return "산출 불가", "-"


# -------------------------------------------------------------------
# 이변량 : 상관분석
# -------------------------------------------------------------------
def _insight_correlation(data, x, y, res, row, alpha):
    """상관분석 판정표 (선형/비선형 · 단조성 · 산포 · 이상치 · 채택 여부)"""
    from scipy.stats import pearsonr, spearmanr

    result = res.iloc[0]
    pair = data[[x, y]].dropna()
    rows = []

    # ── 선형/비선형: 피어슨(직선)과 스피어만(단조)의 차이로 곡선 관계를 드러낸다 ──
    try:
        r = float(pearsonr(pair[x], pair[y])[0])
        rho = float(spearmanr(pair[x], pair[y])[0])
        observed = f"Pearson r={r:+.3f} / Spearman ρ={rho:+.3f}"
        if result["linearity"]:
            shape = "선형"
        else:
            shape = "비선형(단조)" if abs(rho) - abs(r) >= 0.02 else "비선형"
    except Exception:
        rho, observed, shape = float(result["coef"]), "산출 불가", "-"
    rows.append(("선형/비선형", f"{observed} · RESET 선형성 "
                 f"{'충족' if result['linearity'] else '위배'}", shape))

    # ── 단조성: 한 방향으로 꾸준히 오르내리는지 ──
    size = abs(rho)
    level = ("강한" if size > 0.7 else "중간" if size > 0.3 else
             "약한" if size > 0.1 else "거의 없는")
    direction = "양의 단조 증가" if rho > 0 else "음의 단조 감소"
    rows.append(("단조성", f"ρ={rho:+.3f}",
                 "단조성 없음" if level == "거의 없는" else f"{level} {direction}"))

    # ── 산포: 잔차의 퍼짐이 일정한지(등분산) ──
    # 관계 자체가 없는 변수는 등분산 여부를 따질 실익이 없으므로 그 사실을 함께 적는다.
    observed_scatter, verdict_scatter = _hetero_signal(data, x, y, alpha)
    if abs(rho) < _WEAK_MONOTONE and verdict_scatter != "-":
        verdict_scatter = f"{verdict_scatter} (관계 미약 — 해석 의미 낮음)"
    rows.append(("산포", observed_scatter, verdict_scatter))

    # ── 이상치: 상관계수를 왜곡하는 영향점인지 ──
    rows.append(("이상치", f"IQR 제외 시 r 변화 "
                 f"{'≥0.1' if result['influential_outlier'] else '<0.1'}",
                 "영향점 있음(계수 왜곡)" if result["influential_outlier"] else "영향점 없음"))

    rows.append(_verdict_row(row))
    return _insight_frame(rows)


# -------------------------------------------------------------------
# 이변량 : T검정
# -------------------------------------------------------------------
def _insight_ttest(res, row, alpha):
    """T검정 판정표 (집단 차이 · 가정 점검 · 효과크기 · 채택 여부)"""
    two = res.xs("two-sided", level="alternative")
    test_name = two.index[0]
    p = float(two["p-value"].iloc[0])

    rows = [
        ("집단 차이", f"{test_name} · p={_fmt_p(p)}",
         "유의" if p < alpha else "유의하지 않음"),
        ("가정 점검", row["가정점검"],
         "모수 검정" if "Mann-Whitney" not in test_name else "비모수 검정"),
        # 효과크기를 제공하지 않으므로 공선성 그룹의 대표 선정 근거로는 쓸 수 없다
        ("효과크기", "미제공", "유의성으로만 판정"),
        _verdict_row(row),
    ]
    return _insight_frame(rows)


# -------------------------------------------------------------------
# 이변량 : 분산분석
# -------------------------------------------------------------------
def _insight_anova(aov, posthoc, cat, row, alpha):
    """분산분석 판정표 (집단 차이 · 효과크기 · 사후검정 구분력 · 채택 여부)"""
    frow = aov[aov["Source"] == cat].iloc[0]
    pcol = "p-unc" if "p-unc" in aov.columns else "p_unc"
    p = float(frow[pcol])
    np2 = float(frow["np2"])

    rows = [
        ("집단 차이", f"{row['검정방법']} · p={_fmt_p(p)}",
         "유의" if p < alpha else "유의하지 않음"),
        ("효과크기", f"η²={np2:.3f}", _kr(frow["effect_size"])),
    ]

    # ── 사후검정 구분력: 몇 쌍이나 '실질적으로' 구분되는지 ──
    # 유의성만 보면 대표본에서 거의 모든 쌍이 유의해져 변별되지 않으므로,
    # 효과크기가 Negligible 이 아닌 쌍(=실질 구분 쌍)의 비율로 판정한다.
    if posthoc is not None and "significant" in posthoc.columns and len(posthoc):
        total = int(len(posthoc))
        hit = int(posthoc["significant"].sum())
        real = int((posthoc["significant"] &
                    (posthoc["effect_size"] != "Negligible")).sum()) \
            if "effect_size" in posthoc.columns else hit
        rows.append(("사후검정 구분력",
                     f"유의 쌍 {hit}/{total}({hit / total:.0%}) · "
                     f"실질 구분 쌍 {real}/{total}({real / total:.0%})",
                     _grade(real / total, _POSTHOC_BOUNDS, "높음")))
    else:
        rows.append(("사후검정 구분력", "산출 불가", "-"))

    rows.append(_verdict_row(row))
    return _insight_frame(rows)


# -------------------------------------------------------------------
# 이변량 : 교차분석
# -------------------------------------------------------------------
def _insight_chi2(res, row, alpha):
    """교차분석 판정표 (연관성 · 효과크기 · 기대빈도 가정 · 채택 여부)"""
    result = res.iloc[0]
    p = float(result["p-value"])

    rows = [
        ("연관성", f"{result['test']} · p={_fmt_p(p)}",
         "유의" if p < alpha else "유의하지 않음"),
        ("효과크기", f"Cramér's V={float(result['effect(V)']):.3f}",
         _kr(result["strength"])),
        ("기대빈도 가정", f"최소 기대빈도 {float(result['min_expected']):.1f}",
         "충족" if result["assumption"] else "위배(해석 주의)"),
        _verdict_row(row),
    ]
    return _insight_frame(rows)


# ===================================================================
# 노트북 흐름(#01~#05)에서 공통으로 쓰는 헬퍼
# ===================================================================
# 단변량·이변량 그래프 기본 크기 (노트북 실습 코드와 동일)
_FIG_WIDTH = 1280
_FIG_HEIGHT = 640

# 명목형 단변량 빈도 그래프 기본 팔레트 (노트북 실습 코드와 동일)
_NOMINAL_PALETTE = "Set2"

# hue 범례가 읽힐 수 있는 명목형 변수의 최대 범주 수 (초과 시 색 구분이 무의미)
_MAX_HUE_LEVELS = 10


def _label(col, column_means=None):
    """변수명에 컬럼의 의미를 덧붙인 표시용 이름을 만든다 (예: `carat(중량)`).

    노트북 #01-5 의 `column_means` 딕셔너리를 그대로 전달받아 사용하며,
    딕셔너리에 없는 변수는 변수명만 반환한다.
    """
    if column_means and col in column_means:
        return f"{col}({column_means[col]})"
    return col


def _num_desc(data, columns, precomputed=None):
    """연속형 기술통계량 표를 얻는다.

    노트북처럼 품질점검 단계에서 미리 산출해 둔 표를 전달하면 다시 계산하지 않고
    그대로 사용한다. 대상 컬럼이 없으면 None.
    """
    if precomputed is not None:
        return precomputed
    return my_qtcheck.numerical_summary(data, columns=columns) if columns else None


def _cat_desc(data, columns, precomputed=None):
    """명목형 기술통계량 표를 얻는다 (미리 산출한 표가 있으면 재사용)."""
    if precomputed is not None:
        return precomputed
    if not columns:
        return None
    return my_qtcheck.categorical_summary(data, columns=columns, value_counts=False)


# ===================================================================
# 탭 ① #01 준비작업 (변수 분류 · 기술통계량 · EDA 범위)
# ===================================================================
# 노트북 #01-7 'EDA 범위 정리' 표의 행 정의
# (종속변수 유형, 독립변수 유형, 분석 유형)
_SCOPE_ROWS = [
    ("연속형",        "연속형",        "상관분석"),
    ("연속형",        "명목형(2집단)",  "T검정"),
    ("연속형",        "명목형(3집단↑)", "ANOVA"),
    ("명목형(2집단)",  "연속형",        "T검정"),
    ("명목형(2집단)",  "명목형(2집단)",  "교차분석"),
    ("명목형(2집단)",  "명목형(3집단↑)", "교차분석"),
    ("명목형(3집단↑)", "연속형",        "ANOVA"),
    ("명목형(3집단↑)", "명목형(2집단)",  "교차분석"),
    ("명목형(3집단↑)", "명목형(3집단↑)", "교차분석"),
]


def _dependent_kind(data, target, is_clf):
    """종속변수의 유형 라벨을 EDA 범위 표의 표기법으로 반환한다."""
    if not is_clf:
        return "연속형"
    return "명목형(2집단)" if _nlevels(data, target) == 2 else "명목형(3집단↑)"


def _feature_group(data, kind, feat_cont, feat_nom):
    """독립변수 유형 라벨에 해당하는 실제 변수 목록을 반환한다."""
    if kind == "연속형":
        return list(feat_cont)
    if kind == "명목형(2집단)":
        return [n for n in feat_nom if _nlevels(data, n) == 2]
    return [n for n in feat_nom if _nlevels(data, n) >= 3]


def _scope_frame(data, target, is_clf, feat_cont, feat_nom, column_means=None):
    """노트북 #01-7 의 'EDA 범위 정리' 표를 데이터로부터 자동 생성한다.

    종속변수의 유형이 일치하는 행만 후보가 되고, 그 중 해당 유형의 독립변수가
    실제로 존재하는 행만 '✅ 적용' 이 된다.
    """
    dep_kind = _dependent_kind(data, target, is_clf)

    rows = []
    for dep, indep, method in _SCOPE_ROWS:
        if dep != dep_kind:                  # 종속변수 유형이 다르면 후보조차 아니다
            cols, applied, note = [], "❌ 미적용", f"종속변수 유형이 {dep_kind}"
        else:
            cols = _feature_group(data, indep, feat_cont, feat_nom)
            applied = "✅ 적용" if cols else "❌ 미적용"
            note = f"{indep} 독립변수 {len(cols)}종" if cols else f"{indep} 독립변수 없음"

        rows.append({
            "종속유형": dep, "독립유형": indep, "분석 유형": method,
            "대상 변수": ", ".join(_label(c, column_means) for c in cols) or "—",
            "적용 여부": applied, "비고": note,
        })
    return DataFrame(rows)


def show_descriptive(data, target, task="regression", column_means=None,
                     num_desc=None, cat_desc=None, _mode="display"):
    """#01 준비작업 탭: 변수 분류 · 기술통계량 · EDA 범위를 페이지로 구분해 출력한다.

    노트북의 #01(준비작업) 단계에 대응한다. 기술통계량은 품질점검 단계에서 미리
    산출해 둔 표(num_desc / cat_desc)를 전달하면 그대로 재사용하고, 전달하지
    않으면 my_qtcheck 로 직접 산출한다.
    (다중공선성 VIF 는 '#04 다변량 분석' 탭으로 옮겨졌다.)
    """
    is_clf = _resolve_task(task)
    number_cols, category_cols, feat_cont, feat_nom = _classify(data, target)
    ndesc = _num_desc(data, number_cols, num_desc)
    cdesc = _cat_desc(data, category_cols, cat_desc)

    # ── 변수 분류: 종속변수·모델링 유형·독립변수 목록(의미 포함) ──
    def basic():
        display(Markdown("#### 변수 유형 분류"))
        display(Markdown(f"- **종속변수** : `{_label(target, column_means)}`"))
        display(Markdown(f"- **종속변수 유형** : "
                         f"{'명목형(분류)' if is_clf else '연속형(예측)'}"))
        display(Markdown("- **연속형 독립변수** : "
                         + (", ".join(f"`{_label(c, column_means)}`" for c in feat_cont) or "없음")))
        display(Markdown("- **명목형 독립변수** : "
                         + (", ".join(f"`{_label(c, column_means)}`" for c in feat_nom) or "없음")))

    # ── 연속형 기술통계량 ──
    def numeric():
        display(Markdown("#### 연속형 기술통계량"))
        display(ndesc)

    # ── 명목형 기술통계량 ──
    def category():
        display(Markdown("#### 명목형 기술통계량"))
        display(cdesc)

    # ── EDA 범위 정리 (노트북 #01-7) ──
    def scope():
        display(Markdown("#### EDA 범위 정리"))
        _blank()
        display(Markdown(
            f"종속변수 `{target}` 가 **{_dependent_kind(data, target, is_clf)}** 이므로 "
            "해당 종속유형의 행만 후보가 되고, 그 중 독립변수가 실제로 존재하는 "
            "행만 적용된다."))
        _blank()
        frame = _scope_frame(data, target, is_clf, feat_cont, feat_nom, column_means)

        def _mark(r):
            color = "#e8f5e9" if r["적용 여부"] == "✅ 적용" else ""
            return [f"background-color: {color}"] * len(r)

        display(frame.style.apply(_mark, axis=1).hide(axis="index"))

    sections = [("변수 분류", basic)]
    if ndesc is not None:
        sections.append(("연속형 기술통계량", numeric))
    if cdesc is not None:
        sections.append(("명목형 기술통계량", category))
    sections.append(("EDA 범위", scope))

    return _emit(sections, _mode)


# ===================================================================
# #02 단변량 분석 - 그래프 아래에 표시할 변수별 기술통계량
# ===================================================================
def _desc_of(data, var, num_desc=None):
    """해당 변수가 담긴 연속형 기술통계량 표를 얻는다.

    미리 산출해 둔 표(num_desc)에 그 변수가 있으면 그대로 쓰고, 없으면 그 변수만
    새로 산출한다. 기술통계량 표시와 판정표가 같은 표를 보도록 한 곳에 모았다.
    """
    if num_desc is not None and var in num_desc.index:
        return num_desc
    return my_qtcheck.numerical_summary(data, columns=[var])


def _var_stats(data, var, continuous, num_desc=None, cat_desc=None):
    """단변량 그래프 아래에 해당 변수의 기술통계량을 표로 표시한다.

    - 연속형: 연속형 기술통계량에서 해당 변수 행만 뽑아 전치(세로 표)해 표시
    - 명목형: 명목형 기술통계량 + 범주별 빈도·비율 표시 (노트북 #02-3 과 동일)
    """
    display(Markdown("**기술통계량**"))
    if continuous:
        desc = _desc_of(data, var, num_desc)
        display(desc[desc.index == var].T)
        return

    if cat_desc is not None and var in cat_desc.columns:
        display(cat_desc[[var]].T)

    # 빈도표 (빈도 + 전체 대비 비율) — 한 번 센 결과를 나눠 비율까지 만든다
    counts = data[var].value_counts()
    freq = counts.rename("빈도").to_frame()
    freq["비율"] = counts / counts.sum()
    display(freq)


def _univariate_plot(data, var, title, continuous, palette=None):
    """단변량 분포 그래프를 그린다.

    - 연속형: 히스토그램 + 상자그림 (1행 2열)
    - 명목형: 범주별 빈도 막대그래프
    """
    if continuous:
        fig, ax = my_plot.init(rows=1, cols=2, title=title,
                               width=_FIG_WIDTH, height=_FIG_HEIGHT)
        my_plot.histplot(data=data, x=var, kde=True, ax=ax[0])
        my_plot.boxplot(data=data, x=var, ax=ax[1])
        my_plot.show()
    else:
        my_plot.countplot(data=data, x=var, hue=var,
                          palette=palette or _NOMINAL_PALETTE,
                          title=title, width=_FIG_WIDTH, height=_FIG_HEIGHT)


def _univariate_page(data, var, title, continuous, palette=None,
                     num_desc=None, cat_desc=None, is_target=False, is_clf=False):
    """단변량 분석 페이지 렌더함수를 만든다 (종속·독립변수 공용).

    한 페이지는 '분포 그래프 → 기술통계량 → 인사이트 판정표' 순서로 구성된다.
    """
    def render():
        _univariate_plot(data, var, title, continuous, palette)
        _var_stats(data, var, continuous, num_desc=num_desc, cat_desc=cat_desc)
        _blank()
        _show_insight(_insight_continuous(data, var, num_desc) if continuous
                      else _insight_nominal(data, var, is_target=is_target, is_clf=is_clf))
    return render


# ===================================================================
# 탭 ② #02 단변량 분석 - 종속변수
# ===================================================================
def viz_dependent(data, target, task="regression", palette=None, column_means=None,
                  num_desc=None, cat_desc=None, _mode="display"):
    """#02-1 종속변수 단변량 분석: 분포 그래프 + 기술통계량 표.

    - 연속형: 히스토그램 + 상자그림
    - 명목형: 빈도 막대그래프
    """
    is_clf = _resolve_task(task)
    name = _label(target, column_means)

    sections = [(f"종속 · {name}",
                 _univariate_page(data, target, f"종속변수 {name} 분포",
                                  continuous=not is_clf, palette=palette,
                                  num_desc=num_desc, cat_desc=cat_desc,
                                  is_target=True, is_clf=is_clf))]
    return _emit(sections, _mode)


# ===================================================================
# 탭 ③ #02 단변량 분석 - 독립변수
# ===================================================================
def viz_independent(data, target, task="regression", palette=None, column_means=None,
                    num_desc=None, cat_desc=None, _mode="display"):
    """#02-2·3 독립변수 단변량 분석: 독립변수별 페이지로 분포를 그린다.

    - 연속형: 히스토그램 + 상자그림 + 기술통계량
    - 명목형: 빈도 막대그래프 + 기술통계량 + 빈도·비율표
    """
    is_clf = _resolve_task(task)
    _, _, feat_cont, feat_nom = _classify(data, target)

    def page(var, continuous):
        name = _label(var, column_means)
        prefix = "연속" if continuous else "명목"
        return (f"{prefix} · {name}",
                _univariate_page(data, var, f"독립변수 {name} 분포",
                                 continuous=continuous, palette=palette,
                                 num_desc=num_desc, cat_desc=cat_desc, is_clf=is_clf))

    sections = ([page(f, True) for f in feat_cont]
                + [page(n, False) for n in feat_nom])

    return _emit(sections, _mode)


# ===================================================================
# 탭 ④ #03 이변량 분석 - 연속 × 연속 (상관분석)
# ===================================================================
def infer_cont_cont(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None,
                    column_means=None, _mode="display"):
    """#03-1 상관분석 (x: 연속형 독립변수 → y: 연속형 종속변수).

    연속형 종속변수(예측)일 때만 수행된다. 노트북과 동일하게 산점도를 먼저 그리고
    상관분석 결과표를 이어서 표시한다.

    Args:
        verbose (bool): True 면 산점도·결과표를 표시, False 면 결과만 수집.
        _mode (str): "display"(그 자리 표시) / "widget" / "sections"

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트
                    (_mode 가 "widget"/"sections" 이면 (rows, 위젯|섹션) 튜플)
    """
    if _resolve_task(task):
        return _emit_with_rows([], [], _mode)      # 연속형 종속변수 아님 → 대상 없음
    _, _, feat_cont, _ = _classify(data, target)

    def work(f, plot):
        row, res = _run_correlation(data, f, target, alpha, corr_threshold)
        if plot and row is not None:
            display(Markdown(f"#### ▶︎ {_label(f, column_means)} → {target}"))
            # 데이터 밀도에 따라 외곽선 굵기와 점 크기를 줄여 겹침을 완화한다.
            my_plot.scatterplot(data=data, x=f, y=target, outline=False,
                                linewidth=0.2, size=30, palette=palette or "tab10",
                                title=f"{_label(f, column_means)} ↔ {target}",
                                width=_FIG_WIDTH, height=_FIG_HEIGHT)
            display(res)
            _blank()
            _show_insight(_insight_correlation(data, f, target, res, row, alpha))
        return row

    items = [(f"상관 · {_label(f, column_means)} ↔ {target}", f) for f in feat_cont]
    return _emit_sections(items, work, verbose, mode=_mode)


# ===================================================================
# 탭 ④-보조 #03 이변량 분석 - (연속 × 종속) + 범주
# -------------------------------------------------------------------
# 이변량 산점도에 명목형 변수를 색으로 얹어, 범주에 따른 관계 차이(상호작용·교란
# 후보)를 눈으로 확인한다. 노트북 기본 흐름에는 없는 보조 단계이므로 eda() 에서는
# interaction=True 일 때만 포함된다.
# ===================================================================
def viz_cont_target_hue(data, target, task="regression", palette=None,
                        column_means=None, _mode="display"):
    """(연속 × 종속) + 범주 시각화 (연속독립 × 명목 조합별 페이지).

    연속형 독립변수(x)와 연속형 종속변수(y)의 산점도에 명목형 변수를 hue(색)로
    적용해, 범주에 따라 관계가 어떻게 갈리는지 보여준다.
    """
    # 명목형 종속변수(분류)에는 '연속 × 종속' 산점도 자체가 없으므로 대상 아님
    if _resolve_task(task):
        return _emit([], _mode)
    _, _, feat_cont, feat_nom = _classify(data, target)

    # hue 로 쓸 만한 명목형 변수만 추림 (2 ≤ 범주 수 ≤ 상한)
    hue_vars = [c for c in feat_nom if 2 <= _nlevels(data, c) <= _MAX_HUE_LEVELS]
    if not feat_cont or not hue_vars:
        return _emit([], _mode)

    def _page(f, c):
        def render():
            my_plot.scatterplot(data=data, x=f, y=target, hue=c,
                                palette=palette or "tab10", outline=False,
                                linewidth=0.2, size=30,
                                title=f"{_label(f, column_means)} ↔ {target} · 색: {c}",
                                xlabel=f, ylabel=target,
                                width=_FIG_WIDTH, height=_FIG_HEIGHT)
            display(Markdown(
                f"색은 명목형 변수 **`{_label(c, column_means)}`** 의 범주를 구분한다. "
                "범주마다 점의 기울기·위치가 다르면 **상호작용·교란 후보**로 볼 수 있다."))
        return render

    # 연속형 독립변수(바깥) × 명목형 변수(안쪽) 조합별 페이지
    sections = [(f"상호작용 · {f} ↔ {target} · {c}", _page(f, c))
                for f in feat_cont for c in hue_vars]

    return _emit(sections, _mode)


# ===================================================================
# 탭 ⑤ #03 이변량 분석 - 연속 × 명목(2집단) (T검정)
# ===================================================================
def infer_cont_nom2(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None,
                    column_means=None, _mode="display"):
    """#03-2 T검정 (x: 2집단 명목형 독립변수 → y: 종속변수).

    - 예측: 2수준 명목형 독립변수 ↔ 연속형 종속변수
    - 분류: 연속형 독립변수 ↔ 2범주 종속변수

    2개의 고유값만 존재하는 명목형이 없으면 결과가 없을 수 있다.

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트 (대상 없으면 빈 리스트)
    """
    is_clf = _resolve_task(task)
    _, _, feat_cont, feat_nom = _classify(data, target)

    # (cat, cont, 표시명) 쌍 구성
    if is_clf:
        if _nlevels(data, target) != 2:
            return _emit_with_rows([], [], _mode)
        pairs = [(target, f, f) for f in feat_cont]
    else:
        pairs = [(n, target, n) for n in feat_nom if _nlevels(data, n) == 2]

    def work(pair, plot):
        cat, cont, name = pair
        label = _label(name, column_means)
        row, res = _run_ttest(data, cat, cont, name, alpha, corr_threshold,
                              plot=plot, palette=palette, title=f"{label} ↔ {target}")
        if plot and row is not None:
            display(Markdown(f"#### ▶︎ {label} → {target}"))
            display(res)
            _blank()
            _show_insight(_insight_ttest(res, row, alpha))
        return row

    items = [(f"T검정 · {_label(name, column_means)} ↔ {target}", (cat, cont, name))
             for cat, cont, name in pairs]
    return _emit_sections(items, work, verbose, mode=_mode)


# ===================================================================
# 탭 ⑥ #03 이변량 분석 - 연속 × 명목(3집단↑) (분산분석)
# ===================================================================
def infer_cont_nom3(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, verbose=True, palette=None,
                    column_means=None, posthoc_plot=False, _mode="display"):
    """#03-3 ANOVA + 사후검정 (x: 3집단↑ 명목형 독립변수 → y: 종속변수).

    - 예측: 3수준 이상 명목형 독립변수 ↔ 연속형 종속변수
    - 분류: 연속형 독립변수 ↔ 3범주 이상 종속변수

    Args:
        posthoc_plot (bool): 사후검정 시각화 수행 여부. 노트북과 동일하게 기본은
            False(결과표만 표시)이며, 보고서에 그래프가 필요할 때만 True 로 둔다.

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트 (대상 없으면 빈 리스트)
    """
    is_clf = _resolve_task(task)
    _, _, feat_cont, feat_nom = _classify(data, target)

    if is_clf:
        if _nlevels(data, target) < 3:
            return _emit_with_rows([], [], _mode)
        pairs = [(target, f, f) for f in feat_cont]
    else:
        pairs = [(n, target, n) for n in feat_nom if _nlevels(data, n) >= 3]

    def work(pair, plot):
        cat, cont, name = pair
        label = _label(name, column_means)
        row, aov = _run_anova(data, cat, cont, name, alpha, corr_threshold)
        if plot and row is not None:
            display(Markdown(f"#### ▶︎ {label} → {target}"))
            display(aov)
            # 사후검정(그룹 쌍별 비교). 시각화는 선택 사항이며, 그릴 경우 집단 수가
            # 많으면 유의성 주석 막대가 세로로 쌓이므로 높이를 동적으로 키운다.
            ph = my_stats.posthoc_oneway(
                data, y=cont, between=cat, alpha=alpha,
                plot=posthoc_plot, palette=palette,
                title=f"{label} 사후검정",
                height=_posthoc_height(_nlevels(data, cat)))
            display(ph)
            _blank()
            _show_insight(_insight_anova(aov, ph, cat, row, alpha))
        return row

    items = [(f"ANOVA · {_label(name, column_means)} ↔ {target}", (cat, cont, name))
             for cat, cont, name in pairs]
    return _emit_sections(items, work, verbose, mode=_mode)


# ===================================================================
# 탭 ⑦ #03 이변량 분석 - 명목 × 명목 (교차분석)
# ===================================================================
def infer_nom_nom(data, target, task="regression", alpha=0.05,
                  corr_threshold=0.3, verbose=True, palette=None,
                  column_means=None, _mode="display"):
    """#03-4 교차분석 (x: 명목형 독립변수 → y: 명목형 종속변수).

    명목형 종속변수(분류)일 때만 수행되며, 각 명목형 독립변수와 종속변수의
    독립성 검정(카이제곱/피셔)을 실행한다.

    Returns:
        list[dict]: 변수 선별표에 사용할 결과행 리스트
    """
    if not _resolve_task(task):
        return _emit_with_rows([], [], _mode)      # 명목형 종속변수 아님 → 대상 없음
    _, _, _, feat_nom = _classify(data, target)

    def work(n, plot):
        row, res = _run_chi2(data, n, target, alpha, corr_threshold,
                             plot=plot, palette=palette)
        if plot and row is not None:
            display(Markdown(f"#### ▶︎ {_label(n, column_means)} → {target}"))
            display(res)
            _blank()
            _show_insight(_insight_chi2(res, row, alpha))
        return row

    items = [(f"교차 · {_label(n, column_means)} ↔ {target}", n) for n in feat_nom]
    return _emit_sections(items, work, verbose, mode=_mode)


# ===================================================================
# 탭 ⑧ #04 다변량 분석 - 독립변수 간 상관 (공선성 신호의 근거)
# ===================================================================
def corr_independent(data, target, alpha=0.05, collinearity_threshold=0.7,
                     palette=None, corr_pairs=None, _mode="display"):
    """#04-1 독립변수 간 상관분석 + 다중공선성.

    연속형 독립변수들 사이의 상관행렬과 변수별 중복(공선성) 집계표를 산출한다.
    이는 변수 선별표의 공선성 신호(🅾️/❎)의 근거가 된다. 연속형 독립변수가
    2개 미만이면 대상이 없다.

    Args:
        corr_pairs (DataFrame): 이미 산출한 쌍별 상관분석 결과표(있으면 재계산 생략)
    """
    _, _, feat_cont, _ = _classify(data, target)
    if len(feat_cont) < 2:
        return _emit([], _mode)

    pairs = corr_pairs if corr_pairs is not None else \
        _compute_corr_pairs(data, feat_cont, alpha)
    if pairs is None:
        return _emit([], _mode)

    # ── 상관행렬 · 변수별 중복 집계 ──
    def corr_page():
        display(Markdown("#### 연속형 독립변수 간 상관행렬"))
        _blank()
        display(Markdown("색이 진할수록 강한 상관 → 서로 중복(공선성) 후보"))
        _blank()
        display(_corr_matrix(pairs, feat_cont).style
                .background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
                .format("{:.3f}"))
        _blank()
        # 변수별로 '가장 강한 상대'와 '중복 쌍의 수'를 집계 (노트북 #04-1 과 동일)
        display(Markdown(f"**변수별 중복(공선성) 집계 — |coef|≥{collinearity_threshold} 인 쌍**"))
        _blank()
        display(my_stats.correlation_summary(pairs))

    # ── 다중공선성(VIF) ──
    def vif_page():
        display(Markdown("#### 다중공선성(VIF)"))
        _blank()
        vif = _compute_vif(data, feat_cont)
        if vif is not None:
            display(vif)
        else:
            display(Markdown("> VIF 를 계산할 수 없습니다."))

    sections = [("독립변수 상관", corr_page), ("다중공선성(VIF)", vif_page)]
    return _emit(sections, _mode)


# ===================================================================
# 탭 ⑨ #05 최종 변수 선택 (변수 선별표)
# ===================================================================
# 노트북 #05 '최종 변수 선택' 표의 컬럼 순서
_SELECT_COLUMNS = ["채택여부", "변수", "검정방법", "유의수준 (p)", "효과크기",
                   "비대칭신호", "공선성신호", "근거", "가정점검"]


def _build_selection_frame(rows, desc, collin):
    """결과행 리스트에 비대칭·공선성 신호를 결합하고 정렬하여 표로 만든다.

    공선성 신호는 VIF가 아니라 연속형 독립변수 간 상관분석 결과(collin)로만 산출한다.
    공선성으로 이어진 변수들은 그룹으로 묶어, 그 중 효과크기 1위만 대표로 채택하고
    나머지는 후보로 내린다.
    """
    if not rows:
        return DataFrame(columns=_SELECT_COLUMNS)

    # 공선성으로 이어진 변수들을 그룹(연결 성분)으로 묶는다
    groups = _collinearity_groups(collin) if collin else {}

    for row in rows:
        var = row["변수"]
        row["비대칭신호"] = _asymmetry_signal(desc, var)
        row["공선성신호"] = _collinearity_signal(collin, var, groups)

    # 그룹마다 대표변수 하나만 채택으로 남기고 나머지는 후보로 내린다
    if groups:
        _select_group_representative(rows, groups)

    table = DataFrame(rows)
    # 정렬: 채택(0)→후보(1)→제외(2), 그 안에서 효과크기↓ · p값↑
    rank = {"✅ 채택": 0, "🟡 후보": 1, "❌ 제외": 2}
    table["_rank"] = table["채택여부"].map(rank)
    table = table.sort_values(by=["_rank", "_effect", "_p"],
                              ascending=[True, False, True])
    return table[_SELECT_COLUMNS].reset_index(drop=True)


def _display_selection(table, alpha, corr_threshold, collinearity_threshold):
    """변수 선별표를 채택여부별 배경색과 함께 출력한다."""
    display(Markdown("### 📋 최종 변수 선택"))
    display(Markdown(f"기준 : 유의수준 α={alpha}, 상관 채택 |coef|≥{corr_threshold}, "
                     f"공선성 신호 |r|≥{collinearity_threshold} (상관분석 기반 · VIF 미적용)  ·  "
                     f"공선성 🅾️ G*=그룹번호 / ❎=없음  ·  "
                     f"분류 ✅ 채택 / 🟡 후보 / ❌ 제외"))
    display(Markdown("같은 공선성 그룹(G*)의 변수는 사실상 같은 정보이므로 **효과크기 1위 하나만 "
                     "대표로 채택**하고 나머지는 후보로 둔다. 근거의 **Δ**(1위와의 효과크기 격차)가 "
                     "매우 작다면 통계적으로는 동률이므로, 대표 선택은 도메인 지식으로 다시 확인한다."))
    if not len(table):
        display(Markdown("> 검정 가능한 독립변수가 없습니다."))
        return

    def _highlight(r):
        color = {"✅ 채택": "#e8f5e9", "🟡 후보": "#fffde7",
                 "❌ 제외": "#ffebee"}.get(r["채택여부"], "")
        return [f"background-color: {color}"] * len(r)

    display(table.style.apply(_highlight, axis=1))


def selection_table(data, target, task="regression", alpha=0.05,
                    corr_threshold=0.3, collinearity_threshold=0.7,
                    palette=None, column_means=None, num_desc=None):
    """#05 최종 변수 선택: 4종 추론통계를 조용히 수행하여 선별표만 출력한다.

    노트북에서 단독 호출하면 개별 검정 과정 출력 없이 선별표만 확인할 수 있다.
    공선성 신호는 연속형 독립변수 간 상관분석 결과로만 산출한다(VIF 미적용).
    """
    number_cols, _, feat_cont, _ = _classify(data, target)
    desc = _num_desc(data, number_cols, num_desc)
    pairs = _compute_corr_pairs(data, feat_cont, alpha)
    collin = _collinearity_map(pairs, feat_cont, collinearity_threshold)

    rows = []
    for fn in _INFER_FUNCS:
        rows.extend(fn(data, target, task=task, alpha=alpha,
                       corr_threshold=corr_threshold, verbose=False,
                       palette=palette, column_means=column_means))

    table = _build_selection_frame(rows, desc, collin)
    _display_selection(table, alpha, corr_threshold, collinearity_threshold)


# ===================================================================
# 파이프라인 본체 (노트북 #01~#05 흐름을 그대로 탭으로 구성)
# ===================================================================
# #03 이변량 분석 탭을 구성하는 검정 함수들 (나열된 순서가 곧 페이지 순서).
# 페이지 제목의 기법 접두어("상관 · ", "ANOVA · " 등)는 각 함수가 직접 붙인다.
_INFER_FUNCS = [infer_cont_cont, infer_cont_nom2, infer_cont_nom3, infer_nom_nom]


def eda(data, target, task="regression", alpha=0.05,
        corr_threshold=0.3, collinearity_threshold=0.7, palette=None,
        column_means=None, num_desc=None, cat_desc=None,
        interaction=False, posthoc_plot=False):
    """추론통계 기반 EDA 를 노트북의 분석 흐름 그대로 탭으로 실행하는 파이프라인.

    구성 탭 (노트북의 장 구분과 1:1 대응)

        #01 준비작업      변수 분류 · 연속형/명목형 기술통계량 · EDA 범위 정리
        #02 단변량 분석    종속변수 → 연속형 독립변수 → 명목형 독립변수
        #03 이변량 분석    상관분석 → T검정 → ANOVA → 교차분석
                          (interaction=True 면 '상호작용' 산점도 페이지 추가)
        #04 다변량 분석    독립변수 간 상관행렬·중복 집계 · 다중공선성(VIF)
        #05 최종 변수 선택 변수 선별표

    각 탭 안의 분석 단위는 드롭다운으로 넘기는 페이지로 구분된다.
    (ipywidgets 미설치 시 장 제목을 붙인 순차 출력으로 자동 대체된다.)

    Args:
        data (DataFrame): 품질점검·타입변환(서열척도 포함)이 완료된 데이터프레임
            (명목형은 category 타입으로 지정되어 있어야 자동 분류된다.)
        target (str): 종속변수(목표변수) 컬럼명
        task (str): 모델링 유형. 'regression'/'예측'/'연속형'(기본) 또는
            'classification'/'분류'/'명목형'
        alpha (float): 유의수준 (기본값: 0.05)
        corr_threshold (float): 상관계수 채택 기준 |coef| (기본값: 0.3)
        collinearity_threshold (float): 공선성 신호 기준 |r| (연속형 독립변수 간
            상관분석 기반, 기본값: 0.7). 이 단계에서 VIF는 사용하지 않는다.
        palette (str or list): 시각화 색상 팔레트 (기본값: None)
        column_means (dict): {컬럼명: 의미} 딕셔너리. 그래프·페이지 제목에
            `carat(중량)` 처럼 의미를 함께 표기한다. (기본값: None)
        num_desc (DataFrame): 미리 산출해 둔 연속형 기술통계량
            (품질점검 단계 산출물을 재사용할 때 전달. 기본값: None → 직접 산출)
        cat_desc (DataFrame): 미리 산출해 둔 명목형 기술통계량 (기본값: None)
        interaction (bool): #03 에 '(연속 × 종속) + 범주' 산점도 페이지를 추가할지
            여부 (기본값: False)
        posthoc_plot (bool): ANOVA 사후검정 시각화 여부 (기본값: False → 결과표만)
    """
    is_clf = _resolve_task(task)
    number_cols, category_cols, feat_cont, feat_nom = _classify(data, target)

    # 변수 선별표에 필요한 기술통계량·공선성 신호는 한 번만 계산해 재사용한다.
    # (공선성 신호는 VIF가 아니라 연속형 독립변수 간 상관분석 결과로만 산출)
    desc = _num_desc(data, number_cols, num_desc)
    cdesc = _cat_desc(data, category_cols, cat_desc)
    corr_pairs = _compute_corr_pairs(data, feat_cont, alpha)
    collin = _collinearity_map(corr_pairs, feat_cont, collinearity_threshold)

    # 각 단계로 넘길 공통 인자
    common = dict(task=task, palette=palette, column_means=column_means)
    infer_common = dict(task=task, alpha=alpha, corr_threshold=corr_threshold,
                        palette=palette, column_means=column_means)

    def _infer_kwargs(fn):
        """검정 함수별 추가 인자 (ANOVA 만 사후검정 시각화 옵션을 받는다)"""
        return {"posthoc_plot": posthoc_plot} if fn is infer_cont_nom3 else {}

    # -----------------------------------------------------------------
    # 폴백: ipywidgets 가 없으면 탭 없이 장 제목을 붙여 순차 출력
    # -----------------------------------------------------------------
    if not _HAS_WIDGETS:
        display(Markdown("## #01. 준비작업"))
        show_descriptive(data, target, task, column_means=column_means,
                         num_desc=desc, cat_desc=cdesc)

        display(Markdown("## #02. 단변량 분석"))
        viz_dependent(data, target, num_desc=desc, cat_desc=cdesc, **common)
        viz_independent(data, target, num_desc=desc, cat_desc=cdesc, **common)

        display(Markdown("## #03. 이변량 분석"))
        all_rows = []
        for fn in _INFER_FUNCS:
            rows = fn(data, target, verbose=True, **infer_common, **_infer_kwargs(fn))
            all_rows.extend(rows)
            # 상관분석 다음에 범주(hue) 산점도를 이어서 표시 (요청 시에만)
            if fn is infer_cont_cont and interaction:
                viz_cont_target_hue(data, target, **common)

        if corr_pairs is not None:
            display(Markdown("## #04. 다변량 분석"))
            corr_independent(data, target, alpha=alpha,
                             collinearity_threshold=collinearity_threshold,
                             palette=palette, corr_pairs=corr_pairs)

        display(Markdown("## #05. 최종 변수 선택"))
        table = _build_selection_frame(all_rows, desc, collin)
        _display_selection(table, alpha, corr_threshold, collinearity_threshold)
        return

    # -----------------------------------------------------------------
    # 위젯 모드: 장(章)별로 페이지 섹션을 모아 하나의 탭으로 만든다.
    #
    # 각 탭의 그래프 렌더링이 오래 걸리므로, 빌드 단계를 순서대로 나열한 뒤
    # 프로그래스바로 전체 진행 상황을 보여준다. 각 단계 함수는 실제 렌더링(느린
    # 작업)을 수행하고 (탭제목, 위젯) 또는 None(탭 없음) 을 돌려준다.
    # -----------------------------------------------------------------
    all_rows = []   # 변수 선별표 조립용 (#03 단계가 채운다)

    def _tab(title, sections):
        """페이지 섹션 목록을 하나의 탭으로 만든다 (비어 있으면 탭 없음)."""
        if not sections:
            return None
        widget = _pages_widget(sections)
        return (title, widget) if widget is not None else None

    def _step_prepare():
        return _tab("#01 준비작업",
                    show_descriptive(data, target, task, column_means=column_means,
                                     num_desc=desc, cat_desc=cdesc, _mode="sections"))

    def _step_univariate():
        sections = viz_dependent(data, target, num_desc=desc, cat_desc=cdesc,
                                 _mode="sections", **common)
        sections += viz_independent(data, target, num_desc=desc, cat_desc=cdesc,
                                    _mode="sections", **common)
        return _tab("#02 단변량 분석", sections)

    def _step_bivariate():
        sections = []
        for fn in _INFER_FUNCS:
            rows, secs = fn(data, target, verbose=True, _mode="sections",
                            **infer_common, **_infer_kwargs(fn))
            # 결과행은 섹션이 렌더링될 때 채워지므로, 지금은 목록만 붙잡아 두고
            # 실제 값은 위젯 생성이 끝난 뒤(#05 단계)에 읽는다.
            all_rows.append(rows)
            sections += secs
            # 상관분석 다음에 범주(hue) 산점도 페이지를 이어 배치 (요청 시에만)
            if fn is infer_cont_cont and interaction:
                sections += viz_cont_target_hue(data, target, _mode="sections", **common)
        return _tab("#03 이변량 분석", sections)

    def _step_multivariate():
        return _tab("#04 다변량 분석",
                    corr_independent(data, target, alpha=alpha,
                                     collinearity_threshold=collinearity_threshold,
                                     palette=palette, corr_pairs=corr_pairs,
                                     _mode="sections"))

    def _step_selection():
        # #03 단계에서 각 검정이 채운 결과행을 모아 선별표를 만든다.
        rows = [r for bucket in all_rows for r in bucket]
        table = _build_selection_frame(rows, desc, collin)
        sel_widget = _capture_to_html(
            lambda: _display_selection(table, alpha, corr_threshold,
                                       collinearity_threshold))
        return ("#05 최종 변수 선택", sel_widget)

    # 빌드 단계 조립: (진행 라벨, 실행함수) — 나열된 순서가 곧 탭 순서
    steps = [
        ("#01 준비작업", _step_prepare),
        ("#02 단변량 분석", _step_univariate),
        ("#03 이변량 분석", _step_bivariate),
        ("#04 다변량 분석", _step_multivariate),
        ("#05 최종 변수 선택", _step_selection),
    ]

    # 프로그래스바 표시 (전체 진행 상황)
    total = len(steps)
    progress = _IntProgress(min=0, max=total, value=0, bar_style="info",
                            layout={"width": "320px"})
    plabel = _Label(f"EDA 시작 — 총 {total}단계")
    progress_box = _HBox([progress, plabel])
    display(progress_box)

    # 단계별 실행 + 진행률 갱신 (실제 렌더링이 여기서 수행됨)
    tab_items = []   # (제목, 위젯)
    for i, (label, run) in enumerate(steps):
        plabel.value = f"({i + 1}/{total}) {label} 처리 중…"
        item = run()
        if item is not None:
            tab_items.append(item)
        progress.value = i + 1

    # 완료 표시 후 프로그래스바 정리
    progress.bar_style = "success"
    plabel.value = f"완료 — 탭 {len(tab_items)}개 구성"
    progress_box.close()

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

    # 결측치 (정적 HTML 위젯)
    def na_page():
        display(Markdown("#### 결측치 점검"))
        _blank()
        display(my_qtcheck.check_missing_values(df))
    tab_items.append(("결측치", _capture_to_html(na_page)))

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

    # 인사이트 (결론) — 마지막 탭 (정적 HTML 위젯)
    tab_items.append(("💡 인사이트(결론)",
                      _capture_to_html(lambda: _qtcheck_insight(df, dup_count, dup_removed))))

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


def qtcheck(data, save_path=None,
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



def _ols_vif_targets(data, y, cont_cols):
    """VIF 필터링 대상이 될 연속형 독립변수 목록을 결정한다.

    cont_cols 가 주어지면 그중 실제 존재하는 독립변수만 사용하고, None 이면
    수치형이면서 0/1 더미가 아닌 컬럼을 연속형으로 자동 판별한다.
    """
    if cont_cols is not None:
        return [c for c in cont_cols if c in data.columns and c != y]

    targets = []
    for c in data.columns:
        if c == y:
            continue
        s = data[c]
        if not np.issubdtype(s.dtype, np.number):
            continue                                # 범주형(문자·카테고리·불리언)은 VIF 대상 아님
        if set(np.unique(s.dropna().values).tolist()) <= {0, 1}:
            continue                                # 0/1 더미 변수도 제외
        targets.append(c)
    return targets


def _display_log_md(title, lines):
    """제거 과정 로그를 `제목 + 불릿 목록` 형태의 마크다운으로 출력한다.

    ols의 모든 출력을 Markdown 객체로 통일하기 위한 헬퍼다. 내용이 없으면
    제목만 덩그러니 남지 않도록 "해당 없음"을 대신 적는다.
    """
    body = [f"**{title}**", ""]

    items = [ln.strip() for ln in lines if ln.strip()]
    if items:
        body.extend(f"- {item}" for item in items)
    else:
        body.append("- 제거된 변수 없음")

    display(Markdown("\n".join(body)))


def _ols_auto_select(data, y, fit, summary, report, cont_cols, vif_threshold, sig_level):
    """VIF·유의성 기준으로 부적절한 독립변수를 자동 제거하고 재적합한다.

    Returns:
        tuple: (변수 제거가 반영된 DataFrame, 재적합된 회귀모형)
    """
    # 1) 다중공선성(VIF) 자동 제거 후 재적합
    # VIF는 수치 연산이므로 연속형(수치형) 독립변수만 대상으로 하고, 범주형·더미(0/1) 변수는 제외한다.
    # reduce_vif가 대상이 아닌 컬럼(범주형·더미·종속변수)을 원래 순서대로 보존하므로,
    # 필터링을 통과한 연속형 변수와 더미 변수가 자동으로 다시 결합된다.
    targets = _ols_vif_targets(data, y, cont_cols)

    # 대상 연속형 변수만으로 VIF를 계산해 임계값 초과가 있으면 그 대상만 제거·재결합
    vif_logs = []
    if targets and (my_stats.compute_vif(data, columns=targets)["VIF"] >= vif_threshold).any():
        # reduce_vif는 제거 과정을 print로 내보낸다(다른 호출자와 공유하는 동작이라 그대로 둔다).
        # 여기서는 그 표준출력을 가로채 두었다가 아래에서 마크다운 불릿으로 다시 내보낸다.
        with capture_output() as cap:
            data = my_prep.reduce_vif(data, columns=targets, threshold=vif_threshold, verbose=report)

        vif_logs = cap.stdout.splitlines()
        fit = my_ols.fit_model(data, y, summary=summary)

    if report:
        _display_log_md(f"다중공선성(VIF) 제거 (임계값 {vif_threshold})", vif_logs)

    # 2) 유의하지 않은 변수 순차 제거(후진제거) 후 재적합
    # 유의확률이 임계값 이상인 변수가 사라질 때까지, 유의확률이 가장 큰 변수부터 하나씩 제거하며 다시 적합한다.
    # (변수를 하나 빼면 남은 변수들의 유의확률이 바뀌므로 반드시 한 번에 하나씩 제거해야 한다)
    logs = []   # 제거 과정을 모아 두었다가 마크다운으로 한 번에 출력한다
    variables = my_ols.report_variables(fit, data)
    while len(variables) > 1 and variables["유의확률"].max() >= sig_level:
        worst_idx = variables["유의확률"].idxmax()
        worst = variables.loc[worst_idx, "독립변수"]
        logs.append(f"`{worst}` 제거 (유의확률 = {variables.loc[worst_idx, '유의확률']:.3f})")
        data = data.drop(columns=[worst])
        fit = my_ols.fit_model(data, y, summary=summary)
        variables = my_ols.report_variables(fit, data)

    if report:
        _display_log_md(f"유의성 기준 후진제거 (유의수준 {sig_level})", logs)

    return data, fit


def _ols_use_hc3(fit, robust, alpha=0.05):
    """등분산성 검정 결과에 따라 회귀계수표에 HC3 로버스트 표준오차를 쓸지 결정한다.

    등분산 가정이 위배되면(Breusch-Pagan 기각) 일반 OLS 표준오차는 과소·과대
    추정되므로, 이분산에 강건한 HC3 표준오차로 자동 전환한다.

    Returns:
        bool: 등분산 위배 시 True, 충족 시(또는 robust=False) False
    """
    if not robust:
        return False

    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
    # 등분산 충족시 True, 위배시 False (유의수준 alpha 기준)
    homoscedasticity = bool(float(f_p) >= alpha)

    return not homoscedasticity


def _ols_coef_note(hc3):
    """회귀계수표 위에 붙일 표준오차 방식 안내 문구를 만든다."""
    if not hc3:
        return "등분산성 가정을 충족하여 **일반 OLS 표준오차**로 보고한다."
    return ("⚠️ 등분산성 가정이 위배되어 이분산에 강건한 **HC3 표준오차**를 "
            "함께 보고한다. 계수(B)·β·공차·VIF 는 동일하며, "
            "`표준오차(HC3)`·`t(HC3)`·`유의확률(HC3)` 로 유의성을 판단한다.")


def _ols_coef_render(fit, data, hc3, log_y, log_x, log1p_y, log1p_x, beta_plot, width):
    """회귀계수 보고 일체(안내 문구 → 계수표 → 해석 문장 → β 영향력 그래프)를 출력한다.

    탭 방식과 순차 출력 방식이 같은 내용을 보여주도록 렌더링을 한 곳에 모은다.
    """
    display(Markdown(_ols_coef_note(hc3)))
    display(my_ols.report_variables(fit, data, hc3=hc3))

    # 독립변수별 회귀계수 해석 문장 (등분산 위배 시 HC3 기준 t·유의확률로 서술된다)
    display(Markdown(my_ols.report_variables_text(fit, log_y=log_y, log_x=log_x,
                                                  log1p_y=log1p_y, log1p_x=log1p_x,
                                                  hc3=hc3)))

    # 표준화계수(β) 기준 영향력 순위 그래프
    # height는 넘기지 않는다 → plot_beta가 독립변수 수 × 80으로 자동 계산한다.
    if beta_plot:
        my_ols.plot_beta(fit, data, width=width)


def _ols_test_sections(fit, plot, width, height):
    """회귀모형 가정 검정 4종을 (제목, 렌더함수) 목록으로 만든다."""
    return [
        ("1) 선형성 검정", lambda: my_ols.test_linear(fit, plot=plot, width=width, height=height)),
        ("2) 정규성 검정", lambda: my_ols.test_normal(fit, plot=plot, width=width, height=height)),
        ("3) 등분산성 검정", lambda: my_ols.test_equalvar(fit)),
        ("4) 독립성 검정", lambda: my_ols.test_independent(fit)),
    ]


def _ols_noninteractive(data, y, fit, summary, report, log_y, log_x, log1p_y, log1p_x,
                        test, plot, width, height,
                        auto_select, cont_cols, vif_threshold, sig_level, robust, beta_plot):
    """ipywidgets 가 없을 때의 순차 출력 회귀분석 (탭 없이 위에서 아래로 표시)."""
    if auto_select:
        display(Markdown("#### ▶︎ 변수 자동 선택"))
        data, fit = _ols_auto_select(data, y, fit, summary, report,
                                     cont_cols, vif_threshold, sig_level)
        _step_divider_md()

    if report:
        display(Markdown("#### ▶︎ 모형 적합도"))
        display(Markdown(my_ols.report_fitness(fit, log_y=log_y, log_x=log_x,
                                               log1p_y=log1p_y, log1p_x=log1p_x)))
        _step_divider_md()

        hc3 = _ols_use_hc3(fit, robust)
        display(Markdown("#### ▶︎ 회귀계수"))
        _ols_coef_render(fit, data, hc3, log_y, log_x, log1p_y, log1p_x, beta_plot, width)

    if report and test:
        _step_divider_md()

    if test:
        display(Markdown("#### ▶︎ 회귀모형 가정 검정"))
        for title, render in _ols_test_sections(fit, plot, width, height):
            display(Markdown(f"##### {title}"))
            render()

    return fit


def ols(data, y, summary = False, report = True, log_y=False, log_x=None, log1p_y=False, log1p_x=None, test=True, plot=False, width=1280, height=640, auto_select=True, cont_cols=None, vif_threshold=10.0, sig_level=0.05, robust=True, beta_plot=True):
    """자동으로 회귀분석을 수행하고, 결과를 탭으로 종합해 표시한다.

    구성 탭:
        (변수 자동 선택 / auto_select=True 일 때만) · 모형 적합도 · 회귀계수 ·
        (회귀모형 가정 검정 / test=True 일 때만, 드롭다운으로 4개 검정 전환)

    회귀계수 탭은 등분산성(Breusch-Pagan) 검정을 먼저 수행해, 등분산 가정이
    위배되면 이분산에 강건한 HC3 표준오차를 함께 보고하도록 자동 전환한다
    (robust=False 로 끌 수 있다). 이 판정은 계수표뿐 아니라 독립변수별 해석 문장의
    t·유의확률에도 함께 적용된다.

    회귀계수 탭에는 계수표·해석 문장에 이어 표준화계수(β) 기준 영향력 순위
    가로 막대그래프가 함께 표시된다 (beta_plot=False 로 끌 수 있다).

    auto_select=True이면 적합 후 다중공선성(VIF)과 유의성(유의확률)을 점검해, 부적절한
    독립변수를 자동으로 제거하고 다시 적합하는 과정을 반복한다. 전체 변수를 그대로 보고
    싶은 1차 분석 등에서는 auto_select=False로 자동 제거를 끈다.

    VIF 필터링은 연속형 변수만 대상으로 한다. cont_cols로 연속형 변수 이름 리스트를 직접
    지정하면 그 변수들에 대해서만 VIF를 계산·제거한다. 지정하지 않으면(None) 수치형이면서
    0/1 더미가 아닌 컬럼을 연속형으로 자동 판별한다. 어느 경우든 대상이 아닌 범주형·더미
    변수는 그대로 보존되어 VIF 제거를 통과한 연속형 변수와 다시 결합된다.

    Args:
        auto_select (bool): VIF·유의성 기준으로 부적절한 변수를 자동 제거할지 여부 (기본값: True).
        cont_cols (list, optional): VIF 필터링 대상으로 삼을 연속형 변수 이름 리스트. 지정 시 이
            리스트에 대해서만 VIF를 진행한다. None이면 연속형 변수를 자동 판별한다 (기본값: None).
        vif_threshold (float): VIF가 이 값 이상인 변수를 다중공선성 변수로 보고 제거한다 (기본값: 10.0).
        sig_level (float): 유의확률이 이 값 이상인 변수를 유의하지 않은 변수로 보고 제거한다 (기본값: 0.05).
        robust (bool): 등분산성 위배 시 회귀계수표를 HC3 로버스트 표준오차로 자동
            전환할지 여부 (기본값: True). False 면 항상 일반 OLS 표준오차로 보고한다.
        beta_plot (bool): 회귀계수 탭에 표준화계수(β) 영향력 순위 그래프를 함께
            표시할지 여부 (기본값: True). 그래프 높이는 독립변수 수에 맞춰 자동 계산된다.

    Returns:
        RegressionResults: 최종 적합된 회귀모형

    Note:
        ipywidgets 가 없으면 탭 없이 순차 출력한다.
    """
    # 회귀모델 적합
    fit = my_ols.fit_model(data, y, summary=summary)

    # 폴백: 순차 출력
    if not _HAS_WIDGETS:
        return _ols_noninteractive(data, y, fit, summary, report,
                                   log_y, log_x, log1p_y, log1p_x,
                                   test, plot, width, height,
                                   auto_select, cont_cols, vif_threshold, sig_level,
                                   robust, beta_plot)

    tab_items = []   # (탭 제목, 위젯)

    # ── 변수 자동 선택 ──
    # 제거 과정의 로그(print·verbose 출력)를 그대로 캡처해 탭에 담는다.
    # (_capture_to_html 이 render_fn 을 실행하므로 실제 재적합도 이 안에서 수행된다)
    if auto_select:
        result = {}

        def select_page():
            result["data"], result["fit"] = _ols_auto_select(
                data, y, fit, summary, report, cont_cols, vif_threshold, sig_level)

        widget = _capture_to_html(select_page)
        data, fit = result["data"], result["fit"]
        tab_items.append(("변수 자동 선택", widget))

    # ── 모형 적합도 · 회귀계수 ──
    if report:
        def fitness_page():
            display(Markdown(my_ols.report_fitness(fit, log_y=log_y, log_x=log_x,
                                                   log1p_y=log1p_y, log1p_x=log1p_x)))
        tab_items.append(("모형 적합도", _capture_to_html(fitness_page)))

        # 등분산성이 위배되면 회귀계수표를 HC3 로버스트 표준오차로 자동 전환
        hc3 = _ols_use_hc3(fit, robust)

        def coef_page():
            _ols_coef_render(fit, data, hc3, log_y, log_x, log1p_y, log1p_x, beta_plot, width)
        tab_items.append(("회귀계수", _capture_to_html(coef_page)))

    # ── 회귀모형 가정 검정 (드롭다운으로 넘기는 4개 페이지) ──
    if test:
        pages = _pages_widget(_ols_test_sections(fit, plot, width, height))
        if pages is not None:
            tab_items.append(("회귀모형 가정 검정", pages))

    if not tab_items:
        return fit

    tab = _Tab(children=[w for _, w in tab_items])
    for i, (t, _) in enumerate(tab_items):
        tab.set_title(i, t)
    display(tab)

    return fit