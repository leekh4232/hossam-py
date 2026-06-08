"""
hossam 명령줄 도구의 대화형 데이터 품질 점검(auto_qtcheck) 및 보고서 생성 기능.

라이브러리의 기본 점검 함수(set_type, check_missing_values, numerical_summary 등)는
my_qtcheck.py에 두고, 명령줄/대화형/HTML 보고서 관련 코드만 이 모듈로 분리했다.
"""

import contextlib
import io
import re as _re
import sys
from datetime import datetime
from html import escape as _esc

from pandas import DataFrame
from tabulate import tabulate as _tabulate

from .my_qtcheck import (
    get_number_column_names,
    get_categorical_column_names,
    check_missing_values,
    categorical_summary,
    numerical_summary,
)


# =====================================================================
# 대화형 데이터 품질 점검 자동화
# =====================================================================

def _prompt(message, default=""):
    """
    기본값을 지원하는 입력 프롬프트. 엔터만 누르면 기본값을 사용한다.

    Args:
        message (str): 사용자에게 보여줄 안내 문구
        default (str): 입력이 없을 때 사용할 기본값

    Returns:
        str: 사용자 입력값(없으면 기본값)
    """
    hint = _dim(f" [{default}]") if default != "" else ""
    try:
        answer = input(f"{_yellow('?')} {message}{hint}\n{_cyan('›')} ").strip()
    except EOFError:
        answer = ""
    return answer if answer else default


def _prompt_yes_no(message, default=True):
    """
    예/아니오 입력 프롬프트.

    Args:
        message (str): 질문 문구
        default (bool): 입력이 없을 때의 기본 선택

    Returns:
        bool: True(예) / False(아니오)
    """
    d = "Y/n" if default else "y/N"
    answer = _prompt(f"{message} ({d})", "").lower()
    if answer in ("y", "yes", "예", "ㅇ"):
        return True
    if answer in ("n", "no", "아니오", "ㄴ"):
        return False
    return default


# ---------------------------------------------------------------------
# 터미널 꾸미기 + HTML 보고서 작성 도우미
# ---------------------------------------------------------------------
# 터미널 출력은 ANSI 색상 + tabulate 표로, 보고서는 같은 내용을 HTML 조각으로
# 누적했다가 마지막에 스타일이 적용된 하나의 HTML 문서로 저장한다.
# (report 리스트에는 HTML 조각 문자열이 순서대로 쌓인다.)
_USE_COLOR = sys.stdout.isatty()
_WIDTH = 64  # 구분선 너비


def _c(text, code):
    """ANSI 색상 코드를 적용한다(터미널일 때만)."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c(t, "1")


def _cyan(t):
    return _c(t, "96")


def _green(t):
    return _c(t, "92")


def _yellow(t):
    return _c(t, "93")


def _red(t):
    return _c(t, "91")


def _dim(t):
    return _c(t, "90")


def _banner(title, subtitle=None):
    """프로그램 시작/종료 시 보여줄 박스형 배너를 출력한다."""
    line = "═" * _WIDTH
    print(_cyan(line))
    print(_cyan(_bold(f"  {title}")))
    if subtitle:
        print(_dim(f"  {subtitle}"))
    print(_cyan(line))


def _section_is_open(report):
    """현재 열려 있는 <section> 카드가 있는지(닫히지 않았는지) 확인한다."""
    opens = sum(f.count("<section") for f in report)
    closes = sum(f.count("</section>") for f in report)
    return opens > closes


def _section(report, step, total, title):
    """단계 구분 헤더를 터미널(구분선)과 보고서(카드)에 함께 기록한다."""
    line = "─" * _WIDTH
    print()
    print(_cyan(line))
    print(_cyan(_bold(f"  STEP {step}/{total}  {title}")))
    print(_cyan(line))

    # 이전 섹션 카드를 닫고 새 카드를 연다.
    if _section_is_open(report):
        report.append("</section>")
    sec_id = sum(f.count('<section') for f in report) + 1
    report.append(
        f'<section id="sec-{sec_id}" class="card">'
        f'<h2><span class="badge">STEP {step}/{total}</span>{_esc(title)}</h2>'
    )


def _status_html(kind, icon, msg):
    """상태 한 줄(성공/경고/안내/동작)을 HTML 문단으로 만든다."""
    return f'<p class="status {kind}"><span class="ico">{icon}</span>{_esc(msg)}</p>'


def _emit_text(report, term_msg, html_msg=None):
    """한 줄 메시지를 터미널과 보고서에 함께 기록한다(보고서는 평문 → HTML)."""
    print(term_msg)
    report.append(_status_html("plain", "", html_msg if html_msg is not None else term_msg))


def _ok(report, msg):
    print(_green(f"✅ {msg}"))
    report.append(_status_html("ok", "✅", msg))


def _warn(report, msg):
    print(_yellow(f"⚠️  {msg}"))
    report.append(_status_html("warn", "⚠️", msg))


def _info(report, msg):
    print(_dim(f"ℹ️  {msg}"))
    report.append(_status_html("info", "ℹ️", msg))


def _action(report, msg):
    print(f"➡️  {msg}")
    report.append(_status_html("action", "➡️", msg))


def _emit_df(report, df, caption=None, floatfmt=".3f", index=True):
    """
    DataFrame을 터미널에는 tabulate 표로, 보고서에는 HTML 표로 기록한다.

    Args:
        report (list): 보고서 HTML 조각 누적 리스트
        df (DataFrame): 출력할 데이터프레임
        caption (str): 표 위에 붙일 제목
        floatfmt (str): 실수 표시 형식
        index (bool): 인덱스 표시 여부
    """
    if caption:
        print(_bold(caption))
        report.append(f"<h3>{_esc(caption)}</h3>")

    print(
        _tabulate(
            df,
            headers="keys",
            tablefmt="rounded_outline",
            showindex=index,
            floatfmt=floatfmt,
            numalign="right",
        )
    )

    def _fmt(v):
        try:
            return format(v, floatfmt)
        except (ValueError, TypeError):
            return v

    table_html = df.to_html(
        index=index,
        border=0,
        classes="dtable",
        float_format=_fmt,
        na_rep="—",
    )
    report.append(f'<div class="table-wrap">{table_html}</div>')


# HTML 문서 골격 + 스타일(CSS). {title}/{body} 자리표시자만 치환한다.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg:#f4f6fb; --card:#ffffff; --ink:#1f2733; --muted:#6b7785;
  --line:#e6e9f0; --accent:#3b6ef0; --accent-soft:#eaf0ff;
  --ok:#1a7f4b; --ok-bg:#e7f6ee; --warn:#9a6700; --warn-bg:#fff6e0;
  --info:#5a6573; --info-bg:#f0f2f6; --action:#2a5fb0; --action-bg:#eef3ff;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#0f1420; --card:#161c2b; --ink:#e6eaf2; --muted:#9aa6b6;
    --line:#283041; --accent:#6f96ff; --accent-soft:#1d2740;
    --ok:#6ee7a8; --ok-bg:#11301f; --warn:#ffcf66; --warn-bg:#332806;
    --info:#aab4c4; --info-bg:#1b2231; --action:#9fc0ff; --action-bg:#16233f;
  }}
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    "Apple SD Gothic Neo","Noto Sans KR",Pretendard,sans-serif;
  line-height:1.6; font-size:15px;
}}
.container {{ max-width:980px; margin:0 auto; padding:32px 20px 80px; }}
header.report-head {{
  background:linear-gradient(135deg,var(--accent),#7a52f0);
  color:#fff; border-radius:18px; padding:28px 30px; margin-bottom:8px;
  box-shadow:0 10px 30px rgba(59,110,240,.25);
}}
header.report-head h1 {{ margin:0 0 12px; font-size:24px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; }}
.chip {{
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28);
  padding:4px 12px; border-radius:999px; font-size:13px; backdrop-filter:blur(4px);
}}
nav.toc {{
  position:sticky; top:12px; z-index:5; margin:18px 0 10px;
  background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:12px 16px; box-shadow:0 4px 14px rgba(0,0,0,.05);
}}
nav.toc strong {{ display:block; font-size:12px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }}
nav.toc ol {{ margin:0; padding-left:18px; columns:2; }}
nav.toc a {{ color:var(--accent); text-decoration:none; }}
nav.toc a:hover {{ text-decoration:underline; }}
section.card {{
  background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:22px 24px; margin:16px 0; box-shadow:0 4px 16px rgba(20,30,60,.05);
}}
section.card.notes {{ background:var(--accent-soft); }}
h2 {{ font-size:18px; margin:0 0 14px; display:flex; align-items:center; gap:10px; }}
.badge {{
  background:var(--accent); color:#fff; font-size:12px; font-weight:600;
  padding:3px 10px; border-radius:8px; white-space:nowrap;
}}
h3 {{ font-size:14px; color:var(--muted); margin:18px 0 8px; font-weight:600; }}
.status {{ margin:6px 0; padding:8px 12px; border-radius:10px;
  display:flex; gap:8px; align-items:flex-start; }}
.status .ico {{ flex:none; }}
.status.ok {{ background:var(--ok-bg); color:var(--ok); }}
.status.warn {{ background:var(--warn-bg); color:var(--warn); }}
.status.info {{ background:var(--info-bg); color:var(--info); }}
.status.action {{ background:var(--action-bg); color:var(--action); }}
.status.plain {{ background:var(--info-bg); }}
.table-wrap {{ overflow-x:auto; margin:6px 0 4px; border-radius:12px;
  border:1px solid var(--line); }}
table.dtable {{ border-collapse:collapse; width:100%; font-size:13.5px;
  font-variant-numeric:tabular-nums; }}
table.dtable th, table.dtable td {{ padding:8px 12px; text-align:right;
  border-bottom:1px solid var(--line); white-space:nowrap; }}
table.dtable thead th {{ position:sticky; top:0; background:var(--accent-soft);
  color:var(--ink); font-weight:600; text-align:right; }}
table.dtable tbody th {{ text-align:left; color:var(--muted); font-weight:600;
  background:transparent; }}
table.dtable tbody tr:nth-child(even) {{ background:rgba(127,127,127,.045); }}
table.dtable tbody tr:hover {{ background:var(--accent-soft); }}
.notes ul {{ margin:6px 0 0; padding-left:20px; }}
.notes li {{ margin:6px 0; color:var(--ink); }}
footer.report-foot {{ text-align:center; color:var(--muted); font-size:12px;
  margin-top:26px; }}
</style>
</head>
<body>
<div class="container">
{body}
<footer class="report-foot">Generated by hossam · 데이터 품질 점검 보고서</footer>
</div>
</body>
</html>
"""


def _build_toc(body_html):
    """본문에서 각 섹션 제목을 뽑아 목차(TOC)를 만든다."""
    items = _re.findall(
        r'<section id="(sec-\d+)"[^>]*><h2>(.*?)</h2>', body_html, flags=_re.S
    )
    if not items:
        return ""
    lis = []
    for sec_id, h2 in items:
        label = _re.sub(r"<[^>]+>", " ", h2)  # 내부 태그 제거
        label = _re.sub(r"\s+", " ", label).strip()
        lis.append(f'<li><a href="#{sec_id}">{label}</a></li>')
    return ('<nav class="toc"><strong>목차</strong><ol>'
            + "".join(lis) + "</ol></nav>")


def _save_report(report, path, title="데이터 품질 점검 보고서"):
    """누적된 HTML 조각을 하나의 스타일 적용 HTML 문서로 저장한다."""
    # 열려 있는 마지막 섹션을 닫는다.
    if _section_is_open(report):
        report.append("</section>")
    body = "\n".join(report)
    # 목차는 본문(헤더 다음)에 삽입한다. 헤더(</header>) 바로 뒤에 끼워 넣는다.
    toc = _build_toc(body)
    if toc and "</header>" in body:
        body = body.replace("</header>", "</header>\n" + toc, 1)
    elif toc:
        body = toc + body
    html_doc = _HTML_TEMPLATE.format(title=_esc(title), body=body)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def _parse_columns(raw, data):
    """
    콤마로 구분된 컬럼 문자열을 데이터프레임에 실제 존재하는 컬럼 리스트로 변환한다.

    Args:
        raw (str): "cut, color, clarity" 형태의 문자열
        data (DataFrame): 컬럼 존재 여부를 검사할 데이터프레임

    Returns:
        list: 유효한 컬럼명 리스트
    """
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    valid, invalid = [], []
    for c in cols:
        (valid if c in data.columns else invalid).append(c)
    if invalid:
        print(f"⚠️  존재하지 않는 컬럼은 무시합니다: {invalid}")
    return valid


def auto_qtcheck(data, dataset_name="dataset", report_path=None, save_report=True,
                 missing_drop_threshold=5.0):
    """
    데이터 품질 점검 프로세스를 대화형으로 자동 실행하는 함수.

    이 도구는 본격적인 정제가 아니라 '진단(diagnosis)'에 초점을 둔다. 결측치·
    이상치의 실제 처리는 EDA를 거친 뒤 모델링 직전에 수행하는 것을 전제로 하며,
    여기서는 무엇을 어떻게 처리하면 좋을지 판단에 필요한 통계를 보여 준다.

    호출하면 아래 단계를 순서대로 진행하며, 각 단계의 분기점마다 사용자 입력을
    받아 my_qtcheck의 기존 기능들을 자동으로 호출한다. 모든 질문은 엔터만 누르면
    대괄호 안의 기본값이 적용된다. 진행 내용은 HTML 보고서로도 자동 저장된다.

        1. 자료형 확인            → 컬럼별 dtype/결측 요약 표
        2. 자료형 변환            → category 변환
        3. 중복 점검/제거         → 중복 건수·비율 표시 후 제거 여부 질문
        4. 결측치 점검/삭제       → 건수·비율 표시 + '결측 행 < 임계값(%)일 때만
                                    삭제 권장' 원칙에 따라 기본값 결정
        5. 명목형 기술 통계량     → categorical_summary()
        6. 연속형 기술 통계량     → numerical_summary()
        7. 이상치 점검/삭제       → IQR 1.5배 경계 기준, 건수·비율 표시 후 질문
        8. 품질 검사 결과 저장    → 정제 완료본(Excel) + HTML 보고서

    Args:
        data (DataFrame): 품질 점검 대상 데이터프레임 (예: load_data로 불러온 origin)
        dataset_name (str): 저장 파일명 접두어로 사용할 데이터셋 이름
        report_path (str): HTML 보고서 저장 경로.
            None이면 "{dataset_name}_qtcheck_report.html"로 자동 지정한다.
        save_report (bool): HTML 보고서 저장 여부 (기본값: True)
        missing_drop_threshold (float): 결측 행 삭제를 '권장'하는 비율 임계값(%).
            결측치가 포함된 행의 비율이 이 값보다 작을 때만 삭제를 기본값으로
            제안하고, 그 이상이면 유지(모델링 단계 대치)를 권장한다. (기본값: 5.0)

    Returns:
        DataFrame: 모든 점검/정제를 마친 데이터프레임
    """
    TOTAL_STEPS = 8

    # 보고서 HTML 조각 누적용 리스트. 진행하면서 채워 마지막에 HTML 문서로 저장한다.
    report = []
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        now = ""
    report.append(
        '<header class="report-head">'
        f'<h1>📋 데이터 품질 점검 보고서 — {_esc(dataset_name)}</h1>'
        '<div class="meta">'
        f'<span class="chip">🕒 {_esc(now)}</span>'
        f'<span class="chip">📐 원본 {data.shape[0]:,}행 × {data.shape[1]:,}열</span>'
        '</div></header>'
    )

    _banner(f"🔍  데이터 품질 점검 — {dataset_name}",
            "각 질문에 엔터만 누르면 대괄호 안의 기본값이 적용됩니다.")

    #-----------------------------------------------------
    # 1) 자료형 확인
    #-----------------------------------------------------
    _section(report, 1, TOTAL_STEPS, "자료형 확인")
    info_df = DataFrame({
        "dtype": data.dtypes.astype(str),
        "non_null": data.notna().sum(),
        "nulls": data.isna().sum(),
    })
    info_df.index.name = "column"
    _emit_df(report, info_df, caption="컬럼별 자료형 / 결측 현황", floatfmt=".0f")

    #-----------------------------------------------------
    # 2) 자료형 변환
    #-----------------------------------------------------
    _section(report, 2, TOTAL_STEPS, "자료형 변환")
    object_cols = data.select_dtypes(include="object").columns.to_list()
    default_cat = ", ".join(object_cols)
    raw = _prompt("범주형(category)으로 변환할 컬럼을 콤마로 구분해 입력하세요. "
                  "(변환하지 않으려면 none 입력)", default_cat)

    if raw.lower() == "none" or raw == "":
        df = data.copy()
        _action(report, "자료형 변환을 건너뜁니다.")
    else:
        category_cols = _parse_columns(raw, data)
        df = data.copy()
        for col in category_cols:
            df[col] = df[col].astype("category")
        _ok(report, f"category로 변환: {category_cols}")
        conv_df = DataFrame({"dtype": df.dtypes.astype(str)})
        conv_df.index.name = "column"
        _emit_df(report, conv_df, caption="변환 후 자료형", floatfmt=".0f")

    n_rows = df.shape[0]

    #-----------------------------------------------------
    # 3) 중복 점검 / 제거
    #-----------------------------------------------------
    _section(report, 3, TOTAL_STEPS, "데이터 중복 점검")
    dup_count = int(df.duplicated().sum())
    dup_pct = (dup_count / n_rows * 100) if n_rows else 0.0

    if dup_count == 0:
        _ok(report, "중복된 행이 없습니다.")
    else:
        _warn(report, f"중복된 행: {dup_count:,}개 (전체 {n_rows:,}행의 {dup_pct:.2f}%)")
        if _prompt_yes_no("중복된 행을 제거할까요?", default=True):
            before = df.shape[0]
            df = df.drop_duplicates()
            removed = before - df.shape[0]
            _emit_text(
                report,
                f"🧹 중복 행 {removed:,}개 제거 "
                f"({removed / before * 100:.2f}%) → 현재 {df.shape[0]:,}행",
                f"- 🧹 중복 행 {removed:,}개 제거 ({removed / before * 100:.2f}%) → 현재 {df.shape[0]:,}행",
            )
        else:
            _action(report, "중복 행을 유지합니다.")

    n_rows = df.shape[0]

    #-----------------------------------------------------
    # 4) 결측치 점검 / 삭제
    #-----------------------------------------------------
    _section(report, 4, TOTAL_STEPS, "결측치 점검")
    missing = check_missing_values(df)
    # 결측치가 있는 컬럼만 표로 보여 준다(없으면 전체 표시).
    missing_view = missing[missing["Missing Count"] > 0]
    if missing_view.empty:
        missing_view = missing
    _emit_df(report, missing_view, caption="컬럼별 결측치")

    total_cells = int(df.size)
    total_missing = int(missing["Missing Count"].sum())
    cell_pct = (total_missing / total_cells * 100) if total_cells else 0.0
    rows_with_missing = int(df.isna().any(axis=1).sum())
    row_pct = (rows_with_missing / n_rows * 100) if n_rows else 0.0

    if total_missing == 0:
        _ok(report, "결측치가 없습니다.")
    else:
        _warn(report, f"총 결측치: {total_missing:,}개 "
                      f"(전체 셀 {total_cells:,}개의 {cell_pct:.2f}%)")
        _warn(report, f"결측치 포함 행: {rows_with_missing:,}개 "
                      f"(전체 {n_rows:,}행의 {row_pct:.2f}%)")

        # ── 변수(컬럼) 단위 결측률 임계값(기본 5%) 원칙 — Ryan(2013) ──
        # Ryan(2013)의 5% 기준은 '변수 단위 결측률'에 대한 것이다. 따라서
        # 어떤 컬럼도 임계값을 넘지 않을 때만 행 삭제를 권장하고, 한 컬럼이라도
        # 임계값 이상이면 행 삭제 대신 모델링 단계의 '대치(imputation)'를 권장한다.
        max_col_pct = float(missing["Missing Ratio (%)"].max())
        high_cols = missing.index[
            missing["Missing Ratio (%)"] >= missing_drop_threshold
        ].to_list()
        if high_cols:
            _warn(report, f"결측률 {missing_drop_threshold:.0f}% 이상 컬럼: {high_cols} "
                          f"→ 행 삭제보다 모델링 단계의 '대치(imputation)'를 권장합니다.")

        recommend_drop = max_col_pct < missing_drop_threshold
        if recommend_drop:
            _info(report, f"모든 컬럼 결측률 < {missing_drop_threshold:.0f}% "
                          f"(최대 {max_col_pct:.2f}%) → 삭제해도 무방합니다. "
                          f"(삭제 권장 · Ryan 2013)")
        else:
            _info(report, f"결측률 {missing_drop_threshold:.0f}% 이상 컬럼이 있습니다 "
                          f"(최대 {max_col_pct:.2f}%) → 삭제 시 손실·편향 위험. "
                          f"유지 후 모델링 단계에서 대치를 권장합니다. (유지 권장 · Ryan 2013)")

        if _prompt_yes_no("결측치가 포함된 행을 삭제할까요?", default=recommend_drop):
            before = df.shape[0]
            df = df.dropna()
            removed = before - df.shape[0]
            _emit_text(
                report,
                f"🧹 결측 행 {removed:,}개 제거 "
                f"({removed / before * 100:.2f}%) → 현재 {df.shape[0]:,}행",
                f"- 🧹 결측 행 {removed:,}개 제거 ({removed / before * 100:.2f}%) → 현재 {df.shape[0]:,}행",
            )
        else:
            _action(report, "결측치를 유지합니다.")

    n_rows = df.shape[0]

    #-----------------------------------------------------
    # 5) 명목형 변수 기술 통계량
    #-----------------------------------------------------
    _section(report, 5, TOTAL_STEPS, "기술 통계량 — 명목형 변수")
    cat_cols = get_categorical_column_names(df)
    if cat_cols:
        cat_save = None
        if _prompt_yes_no("명목형 요약을 Excel로 저장할까요?", default=True):
            cat_save = f"{dataset_name}_category_summary.xlsx"   # 용도에 맞는 파일명 자동 지정

        # Excel로 저장할 때는 value_counts 시트까지 함께 저장되도록 value_counts=True로 호출한다.
        # (categorical_summary가 콘솔로 직접 찍는 value_counts 출력은, 바로 아래에서 우리가
        #  더 보기 좋게 다시 출력하므로 중복을 막기 위해 그 동안만 표준출력을 가린다.)
        if cat_save:
            with contextlib.redirect_stdout(io.StringIO()):
                desc_cat = categorical_summary(df, value_counts=True, save_path=cat_save)
        else:
            desc_cat = categorical_summary(df, value_counts=False)

        _emit_df(report, desc_cat, caption="명목형 기술 통계량")
        for col in cat_cols:
            vc = DataFrame(df[col].value_counts())
            vc.index.name = col
            vc.sort_index(inplace=True)
            _emit_df(report, vc, caption=f"'{col}' value_counts", floatfmt=".0f")
        if cat_save:
            _emit_text(report, _green(f"💾 저장 완료: {cat_save} (value_counts 시트 포함)"),
                       f"💾 Excel 저장: {cat_save} (value_counts 시트 포함)")
    else:
        _info(report, "범주형(category) 컬럼이 없어 건너뜁니다. "
                      "(2단계에서 자료형을 변환했는지 확인하세요.)")

    #-----------------------------------------------------
    # 6) 연속형 변수 기술 통계량
    #-----------------------------------------------------
    _section(report, 6, TOTAL_STEPS, "기술 통계량 — 연속형 변수")
    num_save = None
    if _prompt_yes_no("연속형 요약을 Excel로 저장할까요?", default=True):
        num_save = f"{dataset_name}_numerical_summary.xlsx"      # 용도에 맞는 파일명 자동 지정
    desc_df = numerical_summary(df, save_path=num_save)

    # 터미널에는 핵심 컬럼만 추려서 보여 주고, 보고서에는 전체 표를 싣는다.
    key_cols = ["count", "mean", "std", "min", "50%", "max",
                "outliers", "outliers_ratio", "skew", "kurt", "log_need"]
    key_cols = [c for c in key_cols if c in desc_df.columns]
    print(_bold("연속형 기술 통계량 (핵심 항목)"))
    print(_tabulate(desc_df[key_cols], headers="keys",
                    tablefmt="rounded_outline", floatfmt=".3f", numalign="right"))
    # 보고서에는 전체 표를 HTML로 싣는다.
    report.append("<h3>연속형 기술 통계량 (전체)</h3>")
    report.append(
        '<div class="table-wrap">'
        + desc_df.to_html(border=0, classes="dtable",
                          float_format=lambda v: format(v, ".3f"), na_rep="—")
        + "</div>"
    )
    if num_save:
        _emit_text(report, _green(f"💾 저장 완료: {num_save}"),
                   f"- 💾 Excel 저장: `{num_save}`")

    #-----------------------------------------------------
    # 7) 이상치 점검 / 삭제
    #    numerical_summary가 계산한 IQR(1.5배) 경계값을 이용해 이상치 행을 탐지
    #-----------------------------------------------------
    _section(report, 7, TOTAL_STEPS, "이상치 점검")
    num_cols = get_number_column_names(df)
    lower = desc_df.loc[num_cols, "lower_bound"]
    upper = desc_df.loc[num_cols, "upper_bound"]
    outlier_mask = ((df[num_cols] < lower) | (df[num_cols] > upper)).any(axis=1)
    n_outlier_rows = int(outlier_mask.sum())
    out_pct = (n_outlier_rows / n_rows * 100) if n_rows else 0.0

    if n_outlier_rows == 0:
        _ok(report, "이상치가 없습니다. (IQR 1.5배 경계 기준)")
    else:
        _warn(report, f"이상치 포함 행: {n_outlier_rows:,}개 "
                      f"(전체 {n_rows:,}행의 {out_pct:.2f}%, IQR 1.5배 경계 기준)")
        if _prompt_yes_no("이상치가 포함된 행을 삭제할까요?", default=False):
            df = df[~outlier_mask]
            _emit_text(
                report,
                f"🧹 이상치 행 {n_outlier_rows:,}개 제거 "
                f"({out_pct:.2f}%) → 현재 {df.shape[0]:,}행",
                f"- 🧹 이상치 행 {n_outlier_rows:,}개 제거 ({out_pct:.2f}%) → 현재 {df.shape[0]:,}행",
            )
        else:
            _action(report, "이상치를 유지합니다.")

    #-----------------------------------------------------
    # 8) 정제 결과 저장 (결측치·이상치 처리가 모두 반영된 최종 데이터셋)
    #-----------------------------------------------------
    _section(report, 8, TOTAL_STEPS, "품질 검사 결과 저장")
    _info(report, f"정제 후 크기: {df.shape[0]:,}행 × {df.shape[1]:,}열")
    if _prompt_yes_no("정제된 데이터셋을 Excel로 저장할까요?", default=True):
        path = f"{dataset_name}_qtcheck.xlsx"          # 용도에 맞는 파일명 자동 지정
        df.to_excel(path, index=False)
        _emit_text(report, _green(f"💾 저장 완료: {path}"),
                   f"💾 정제 데이터 저장: {path}")
    else:
        _action(report, "Excel 저장을 건너뜁니다.")

    #-----------------------------------------------------
    # 9) 완료 — 참고(각주) 추가 후 HTML 보고서 저장 및 종료 배너
    #-----------------------------------------------------
    if _section_is_open(report):
        report.append("</section>")
    report.append(
        '<section class="card notes"><h2>📎 참고</h2><ul>'
        f'<li>결측치 삭제 기준({missing_drop_threshold:.0f}%)은 <b>경험칙(rule of thumb)</b>입니다. '
        f'변수(컬럼) 단위 결측률이 {missing_drop_threshold:.0f}% 미만이면 행 삭제의 영향이 작다는 '
        '관례에 따릅니다. — Ryan, T. P. (2013), <i>Sample Size Determination and Power</i>, Wiley.</li>'
        '<li>행 삭제(listwise deletion)는 결측이 완전 무작위(MCAR)일 때만 무편향입니다. '
        "결측이 많거나 무작위가 아니라면 모델링 단계에서 '대치(imputation)'를 권장합니다.</li>"
        '</ul></section>'
    )

    print()
    if save_report:
        if report_path is None:
            report_path = f"{dataset_name}_qtcheck_report.html"
        _save_report(report, report_path,
                     title=f"데이터 품질 점검 보고서 — {dataset_name}")
        print(_green(_bold(f"📝 품질 점검 HTML 보고서가 저장되었습니다: {report_path}")))

    _banner("🎉  데이터 품질 점검이 완료되었습니다.")

    return df
