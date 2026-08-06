# -*- coding: utf-8 -*-
"""제출 코드 대조 모듈

수업에서 배포한 원본 모듈(my_logit, my_ols 등)을 학생이 직접 타이핑해 제출했을 때,
제출 파일이 원본과 어디에서 어떻게 갈라지는지 찾아 준다.

대조 기준이 되는 원본은 다음 순서로 찾는다.

    1) `source_dir` 인자 또는 `HOSSAM_CHECKER_SRC` 환경변수가 가리키는 폴더
       -> 강사가 `helpers` 를 고치는 중일 때 재배포 없이 즉시 반영된다.
    2) 설치된 `hossam` 패키지 안의 같은 이름 모듈
       -> `helpers` 의 내용이 배포 시점에 패키지로 동기화되므로, 학생이
          `pip install -U hossam` 하면 최신 원본이 기준이 된다.
    3) 패키지에 동봉된 지문 파일(`_fingerprints/*.py.json`)
       -> 원본 모듈을 배포에서 빼는 경우를 위한 대비책.

보고서에는 **제출한 코드만** 실린다. 어느 함수의 어느 줄이 갈라지는지를 학생
본인의 코드로 짚어 주되, 그 자리에 들어가야 할 원본 코드는 싣지 않는다. 따라서
학생은 고쳐야 할 지점을 정확히 알면서도 정답을 받아 적을 수는 없다.
(3)번 경로는 해시만 담기므로 원본 복원 자체가 불가능하다.

대조는 세 가지를 본다.

    1) 시그니처 · 기본값   `backward=False` 를 `backward=True` 로 적는 유형의 사고.
                          호출부에서 인자를 넘기지 않는 경우 조용히 결과가 달라지므로
                          가장 먼저, 가장 크게 보고한다.
    2) 본문 구조          주석 · 문서화 문자열 · 공백 · 따옴표 종류를 모두 지우고
                          AST 로 정규화한 뒤 비교한다. 손으로 옮겨 적은 코드는 표기가
                          제각각이므로, 정규화 없이는 diff 가 의미를 갖지 못한다.
    3) 임포트             원본이 가져오는 이름이 제출에 빠져 있는지 확인한다.

본문 차이는 다시 두 가지로 나뉜다.

    구조 차이   코드의 모양 자체가 다르다. 실행 결과가 달라질 가능성이 높다.
    문자열 차이 코드 모양은 같고 문자열 내용만 다르다. 대개 표기 문제지만,
                딕셔너리 키나 비교 대상 문자열이면 실행에 영향을 준다.

사용 방법 (학생)

    from hossam import code_checker

    r = code_checker.diff("my_logit", "1.py")   # 대조 결과를 보고서로 출력
    r.show("fit_pipeline")                        # 특정 함수만 자세히 보기
    r.defaults                                    # 기본값 불일치 표 (DataFrame)
    r.functions                                   # 함수별 판정 표 (DataFrame)

노트북에서는 HTML 보고서로, 그 밖에서는 글자 보고서로 자동 전환된다. 형식을
직접 고르거나 파일로 남기려면 다음을 쓴다.

    r.report("markdown")                          # 마크다운으로 출력
    open("결과.md", "w").write(r.to_markdown())    # 파일로 저장
    open("결과.html", "w").write(r.to_html())

사용 방법 (강사)

    # 아직 배포되지 않은 helpers 의 최신 내용을 기준으로 점검
    code_checker.diff("my_logit", "1.py", source_dir="./helpers")

    # 매번 인자를 넘기기 번거로우면 환경변수로 지정한다
    os.environ["HOSSAM_CHECKER_SRC"] = "./helpers"

    # 원본 모듈을 배포에서 빼는 경우에만 필요한 지문 생성
    code_checker.build("./helpers")
"""

import io
import os
import ast
import copy
import json
import html
import keyword
import builtins
import unicodedata
import hashlib
import tokenize
import importlib.util
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

from pandas import DataFrame

from ._config import PACKAGE_NAME

# -------------------------------------------------------------
# 상수
# -------------------------------------------------------------
SCHEMA_VERSION = 2                          # 지문 파일 형식 버전
MODULE_DIR = Path(__file__).resolve().parent
FINGERPRINT_DIR = MODULE_DIR / "_fingerprints"      # 동봉되는 지문 보관 폴더
SOURCE_ENV = f"{PACKAGE_NAME.upper()}_CHECKER_SRC"  # 원본 폴더 지정 환경변수 (강사용)

# 문자열 상수를 가릴 때 사용하는 대체값
# -> 코드 모양은 같고 문자열만 다른 경우를 구분해 내기 위한 장치
_STR_MASK = "\x00str"

# 본문 불일치 유형
_KIND_STRUCTURE = "코드가 다름"
_KIND_STRING = "문자열이 다름"
_KIND_EXTRA = "원본에 없는 코드"
_KIND_MISSING = "빠진 코드"

# 불일치 유형별 설명과 표시 색상 (실행에 영향을 주는 순서대로)
_KIND_INFO = {
    _KIND_STRUCTURE: ("코드의 모양 자체가 다릅니다. 실행 결과가 달라질 수 있습니다.", "#d93025"),
    _KIND_MISSING: ("원본에 있는 코드가 빠졌습니다.", "#d93025"),
    _KIND_EXTRA: ("원본에 없는 코드가 들어 있습니다.", "#d93025"),
    _KIND_STRING: ("코드 모양은 같고 문자열 내용만 다릅니다. "
                   "딕셔너리 키나 비교 대상 문자열이면 실행에 영향을 줍니다.", "#e8710a"),
}

# 한 곳에서 보여 줄 코드의 최대 줄 수 (그 이상은 줄여서 표시한다)
_SNIPPET_LIMIT = 8

# 문법 강조 색상 (밝은 테마용, 어두운 테마용)
# -> 노트북 테마를 가리지 않도록 두 벌을 준비해 CSS 로 전환한다
_SYNTAX = {
    "kw": ("#a626a4", "#d886e0"),       # 키워드
    "str": ("#1a7f37", "#7ee787"),      # 문자열
    "num": ("#0550ae", "#79c0ff"),      # 숫자
    "com": ("#6e7781", "#9198a1"),      # 주석
    "bif": ("#953800", "#ffa657"),      # 내장 함수 · 상수
    "fn": ("#0550ae", "#d2a8ff"),       # 호출되는 이름
}

# 문법 강조용 CSS 클래스 이름 앞에 붙이는 말 (다른 출력과 섞이지 않게 한다)
_CSS_PREFIX = "hcc"

# 보고서 맨 앞에 붙이는 베타 안내
_BETA_TITLE = "베타 기능입니다"
_BETA_NOTICE = ("코드를 실행해 보고 판단하는 것이 아니라 구조를 견주어 보는 방식이라, "
                "실제로는 문제가 없는 곳을 짚거나 문제가 있는 곳을 놓칠 수 있습니다. "
                "결과는 참고용으로만 사용하고, 최종 판단은 직접 확인해 주세요.")
_BETA_COLOR = "#e8710a"

# 문자열 차이를 빼고 보여 줄 때 덧붙이는 안내
_HIDDEN_HINT = ("문자열 내용만 다른 곳 {n}군데는 빼고 보여 줍니다. "
                "대부분 출력 문구의 표기 차이라 실행에 영향이 없지만, "
                "딕셔너리 키나 비교 대상 문자열이면 문제가 될 수 있습니다. "
                "함께 보려면 force=True 를 주세요.")


# -------------------------------------------------------------
# 내부 유틸리티
# -------------------------------------------------------------
def _hash(text):
    """문자열을 짧은 해시로 변환한다.

    지문에는 원본 코드가 아니라 이 해시만 담긴다.

    Args:
        text (str): 해시를 계산할 문자열.

    Returns:
        str: 16자리 16진수 해시.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _StripDocstring(ast.NodeTransformer):
    """모듈 · 클래스 · 함수의 문서화 문자열을 제거하는 변환기.

    문서화 문자열은 실행에 영향을 주지 않으면서 사람마다 크게 달라지므로,
    비교 전에 제거해야 실질적인 차이만 남는다.
    """

    def _strip(self, node):
        self.generic_visit(node)

        body = node.body
        is_doc = (body
                  and isinstance(body[0], ast.Expr)
                  and isinstance(body[0].value, ast.Constant)
                  and isinstance(body[0].value.value, str))

        if is_doc:
            rest = body[1:]

            # 본문이 문서화 문자열뿐이었다면 pass 로 채운다 (구문 오류 방지)
            if not rest and not isinstance(node, ast.Module):
                rest = [ast.copy_location(ast.Pass(), body[0])]

            node.body = rest

        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


class _MaskStrings(ast.NodeTransformer):
    """모든 문자열 상수를 동일한 값으로 치환하는 변환기.

    치환 후에도 코드가 같다면 '문자열 내용만 다른 것'으로 판정할 수 있다.
    f-string 안의 고정 문구도 함께 가려지지만, 중괄호 안의 식은 그대로 남는다.
    """

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=_STR_MASK), node)

        return node


def _normalize(node):
    """AST 노드를 정규화된 소스 문자열로 되돌린다.

    `ast.unparse` 는 들여쓰기 · 따옴표 종류 · 줄바꿈 위치 · 괄호 유무를 모두
    표준형으로 통일하므로, 손으로 옮겨 적으면서 생긴 표기 차이가 사라진다.

    Args:
        node: 변환할 AST 노드.

    Returns:
        str: 정규화된 소스 문자열.
    """
    return ast.unparse(node)


def _shape(node):
    """문자열 상수를 가린 뒤 정규화한 소스 문자열을 돌려준다.

    Args:
        node: 변환할 AST 노드.

    Returns:
        str: 문자열 내용이 지워진 정규화 소스 문자열.
    """
    return ast.unparse(_MaskStrings().visit(copy.deepcopy(node)))


def _read_params(args):
    """함수의 인자 목록에서 이름 · 종류 · 기본값을 추출한다.

    위치 인자와 키워드 전용 인자를 구분해 담는다. 기본값은 소스 형태의
    문자열(`'False'`, `'10.0'`, `"'standard'"`)로 보관한다.

    Args:
        args (ast.arguments): 함수 정의의 인자 노드.

    Returns:
        list: `{'name', 'kind', 'default'}` 딕셔너리의 목록. 선언된 순서를 지킨다.
    """
    params = []

    # --- 1) 위치 인자 (기본값은 뒤에서부터 채워진다) ---
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    offset = len(positional) - len(defaults)

    for i, a in enumerate(positional):
        default = defaults[i - offset] if i >= offset else None
        params.append({
            "name": a.arg,
            "kind": "위치",
            "default": _normalize(default) if default is not None else None,
        })

    # --- 2) 가변 위치 인자 ---
    if args.vararg:
        params.append({"name": args.vararg.arg, "kind": "*args", "default": None})

    # --- 3) 키워드 전용 인자 (기본값이 없으면 None 이 들어 있다) ---
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        params.append({
            "name": a.arg,
            "kind": "키워드전용",
            "default": _normalize(d) if d is not None else None,
        })

    # --- 4) 가변 키워드 인자 ---
    if args.kwarg:
        params.append({"name": args.kwarg.arg, "kind": "**kwargs", "default": None})

    return params


def _blocks(node):
    """복합문이 품고 있는 하위 블록의 문장 목록을 돌려준다.

    `for` · `if` · `while` · `with` · `try` 처럼 몸통을 가진 문장을 그 안까지
    파고들기 위한 준비 작업이다.

    Args:
        node: 검사할 AST 노드.

    Returns:
        list: 하위 블록(문장 목록)의 목록. 복합문이 아니면 빈 목록.
    """
    found = []

    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)

        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            found.append(value)

    # try 문의 except 절은 별도의 목록에 담겨 있다
    for handler in getattr(node, "handlers", None) or []:
        if handler.body:
            found.append(handler.body)

    return found


def _header(node):
    """복합문에서 몸통을 걷어낸 머리 부분만 남긴 사본을 만든다.

    `for x in xnames:` 처럼 조건과 선언만 남기므로, 몸통 안의 차이와 머리 자체의
    차이를 따로 볼 수 있다.

    Args:
        node: 복합문 AST 노드.

    Returns:
        ast.stmt: 몸통이 `pass` 로 바뀐 사본.
    """
    clone = copy.deepcopy(node)

    for field in ("body", "orelse", "finalbody"):
        value = getattr(clone, field, None)

        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            setattr(clone, field, [ast.copy_location(ast.Pass(), value[0])])

    for handler in getattr(clone, "handlers", None) or []:
        if handler.body:
            handler.body = [ast.copy_location(ast.Pass(), handler.body[0])]

    return clone


def _read_statements(node):
    """함수 본문을 문장 단위로 쪼개 해시 목록을 만든다.

    복합문은 머리와 몸통을 나눠 담는다. 그래야 `for` 문 안쪽 한 줄만 다를 때
    루프 전체가 아니라 그 한 줄을 짚어 줄 수 있다.

    문장마다 두 가지 해시를 남긴다.
        hash  : 문자열 내용까지 포함한 해시
        shape : 문자열 내용을 가린 해시 (코드 모양만 반영)

    Args:
        node: 함수 정의 AST 노드. 문서화 문자열은 이미 제거된 상태여야 한다.

    Returns:
        list: `{'line', 'end', 'depth', 'hash', 'shape'}` 딕셔너리의 목록.
            `line` 과 `end` 는 문장이 차지하는 줄 범위로, 제출 코드를 그대로
            보여 줄 때 사용한다.
    """
    statements = []

    def _walk(body, depth):
        for st in body:
            line = getattr(st, "lineno", 0)
            end = getattr(st, "end_lineno", None) or line
            blocks = _blocks(st)

            # 단순 문장은 통째로 하나의 항목이 된다
            if not blocks:
                statements.append({
                    "line": line, "end": end, "depth": depth,
                    "hash": _hash(_normalize(st)), "shape": _hash(_shape(st)),
                })
                continue

            # 복합문은 머리만 항목으로 남기고 몸통은 한 단계 안으로 들어간다
            head = _header(st)
            head_end = max(line, min(s[0].lineno for s in blocks if s) - 1)

            statements.append({
                "line": line, "end": head_end, "depth": depth,
                "hash": _hash(_normalize(head)), "shape": _hash(_shape(head)),
            })

            for block in blocks:
                _walk(block, depth + 1)

    _walk(node.body, 0)

    return statements


def _read_imports(tree):
    """모듈이 가져오는 이름의 목록을 만든다.

    `import numpy as np` 는 `numpy as np`, `from pandas import Series` 는
    `pandas.Series` 형태로 기록한다. 원본이 쓰는 이름이 제출에 빠져 있으면
    해당 이름을 사용하는 순간 `NameError` 가 나므로 함께 대조한다.

    Args:
        tree (ast.Module): 모듈 AST.

    Returns:
        list: 정렬된 임포트 표기 목록.
    """
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(f"{a.name} as {a.asname}" if a.asname else a.name)
        elif isinstance(node, ast.ImportFrom):
            # 상대 임포트(`from . import my_plot`)는 점으로 단계를 표기한다
            base = "." * (node.level or 0) + (node.module or "")
            for a in node.names:
                label = f"{base}.{a.name}" if node.module else f"{base}{a.name}"
                names.add(f"{label} as {a.asname}" if a.asname else label)

    return sorted(names)


# -------------------------------------------------------------
# 지문 생성
# -------------------------------------------------------------
def analyze_source(source, module=None):
    """파이썬 소스코드를 읽어 지문 딕셔너리를 만든다.

    소스코드를 실행하지 않고 구문만 해석하므로, 임포트가 깨져 있거나 실행에
    부작용이 있는 파일도 안전하게 분석할 수 있다.

    Args:
        source (str): 분석할 파이썬 소스코드.
        module (str): 지문에 기록할 모듈 이름 (기본값: None).

    Returns:
        dict: 지문 딕셔너리. 함수별 시그니처 · 해시 · 문장 해시를 담는다.

    Raises:
        SyntaxError: 소스코드에 구문 오류가 있는 경우.
    """
    # --- 1) 구문 해석 후 문서화 문자열 제거 ---
    tree = ast.parse(source)
    tree = _StripDocstring().visit(tree)
    ast.fix_missing_locations(tree)

    # --- 2) 최상위 함수와 클래스 메서드를 모두 수집 ---
    functions = {}
    targets = []

    for order, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            targets.append((node.name, node, order))
        elif isinstance(node, ast.ClassDef):
            # 클래스 메서드는 'ClassName.method' 이름으로 기록한다
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    targets.append((f"{node.name}.{m.name}", m, order))

    # --- 3) 함수마다 시그니처 · 전체 해시 · 문장 해시를 기록 ---
    for name, node, order in targets:
        functions[name] = {
            "order": order,
            "line": getattr(node, "lineno", 0),
            "code": _hash(_normalize(node)),        # 시그니처를 포함한 전체 해시
            "shape": _hash(_shape(node)),           # 문자열을 가린 전체 해시
            "params": _read_params(node.args),
            "statements": _read_statements(node),
        }

    return {
        "schema": SCHEMA_VERSION,
        "module": module,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "imports": _read_imports(tree),
        "functions": functions,
    }


def analyze_file(path, module=None):
    """파이썬 파일을 읽어 지문 딕셔너리를 만든다.

    Args:
        path (str): 분석할 파이썬 파일 경로.
        module (str): 지문에 기록할 모듈 이름. None 이면 파일명을 사용한다 (기본값: None).

    Returns:
        dict: 지문 딕셔너리.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    return analyze_source(p.read_text(encoding="utf-8"), module=module or p.stem)


def build(src, out=None, modules=None, verbose=True):
    """원본 폴더의 모듈들을 훑어 지문 파일을 생성한다.

    `helpers` 의 모듈이 패키지로 동기화되어 배포된다면 이 단계는 필요하지 않다.
    설치된 모듈 자체가 대조 기준이 되기 때문이다. 원본 모듈을 배포에서 빼야 하는
    경우에만 사용한다. 생성되는 것은 해시와 시그니처뿐이라 원본 코드는 담기지 않는다.

    Args:
        src (str): 원본 모듈이 들어 있는 폴더 경로 (예: './helpers').
        out (str): 지문을 저장할 폴더. None 이면 패키지 안의 `_fingerprints` (기본값: None).
        modules (list): 지문을 만들 모듈 이름 목록. None 이면 `my_*.py` 전체 (기본값: None).
        verbose (bool): 진행 내역 출력 여부 (기본값: True).

    Returns:
        list: 생성된 지문 파일 경로의 목록.

    Raises:
        NotADirectoryError: 원본 폴더가 없는 경우.
    """
    # --- 1) 원본 폴더 확인 ---
    src_dir = Path(src)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"원본 폴더를 찾을 수 없습니다: {src}")

    out_dir = Path(out) if out else FINGERPRINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 2) 대상 모듈 결정 ---
    if modules:
        files = [src_dir / f"{m}.py" for m in modules]
    else:
        files = sorted(src_dir.glob("my_*.py"))

    # --- 3) 모듈마다 지문 생성 ---
    created = []

    for f in files:
        if not f.exists():
            print(f"⚠ 건너뜀 (파일 없음): {f}")
            continue

        try:
            fingerprint = analyze_file(f, module=f.stem)
        except SyntaxError as e:
            print(f"⚠ 건너뜀 (구문 오류): {f} → {e.lineno}줄 {e.msg}")
            continue

        target = out_dir / f"{f.stem}.json"
        target.write_text(
            json.dumps(fingerprint, ensure_ascii=False, indent=1), encoding="utf-8")
        created.append(str(target))

        if verbose:
            print(f"✓ {f.stem:20s} 함수 {len(fingerprint['functions']):3d}개 → {target.name}")

    if verbose:
        print(f"\n지문 {len(created)}개 생성 완료 : {out_dir}")

    return created


# -------------------------------------------------------------
# 지문 조회
# -------------------------------------------------------------
def _source_dir(source_dir=None):
    """원본 폴더를 실시간으로 참조할지 결정한다.

    강사가 `helpers` 를 수정하는 중이라면 지문을 다시 만들지 않고도 최신 내용으로
    대조할 수 있어야 한다. 인자나 환경변수로 원본 폴더가 지정되면 그 자리에서
    지문을 만들어 쓰고, 지정되지 않으면 패키지에 동봉된 지문을 쓴다.

    Args:
        source_dir (str): 원본 폴더 경로 (기본값: None → 환경변수 확인).

    Returns:
        Path: 사용할 원본 폴더. 사용하지 않으면 None.
    """
    path = source_dir or os.environ.get(SOURCE_ENV)

    if not path:
        return None

    p = Path(path)

    return p if p.is_dir() else None


def _installed_module_path(module):
    """설치된 `hossam` 패키지 안에서 원본 모듈의 파일 경로를 찾는다.

    `helpers` 의 내용은 배포 시점에 패키지로 동기화되므로, 설치된 모듈이 곧
    원본이 된다. 임포트하지 않고 위치만 조회하므로 무거운 의존성이 딸려
    올라오지 않는다.

    Args:
        module (str): 원본 모듈 이름 (예: 'my_logit').

    Returns:
        Path: 모듈 파일 경로. 찾지 못하면 None.
    """
    # 패키지 폴더에서 직접 찾는다 (개발 모드 · 설치 모드 모두 동일하게 동작)
    path = MODULE_DIR / f"{module}.py"

    if path.exists():
        return path

    # 다른 위치에 설치된 경우를 대비해 임포트 시스템에도 물어본다
    try:
        spec = importlib.util.find_spec(f"{__package__}.{module}")
    except (ImportError, AttributeError, ValueError):
        return None

    if spec and spec.origin and spec.origin.endswith(".py"):
        return Path(spec.origin)

    return None


def load_fingerprint(module, source_dir=None):
    """대조 기준이 되는 원본의 지문을 불러온다.

    원본은 `원본 폴더 → 설치된 패키지 모듈 → 동봉 지문` 순서로 찾는다.
    앞의 두 경로는 소스에서 그 자리에 지문을 만들어 쓰므로, `helpers` 를 고치고
    배포하면 별도의 지문 생성 없이 그대로 반영된다.

    Args:
        module (str): 원본 모듈 이름 (예: 'my_logit'). 'hossam.my_logit' 처럼
            패키지 이름이 붙어 있어도 된다.
        source_dir (str): 원본 폴더 경로. 지정하면 이 폴더를 최우선으로 참조한다
            (기본값: None → `HOSSAM_CHECKER_SRC` 환경변수 확인).

    Returns:
        dict: 지문 딕셔너리. 어느 경로에서 왔는지가 `origin` 키에 담긴다.

    Raises:
        FileNotFoundError: 어느 경로에서도 원본을 찾지 못한 경우.
    """
    module = module.split(".")[-1]      # 'hossam.my_logit' → 'my_logit'

    # --- 1) 원본 폴더 (강사가 helpers 를 고치는 중일 때) ---
    live = _source_dir(source_dir)

    if live and (live / f"{module}.py").exists():
        target = live / f"{module}.py"
        fingerprint = analyze_file(target, module=module)
        fingerprint["origin"] = f"원본 폴더 {target}"
        return fingerprint

    # --- 2) 설치된 패키지 안의 원본 모듈 (배포 시 helpers 와 동기화된다) ---
    installed = _installed_module_path(module)

    if installed:
        fingerprint = analyze_file(installed, module=module)
        fingerprint["origin"] = f"설치된 {PACKAGE_NAME} {_package_version()} 모듈"

        # 동봉 지문이 함께 있는데 내용이 어긋나면 동기화가 밀린 것이다
        _warn_if_stale(module, fingerprint)

        return fingerprint

    # --- 3) 동봉된 지문 (원본 모듈을 배포에서 뺀 경우) ---
    path = FINGERPRINT_DIR / f"{module}.json"

    if not path.exists():
        available = list_modules()
        raise FileNotFoundError(
            f"'{module}' 의 원본을 찾을 수 없습니다.\n"
            f"대조할 수 있는 모듈: {', '.join(available) if available else '없음'}")

    fingerprint = json.loads(path.read_text(encoding="utf-8"))
    fingerprint["origin"] = f"동봉 지문 {fingerprint.get('generated_at', '')[:10]} 기준"

    return fingerprint


def _package_version():
    """`hossam` 패키지의 버전 문자열을 돌려준다.

    패키지가 정한 값(`hossam.__version__`)을 그대로 쓴다. 소스를 직접 참조하는
    개발 모드에서는 'develop' 이 된다. 순환 임포트를 피하려고 함수 안에서
    가져온다.

    Returns:
        str: 버전 문자열. 확인할 수 없으면 'develop'.
    """
    try:
        from . import __version__
        return __version__
    except Exception:
        return "develop"


def _warn_if_stale(module, fingerprint):
    """설치된 모듈과 동봉 지문이 어긋나면 경고한다.

    두 가지가 함께 배포되는 동안에는 어느 쪽이 최신인지 헷갈릴 수 있다.
    내용이 다르면 동기화가 밀렸다는 뜻이므로 알려 준다.

    Args:
        module (str): 원본 모듈 이름.
        fingerprint (dict): 설치된 모듈에서 만든 지문.
    """
    path = FINGERPRINT_DIR / f"{module}.json"

    if not path.exists():
        return

    try:
        shipped = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return

    live_codes = {k: v["code"] for k, v in fingerprint["functions"].items()}
    ship_codes = {k: v["code"] for k, v in shipped.get("functions", {}).items()}

    if live_codes != ship_codes:
        print(f"⚠ {PACKAGE_NAME}/{module}.py 와 동봉 지문의 내용이 서로 다릅니다.\n"
              f"  설치된 모듈을 기준으로 대조합니다. "
              f"(지문 갱신: python -m {PACKAGE_NAME}.code_checker build --src ./helpers)")


def list_modules():
    """대조할 수 있는 모듈 이름의 목록을 돌려준다.

    설치된 패키지의 `my_*.py` 모듈과 동봉된 지문을 합쳐서 돌려준다.

    Returns:
        list: 대조 가능한 모듈 이름 목록.
    """
    names = {p.stem for p in MODULE_DIR.glob("my_*.py")}
    names.discard(Path(__file__).stem)      # 대조 도구 자신은 제외

    if FINGERPRINT_DIR.is_dir():
        names.update(p.stem for p in FINGERPRINT_DIR.glob("*.json"))

    return sorted(names)


# -------------------------------------------------------------
# 보고서 서식
# -------------------------------------------------------------
def _in_notebook():
    """주피터 노트북에서 실행 중인지 확인한다.

    Returns:
        bool: 노트북이면 True.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()

        if shell is None:
            return False

        # 주피터 커널(노트북 · VSCode · 랩)은 ZMQInteractiveShell 을 쓴다
        if type(shell).__name__ == "ZMQInteractiveShell":
            return True

        return "IPKernelApp" in getattr(shell, "config", {})
    except Exception:
        return False


class _Progress:
    """대조 진행 상황을 알려 주는 진행률 표시줄.

    `tqdm` 이 있으면 노트북에서는 그림 막대로, 터미널에서는 글자 막대로 보여 준다.
    없거나 표시가 꺼져 있으면 아무 일도 하지 않는다.
    """

    def __init__(self, total, enabled=True):
        self.bar = None

        if not enabled:
            return

        try:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=total, leave=False,
                            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}")
        except Exception:
            self.bar = None

    def step(self, desc=None, count=1):
        """막대를 앞으로 옮긴다.

        Args:
            desc (str): 현재 단계 설명 (기본값: None → 그대로 둔다).
            count (int): 나아갈 칸 수 (기본값: 1).
        """
        if self.bar is None:
            return

        if desc:
            self.bar.set_description_str(desc)

        self.bar.update(count)

    def grow(self, count):
        """전체 칸 수를 늘린다.

        대조할 함수의 수는 원본을 읽어 봐야 알 수 있으므로, 막대를 만든 뒤에
        늘릴 수 있어야 한다.

        Args:
            count (int): 더할 칸 수.
        """
        if self.bar is not None:
            self.bar.total += count
            self.bar.refresh()

    def close(self):
        """막대를 닫는다."""
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def _wrap(text, width):
    """글자의 화면 폭을 헤아려 문장을 접는다.

    한글·한자·가나는 터미널에서 두 칸을 차지하므로, 글자 수로 접으면 줄이 넘친다.

    Args:
        text (str): 접을 문장.
        width (int): 한 줄의 최대 표시 폭 (칸 수).

    Returns:
        list: 접힌 줄의 목록.
    """
    def _cells(s):
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

    lines, current, used = [], [], 0

    for word in text.split(" "):
        size = _cells(word)

        # 첫 낱말이 아니면 사이의 빈칸도 폭에 넣는다
        if current and used + 1 + size > width:
            lines.append(" ".join(current))
            current, used = [word], size
        else:
            used += (1 if current else 0) + size
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines


def _token_kind(token, tokens, i):
    """토큰 하나가 어떤 색으로 칠해질지 갈래를 정한다.

    Args:
        token: 검사할 토큰.
        tokens (list): 전체 토큰 목록. 다음 토큰을 살펴보기 위해 받는다.
        i (int): `tokens` 안에서 `token` 의 위치.

    Returns:
        str: `_SYNTAX` 의 갈래 이름. 색을 입히지 않으면 None.
    """
    name = tokenize.tok_name.get(token.type, "")

    if name == "COMMENT":
        return "com"

    # 파이썬 3.12 부터 f-string 은 FSTRING_START/MIDDLE/END 로 쪼개진다
    if name == "STRING" or name.startswith("FSTRING"):
        return "str"

    if name == "NUMBER":
        return "num"

    if name == "NAME":
        if keyword.iskeyword(token.string) or keyword.issoftkeyword(token.string):
            return "kw"

        if hasattr(builtins, token.string):
            return "bif"

        # 바로 뒤에 여는 괄호가 붙어 있으면 함수 호출로 본다
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if nxt is not None and nxt.string == "(" and nxt.start == token.end:
            return "fn"

    return None


def _colorize(code):
    """코드 문자열을 토큰 단위로 나눠 줄마다 색칠할 조각을 만든다.

    잘라 온 코드 조각은 괄호가 열린 채로 끝나는 등 문법적으로 온전하지 않을 수
    있다. 그래도 오류 직전까지 읽어 낸 토큰은 쓸 수 있으므로 예외를 삼키고
    거기까지의 결과를 사용한다.

    Args:
        code (str): 색칠할 코드 문자열.

    Returns:
        list: 줄마다 `(글자, 갈래)` 튜플의 목록. 갈래가 None 이면 색을 입히지 않는다.
    """
    rows = code.split("\n")
    tokens = []

    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            tokens.append(token)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        pass

    # --- 1) 토큰이 차지하는 (행, 칸) 범위에 갈래를 기록한다 ---
    marks = {}

    for i, token in enumerate(tokens):
        kind = _token_kind(token, tokens, i)

        if not kind:
            continue

        (srow, scol), (erow, ecol) = token.start, token.end

        # 여러 줄에 걸친 토큰(삼중 따옴표 문자열 등)은 줄마다 나눠 기록한다
        for r in range(srow, min(erow, len(rows)) + 1):
            a = scol if r == srow else 0
            b = ecol if r == erow else len(rows[r - 1])

            if b > a:
                marks.setdefault(r, []).append((a, b, kind))

    # --- 2) 기록한 범위를 따라 줄마다 조각을 잘라 낸다 ---
    result = []

    for n, text in enumerate(rows, start=1):
        pieces = []
        pos = 0

        for a, b, kind in sorted(marks.get(n, [])):
            if a < pos:            # 범위가 겹치면 앞선 것을 살린다
                continue

            if a > pos:
                pieces.append((text[pos:a], None))

            pieces.append((text[a:b], kind))
            pos = b

        if pos < len(text):
            pieces.append((text[pos:], None))

        result.append(pieces)

    return result


def _syntax_css():
    """문법 강조에 쓸 CSS 를 만든다.

    노트북마다 어두운 테마를 알리는 방법이 달라, 운영체제 설정 ·
    주피터랩 · VSCode 세 가지를 모두 받아 준다.

    Returns:
        str: `<style>` 태그를 포함한 CSS 문자열.
    """
    def _rules(selector, index):
        return "".join(
            f"{selector} .{_CSS_PREFIX}-{k}{{color:{v[index]};}}"
            for k, v in _SYNTAX.items())

    dark = _rules("", 1)

    return ("<style>"
            + _rules("", 0)
            + f"@media (prefers-color-scheme:dark){{{dark}}}"
            + _rules("body[data-jp-theme-light='false']", 1)
            + _rules("body.vscode-dark", 1)
            + "</style>")


def _code_html(code_lines):
    """제출 코드 조각을 줄 번호와 문법 강조가 붙은 HTML 로 만든다.

    Args:
        code_lines (list): `_snippet` 이 돌려준 `(줄번호, 코드)` 목록.

    Returns:
        str: HTML 문자열.
    """
    # 줄임표 항목을 빼고 실제 코드만 모아 한 번에 색칠한다
    body = "\n".join(t for n, t in code_lines if n is not None)
    colored = _colorize(body)

    rows = []
    index = 0

    for n, text in code_lines:
        # 줄임표 항목은 코드가 아니므로 흐리게만 표시한다
        if n is None:
            inner = f'<span style="opacity:.55;">{html.escape(text)}</span>'
        else:
            pieces = colored[index] if index < len(colored) else [(text, None)]
            index += 1
            inner = "".join(
                html.escape(s) if not k else
                f'<span class="{_CSS_PREFIX}-{k}">{html.escape(s)}</span>'
                for s, k in pieces)

        rows.append(
            f'<div style="display:flex; gap:12px;">'
            f'<span style="opacity:.45; min-width:44px; text-align:right;'
            f' user-select:none;">{n if n else ""}</span>'
            f'<span style="white-space:pre;">{inner}</span></div>')

    return "".join(rows)


def _snippet(lines, start, end):
    """제출 파일에서 지정한 줄 범위의 코드를 잘라 온다.

    너무 길면 앞부분만 보여 주고 나머지는 줄임표로 줄인다.

    Args:
        lines (list): 제출 파일의 전체 줄 목록.
        start (int): 시작 줄 번호 (1부터).
        end (int): 끝 줄 번호.

    Returns:
        list: `(줄번호, 코드)` 튜플의 목록. 줄번호가 None 이면 줄임표 표시다.
    """
    if not start or start > len(lines):
        return []

    end = min(max(end or start, start), len(lines))
    picked = [(n, lines[n - 1].rstrip()) for n in range(start, end + 1)]

    # 들여쓰기를 공통으로 덜어 내 화면 폭을 아낀다
    body = [t for _, t in picked if t.strip()]
    indent = min((len(t) - len(t.lstrip()) for t in body), default=0)
    picked = [(n, t[indent:] if t.strip() else "") for n, t in picked]

    if len(picked) > _SNIPPET_LIMIT:
        hidden = len(picked) - _SNIPPET_LIMIT
        picked = picked[:_SNIPPET_LIMIT] + [(None, f"... ({hidden}줄 생략)")]

    return picked


class CompareResult:
    """제출 파일과 원본을 대조한 결과를 담는 객체.

    Attributes:
        module (str): 원본 모듈 이름.
        path (str): 제출 파일 경로.
        origin (str): 대조 기준으로 삼은 원본이 어디에서 왔는지.
        defaults (DataFrame): 기본값이 다른 파라미터 표.
        params (DataFrame): 이름 · 순서가 다른 파라미터 표.
        functions (DataFrame): 함수별 판정 표.
        imports (DataFrame): 임포트 불일치 표.
        details (dict): 함수별 본문 불일치 위치 목록.
        source (list): 제출 파일의 줄 목록. 문제 지점의 코드를 보여 줄 때 쓴다.
        force (bool): 문자열 내용만 다른 곳까지 담았는지 여부.
        suppressed (int): 문자열 차이라서 보고서에서 뺀 곳의 수.
    """

    def __init__(self, module, path, fingerprint, defaults, params,
                 functions, imports, details, source, force=True, suppressed=0):
        self.module = module
        self.path = path
        self.origin = fingerprint.get("origin", "알 수 없음")
        self.defaults = defaults
        self.params = params
        self.functions = functions
        self.imports = imports
        self.details = details
        self.source = source
        self.force = force
        self.suppressed = suppressed

    # ---------------------------------------------------------
    # 요약 수치
    # ---------------------------------------------------------
    @property
    def total(self):
        """int: 원본에 들어 있는 함수의 수."""
        return int((self.functions["판정"] != "원본에 없음").sum())

    @property
    def matched(self):
        """int: 원본과 일치하는 함수의 수."""
        return int((self.functions["판정"] == "일치").sum())

    @property
    def ok(self):
        """bool: 모든 항목이 원본과 일치하는지 여부."""
        return (self.defaults.empty and self.params.empty
                and self.imports.empty and self.matched == self.total)

    @property
    def problems(self):
        """int: 본문에서 발견된 불일치 지점의 총 개수."""
        return sum(len(v) for v in self.details.values())

    # ---------------------------------------------------------
    # 보고서 출력
    # ---------------------------------------------------------
    def report(self, format=None, functions=None):
        """대조 결과를 보고서로 출력한다.

        Args:
            format (str): 'html' · 'markdown' · 'text' 중 하나. None 이면
                노트북에서는 'html', 그 밖에서는 'text' 를 쓴다 (기본값: None).
            functions (list): 보고서에 담을 함수 이름 목록. None 이면 전체 (기본값: None).
        """
        format = format or ("html" if _in_notebook() else "text")

        if format == "html":
            from IPython.display import display, HTML
            display(HTML(self.to_html(functions)))
        elif format == "markdown":
            if _in_notebook():
                from IPython.display import display, Markdown
                display(Markdown(self.to_markdown(functions)))
            else:
                print(self.to_markdown(functions))
        else:
            print(self.to_text(functions))

    def show(self, name, format=None):
        """특정 함수의 불일치 내역만 자세히 출력한다.

        Args:
            name (str): 확인할 함수 이름.
            format (str): 출력 형식 (기본값: None → 자동).
        """
        self.report(format=format, functions=[name])

    # ---------------------------------------------------------
    # 보고서 구성
    # ---------------------------------------------------------
    def _sections(self, functions=None):
        """보고서에 담을 내용을 서식과 무관한 형태로 정리한다.

        HTML · 마크다운 · 글자 보고서가 모두 이 결과를 받아 서식만 입힌다.

        Args:
            functions (list): 보고서에 담을 함수 이름 목록 (기본값: None → 전체).

        Returns:
            dict: `signature`(시그니처 불일치), `bodies`(본문 불일치),
                `imports`(임포트 불일치), `missing`/`extra`(함수 구성) 을 담은 딕셔너리.
        """
        # --- 1) 대상 함수 추리기 ---
        keys = list(self.details.keys())
        names = set(functions) if functions else None

        if names is not None:
            keys = [k for k in keys if k in names]

        def _pick(df):
            return df[df["함수"].isin(names)] if names is not None and not df.empty else df

        defaults = _pick(self.defaults)
        params = _pick(self.params)

        # --- 2) 함수별 시그니처 불일치 ---
        signature = []

        for name in sorted(set(defaults["함수"]) | set(params["함수"])) if not (
                defaults.empty and params.empty) else []:
            signature.append({
                "함수": name,
                "기본값": defaults[defaults["함수"] == name].to_dict("records"),
                "파라미터": params[params["함수"] == name].to_dict("records"),
            })

        # --- 3) 함수별 본문 불일치 (제출 코드를 함께 담는다) ---
        bodies = []

        for name in keys:
            spots = []

            for item in self.details[name]:
                # 빠진 코드는 제출 파일에 보여 줄 것이 없다.
                # -> 뒤따르는 줄의 코드를 보여 주면 그 줄이 문제인 것처럼 읽히므로
                #    자리만 알려 주고 코드는 싣지 않는다.
                missing = item["kind"] == _KIND_MISSING

                spots.append({
                    "kind": item["kind"],
                    "line": item["line"],
                    "count": item.get("count", 1),
                    "code": [] if missing else _snippet(
                        self.source, item["line"], item["end"]),
                    "before": missing,
                })

            bodies.append({"함수": name, "지점": spots})

        # --- 4) 함수 구성 (전체 보고서일 때만 의미가 있다) ---
        missing, extra = [], []

        if names is None:
            missing = self.functions[
                self.functions["판정"] == "미작성"]["함수"].tolist()
            extra = self.functions[
                self.functions["판정"] == "원본에 없음"]["함수"].tolist()

        # --- 5) 이 보고서 범위에서의 요약 (특정 함수만 볼 때는 그 함수 기준) ---
        spots = sum(len(b["지점"]) for b in bodies)

        if names is None:
            summary = (f"불일치 {self.total - self.matched}개 함수 · {spots}곳",
                       f"함수 {self.total}개 중 {self.matched}개 일치")
            ok = self.ok
        else:
            ok = not (spots or signature)
            summary = (f"불일치 {spots}곳" if spots else "본문은 원본과 일치",
                       f"대상 함수 {len(names)}개")

        # 범례는 실제로 나온 갈래만 보여 준다
        kinds = []

        for b in bodies:
            for spot in b["지점"]:
                if spot["kind"] not in kinds:
                    kinds.append(spot["kind"])

        return {
            "kinds": [k for k in _KIND_INFO if k in kinds],
            "hidden": 0 if self.force else self.suppressed,
            "signature": signature,
            "bodies": bodies,
            "imports": self.imports.to_dict("records") if names is None else [],
            "missing": missing,
            "extra": extra,
            "partial": names is not None,
            "summary": summary,
            "ok": ok,
        }

    # ---------------------------------------------------------
    # HTML 보고서
    # ---------------------------------------------------------
    def to_html(self, functions=None):
        """대조 결과를 HTML 문자열로 만든다.

        Args:
            functions (list): 보고서에 담을 함수 이름 목록 (기본값: None → 전체).

        Returns:
            str: HTML 문자열. 파일로 저장하거나 `IPython.display.HTML` 로 표시한다.
        """
        s = self._sections(functions)
        e = html.escape

        # 배경색을 지정하지 않고 반투명 색만 얹어 밝은 테마와 어두운 테마에 모두 맞춘다
        box = ("border:1px solid rgba(128,128,128,.35); border-radius:8px; "
               "padding:14px 16px; margin:0 0 14px 0;")
        mono = ("font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; "
                "font-size:12.5px;")

        out = [_syntax_css(),
               f'<div style="{mono.replace("12.5px", "13.5px")} line-height:1.55; '
               f'max-width:1000px;">']

        # --- 0) 베타 안내 ---
        out.append(
            f'<div style="border:1px solid {_BETA_COLOR}55; border-left:4px solid'
            f' {_BETA_COLOR}; border-radius:8px; padding:11px 14px; margin:0 0 14px 0;'
            f' background:{_BETA_COLOR}14;">'
            f'<div style="font-weight:600; color:{_BETA_COLOR}; margin-bottom:3px;">'
            f'&#9888;&#65039; {e(_BETA_TITLE)}</div>'
            f'<div style="font-size:12.5px; opacity:.85;">{e(_BETA_NOTICE)}</div>'
            f'</div>')

        # --- 1) 머리말 ---
        title = f"{e(self.module)} 대조 결과"
        if s["partial"]:
            title += f" — {e(', '.join(functions))}"

        badge = "✅ 원본과 일치" if s["ok"] else s["summary"][0]
        color = "#1e8e3e" if s["ok"] else "#d93025"

        out.append(
            f'<div style="{box}">'
            f'<div style="font-size:17px; font-weight:600; margin-bottom:6px;">{title}</div>'
            f'<div style="opacity:.75; font-size:12.5px;">'
            f'제출 : {e(self.path)}<br>기준 : {e(self.origin)}</div>'
            f'<div style="margin-top:10px; font-weight:600; color:{color};">{badge}</div>'
            f'<div style="opacity:.75; font-size:12.5px; margin-top:4px;">'
            f'{e(s["summary"][1])}</div>'
            f'</div>')

        if s["ok"]:
            return "".join(out) + "</div>"

        # --- 2) 시그니처 불일치 ---
        if s["signature"]:
            rows = []

            for fn in s["signature"]:
                for d in fn["기본값"]:
                    rows.append(
                        f'<tr><td style="padding:5px 12px 5px 0;">{e(fn["함수"])}'
                        f'(<b>{e(d["파라미터"])}</b>)</td>'
                        f'<td style="padding:5px 12px 5px 0; color:#d93025;">'
                        f'제출 <b>{e(str(d["제출"]))}</b></td>'
                        f'<td style="padding:5px 0; opacity:.8;">'
                        f'원본 {e(str(d["원본"]))}</td></tr>')

                for p in fn["파라미터"]:
                    rows.append(
                        f'<tr><td style="padding:5px 12px 5px 0;">{e(fn["함수"])}</td>'
                        f'<td colspan="2" style="padding:5px 0;">'
                        f'{e(p["구분"])} : <b>{e(p["파라미터"])}</b></td></tr>')

            out.append(
                f'<div style="{box}">'
                f'<div style="font-weight:600; margin-bottom:4px;">시그니처 불일치</div>'
                f'<div style="opacity:.75; font-size:12.5px; margin-bottom:10px;">'
                f'호출할 때 인자를 넘기지 않으면 원본과 다르게 동작합니다.</div>'
                f'<table style="border-collapse:collapse;">{"".join(rows)}</table></div>')

        # --- 3) 본문 불일치 (제출한 코드를 그대로 보여 준다) ---
        for fn in s["bodies"]:
            spots = []

            for spot in fn["지점"]:
                note, tone = _KIND_INFO[spot["kind"]]
                where = (f'{spot["line"]}줄 앞' if spot["before"]
                         else f'{spot["line"]}줄') if spot["line"] else "위치 확인 필요"
                label = (f'{spot["kind"]} {spot["count"]}개'
                         if spot["count"] > 1 else spot["kind"])

                if spot["code"]:
                    code = _code_html(spot["code"])
                else:
                    count = f'{spot["count"]}개' if spot["count"] > 1 else "1개"
                    code = (f'<div style="opacity:.7;">이 자리에 있어야 할 문장 '
                            f'{count}가 제출 파일에 없습니다.</div>')

                spots.append(
                    f'<div style="border-left:3px solid {tone}; padding:2px 0 2px 12px;'
                    f' margin:12px 0 0 0;">'
                    f'<div style="font-size:12.5px; margin-bottom:6px;">'
                    f'<b style="color:{tone};">{e(label)}</b>'
                    f'<span style="opacity:.6;"> · {where}</span></div>'
                    f'<div style="{mono} background:rgba(128,128,128,.10);'
                    f' border-radius:5px; padding:8px 10px; overflow-x:auto;">{code}</div>'
                    f'</div>')

            out.append(
                f'<details style="{box}" open>'
                f'<summary style="font-weight:600; cursor:pointer;">'
                f'{e(fn["함수"])} <span style="opacity:.6; font-weight:400;">'
                f'— {len(fn["지점"])}곳</span></summary>'
                f'{"".join(spots)}</details>')

        # --- 4) 임포트 · 함수 구성 ---
        extras = []

        for r in s["imports"]:
            extras.append(f'<div>{e(r["구분"])} : <b>{e(r["이름"])}</b></div>')

        if s["missing"]:
            extras.append(f'<div>제출에 없는 함수 : <b>{e(", ".join(s["missing"]))}</b></div>')

        if s["extra"]:
            extras.append(f'<div>원본에 없는 함수 : <b>{e(", ".join(s["extra"]))}</b></div>')

        if extras:
            out.append(
                f'<div style="{box}">'
                f'<div style="font-weight:600; margin-bottom:8px;">그 밖의 불일치</div>'
                f'{"".join(extras)}</div>')

        # --- 5) 범례 ---
        legend = "".join(
            f'<div style="margin-top:4px;">'
            f'<b style="color:{_KIND_INFO[k][1]};">{e(k)}</b> '
            f'<span style="opacity:.75;">— {e(_KIND_INFO[k][0])}</span></div>'
            for k in s["kinds"])

        if s["hidden"]:
            legend += (f'<div style="margin-top:10px; opacity:.75;">'
                       f'{e(_HIDDEN_HINT.format(n=s["hidden"]))}</div>')

        if legend:
            out.append(f'<div style="{box} font-size:12.5px;">{legend}</div>')
        out.append("</div>")

        return "".join(out)

    # ---------------------------------------------------------
    # 마크다운 보고서
    # ---------------------------------------------------------
    def to_markdown(self, functions=None):
        """대조 결과를 마크다운 문자열로 만든다.

        Args:
            functions (list): 보고서에 담을 함수 이름 목록 (기본값: None → 전체).

        Returns:
            str: 마크다운 문자열.
        """
        s = self._sections(functions)
        out = []

        # --- 1) 머리말 ---
        title = f"{self.module} 대조 결과"
        if s["partial"]:
            title += f" — {', '.join(functions)}"

        out.append(f"## {title}\n")
        out.append(f"> ⚠️ **{_BETA_TITLE}**  ")
        out.append(f"> {_BETA_NOTICE}\n")
        out.append(f"- 제출 : `{self.path}`")
        out.append(f"- 기준 : {self.origin}")

        if s["ok"]:
            out.append(f"\n**✅ 원본과 일치합니다.** ({s['summary'][1]})\n")

            return "\n".join(out)

        out.append(f"\n**{s['summary'][0]}** ({s['summary'][1]})\n")

        # --- 2) 시그니처 불일치 ---
        if s["signature"]:
            out.append("### 시그니처 불일치\n")
            out.append("호출할 때 인자를 넘기지 않으면 원본과 다르게 동작합니다.\n")
            out.append("| 함수 | 파라미터 | 제출 | 원본 |")
            out.append("|---|---|---|---|")

            for fn in s["signature"]:
                for d in fn["기본값"]:
                    out.append(f"| {fn['함수']} | `{d['파라미터']}` | "
                               f"`{d['제출']}` | `{d['원본']}` |")

                for p in fn["파라미터"]:
                    out.append(f"| {fn['함수']} | `{p['파라미터']}` | "
                               f"{p['구분']} | |")

            out.append("")

        # --- 3) 본문 불일치 ---
        for fn in s["bodies"]:
            out.append(f"### {fn['함수']} — {len(fn['지점'])}곳\n")

            for spot in fn["지점"]:
                where = (f"{spot['line']}줄 앞" if spot["before"]
                         else f"{spot['line']}줄") if spot["line"] else "위치 확인 필요"
                label = (f"{spot['kind']} {spot['count']}개"
                         if spot["count"] > 1 else spot["kind"])
                out.append(f"**{label}** · {where}\n")

                if spot["code"]:
                    out.append("```python")
                    out.extend(t for _, t in spot["code"])
                    out.append("```\n")
                else:
                    n = f"{spot['count']}개" if spot["count"] > 1 else "1개"
                    out.append(f"> 이 자리에 있어야 할 문장 {n}가 제출 파일에 없습니다.\n")

        # --- 4) 임포트 · 함수 구성 ---
        extras = [f"- {r['구분']} : `{r['이름']}`" for r in s["imports"]]

        if s["missing"]:
            extras.append(f"- 제출에 없는 함수 : {', '.join(s['missing'])}")

        if s["extra"]:
            extras.append(f"- 원본에 없는 함수 : {', '.join(s['extra'])}")

        if extras:
            out.append("### 그 밖의 불일치\n")
            out.extend(extras)
            out.append("")

        # --- 5) 범례 ---
        if s["kinds"] or s["hidden"]:
            out.append("---\n")

        for k in s["kinds"]:
            out.append(f"- **{k}** — {_KIND_INFO[k][0]}")

        if s["hidden"]:
            out.append(f"\n> {_HIDDEN_HINT.format(n=s['hidden'])}")

        return "\n".join(out)

    # ---------------------------------------------------------
    # 글자 보고서
    # ---------------------------------------------------------
    def to_text(self, functions=None):
        """대조 결과를 터미널용 글자 보고서로 만든다.

        Args:
            functions (list): 보고서에 담을 함수 이름 목록 (기본값: None → 전체).

        Returns:
            str: 보고서 문자열.
        """
        s = self._sections(functions)
        bar = "─" * 72
        out = []

        # --- 1) 머리말 ---
        title = f"{self.module} 대조 결과"
        if s["partial"]:
            title += f" — {', '.join(functions)}"

        out.append(f"\n{bar}\n {title}\n{bar}")
        out.append(f" ⚠️  {_BETA_TITLE}")

        # 안내 문구가 길어 화면 폭에 맞춰 접는다
        for chunk in _wrap(_BETA_NOTICE, 66):
            out.append(f"    {chunk}")

        out.append(f"{bar}")
        out.append(f" 제출 : {self.path}")
        out.append(f" 기준 : {self.origin}")

        if s["ok"]:
            out.append(f"\n ✅ 원본과 일치합니다. ({s['summary'][1]})\n")

            return "\n".join(out)

        out.append(f"\n {s['summary'][0]}   ({s['summary'][1]})")

        # --- 2) 시그니처 불일치 ---
        if s["signature"]:
            out.append(f"\n{bar}\n 시그니처 불일치"
                       f"\n 호출할 때 인자를 넘기지 않으면 원본과 다르게 동작합니다.\n")

            for fn in s["signature"]:
                for d in fn["기본값"]:
                    out.append(f"   {fn['함수']}({d['파라미터']})")
                    out.append(f"       제출 {d['제출']}   /   원본 {d['원본']}")

                for p in fn["파라미터"]:
                    out.append(f"   {fn['함수']}")
                    out.append(f"       {p['구분']} : {p['파라미터']}")

        # --- 3) 본문 불일치 ---
        for fn in s["bodies"]:
            out.append(f"\n{bar}\n {fn['함수']} — {len(fn['지점'])}곳")

            for spot in fn["지점"]:
                where = (f"{spot['line']}줄 앞" if spot["before"]
                         else f"{spot['line']}줄") if spot["line"] else "위치 확인 필요"
                label = (f"{spot['kind']} {spot['count']}개"
                         if spot["count"] > 1 else spot["kind"])
                out.append(f"\n   [{label}] {where}")

                if spot["code"]:
                    for n, t in spot["code"]:
                        out.append(f"   {str(n) + ' |' if n else '   |':>8s} {t}")
                else:
                    n = f"{spot['count']}개" if spot["count"] > 1 else "1개"
                    out.append(f"       이 자리에 있어야 할 문장 {n}가 제출 파일에 없습니다.")

        # --- 4) 임포트 · 함수 구성 ---
        extras = [f"   {r['구분']} : {r['이름']}" for r in s["imports"]]

        if s["missing"]:
            extras.append(f"   제출에 없는 함수 : {', '.join(s['missing'])}")

        if s["extra"]:
            extras.append(f"   원본에 없는 함수 : {', '.join(s['extra'])}")

        if extras:
            out.append(f"\n{bar}\n 그 밖의 불일치\n")
            out.extend(extras)

        # --- 5) 범례 ---
        if s["kinds"] or s["hidden"]:
            out.append(f"\n{bar}")

        for k in s["kinds"]:
            for i, chunk in enumerate(_wrap(f"{k} — {_KIND_INFO[k][0]}", 70)):
                out.append(f" {chunk}" if i == 0 else f"   {chunk}")

        if s["hidden"]:
            out.append("")
            for chunk in _wrap(_HIDDEN_HINT.format(n=s["hidden"]), 70):
                out.append(f" {chunk}")

        out.append("")

        return "\n".join(out)

    # `_repr_html_` 은 일부러 두지 않는다.
    # -> 노트북에서 `diff(...)` 를 대입 없이 호출하면 함수 안에서 보고서를 한 번
    #    출력하고, 셀의 마지막 값이 된 이 객체를 주피터가 한 번 더 그린다.
    #    그러면 같은 보고서가 두 번 나오므로, 여기서는 짧은 한 줄만 돌려준다.
    #    보고서를 다시 보려면 `r.report()` 를 호출한다.
    def __repr__(self):
        state = "일치" if self.ok else f"{self.total - self.matched}개 함수 불일치"

        return (f"<CompareResult {self.module} ← {Path(self.path).name} : {state}"
                f" · 다시 보려면 .report()>")


def _compare_params(ref_params, sub_params, name, defaults, params):
    """한 함수의 시그니처를 대조해 결과를 누적한다.

    Args:
        ref_params (list): 원본 지문의 파라미터 목록.
        sub_params (list): 제출 파일의 파라미터 목록.
        name (str): 함수 이름.
        defaults (list): 기본값 불일치를 누적할 리스트.
        params (list): 파라미터 불일치를 누적할 리스트.

    Returns:
        bool: 시그니처가 완전히 일치하면 True.
    """
    ref_map = {p["name"]: p for p in ref_params}
    sub_map = {p["name"]: p for p in sub_params}

    # --- 1) 이름이 한쪽에만 있는 파라미터 ---
    only_ref = [n for n in ref_map if n not in sub_map]
    only_sub = [n for n in sub_map if n not in ref_map]

    for n in only_ref:
        params.append({"함수": name, "구분": "제출에 없는 인자", "파라미터": n})

    for n in only_sub:
        params.append({"함수": name, "구분": "원본에 없는 인자", "파라미터": n})

    # --- 2) 양쪽에 있는 파라미터의 기본값 비교 ---
    for n, ref in ref_map.items():
        sub = sub_map.get(n)

        if sub is None:
            continue

        if ref["default"] != sub["default"]:
            defaults.append({
                "함수": name,
                "파라미터": n,
                "원본": ref["default"] if ref["default"] is not None else "(없음)",
                "제출": sub["default"] if sub["default"] is not None else "(없음)",
            })

    # --- 3) 선언 순서 비교 (양쪽 공통 인자만 대상으로 한다) ---
    common_ref = [p["name"] for p in ref_params if p["name"] in sub_map]
    common_sub = [p["name"] for p in sub_params if p["name"] in ref_map]

    if common_ref != common_sub:
        params.append({
            "함수": name,
            "구분": "선언 순서 다름",
            "파라미터": " → ".join(common_sub),
        })

    return not (only_ref or only_sub) and common_ref == common_sub


def _compare_statements(ref_stmts, sub_stmts):
    """한 함수의 본문을 문장 단위로 정렬해 불일치 위치를 찾는다.

    문자열 내용을 가린 해시(shape)로 먼저 정렬한다. 그래야 문자열만 다른 문장이
    '같은 자리'로 짝지어져, 구조가 바뀐 곳과 표기만 바뀐 곳을 구분할 수 있다.

    Args:
        ref_stmts (list): 원본 지문의 문장 목록.
        sub_stmts (list): 제출 파일의 문장 목록.

    Returns:
        list: `{'line', 'end', 'kind'}` 형태의 불일치 항목 목록. `line` 과 `end`
            는 제출 파일에서 문제가 되는 줄 범위다. 원본에만 있던 문장이라
            제출 쪽에 대응하는 코드가 없으면 `line` 이 0 이 된다.
    """
    ref_shape = [s["shape"] for s in ref_stmts]
    sub_shape = [s["shape"] for s in sub_stmts]

    matcher = SequenceMatcher(None, ref_shape, sub_shape, autojunk=False)
    items = []

    def _anchor(index):
        """빠진 코드가 들어가야 할 자리의 줄 번호를 추정한다."""
        if index < len(sub_stmts):
            return sub_stmts[index]["line"]

        return sub_stmts[-1]["end"] if sub_stmts else 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # --- 1) 모양이 같은 구간: 문자열 내용까지 같은지 확인 ---
        if tag == "equal":
            for k in range(i2 - i1):
                if ref_stmts[i1 + k]["hash"] != sub_stmts[j1 + k]["hash"]:
                    st = sub_stmts[j1 + k]
                    items.append({"line": st["line"], "end": st["end"],
                                  "kind": _KIND_STRING})

        # --- 2) 모양이 다른 구간: 제출 코드를 그대로 짚어 준다 ---
        elif tag == "replace":
            for k in range(j1, j2):
                items.append({"line": sub_stmts[k]["line"], "end": sub_stmts[k]["end"],
                              "kind": _KIND_STRUCTURE})

            # 원본 문장이 더 많았다면 그만큼은 자리만 알려 준다
            for _ in range((i2 - i1) - (j2 - j1)):
                items.append({"line": _anchor(j2), "end": 0, "kind": _KIND_MISSING})

        # --- 3) 원본에만 있는 문장: 제출 쪽에 보여 줄 코드가 없다 ---
        elif tag == "delete":
            for _ in range(i1, i2):
                items.append({"line": _anchor(j1), "end": 0, "kind": _KIND_MISSING})

        # --- 4) 제출에만 있는 문장: 그대로 짚어 준다 ---
        elif tag == "insert":
            for k in range(j1, j2):
                items.append({"line": sub_stmts[k]["line"], "end": sub_stmts[k]["end"],
                              "kind": _KIND_EXTRA})

    # 파일에 나타나는 순서대로 정렬해야 학생이 위에서부터 훑어보기 편하다
    items.sort(key=lambda x: (x["line"], x["kind"]))

    # 같은 자리에 여러 문장이 빠진 경우는 한 항목으로 묶는다
    # -> 보여 줄 코드가 없는 항목이 같은 줄 번호로 여러 번 반복되면 읽기 어렵다
    merged = []

    for item in items:
        last = merged[-1] if merged else None
        same = (last and last["kind"] == item["kind"] == _KIND_MISSING
                and last["line"] == item["line"])

        if same:
            last["count"] += 1
        else:
            merged.append({**item, "count": 1})

    return merged


def diff(module, path, source_dir=None, report=True, progress=True, force=False):
    """제출 파일을 원본 모듈의 지문과 대조한다.

    제출 파일은 실행하지 않고 구문만 해석하므로, 임포트가 깨져 있거나 실행 시
    부작용이 있는 파일도 안전하게 대조할 수 있다.

    Args:
        module (str): 원본 모듈 이름 (예: 'my_logit', 'my_ols').
        path (str): 제출한 파이썬 파일의 경로.
        source_dir (str): 원본 폴더 경로. 지정하면 동봉 지문 대신 이 폴더를
            실시간으로 참조한다. 강사용 (기본값: None).
        report (bool): 대조 결과를 바로 출력할지 여부 (기본값: True).
        progress (bool): 진행률 표시줄을 보여 줄지 여부 (기본값: True).
        force (bool): 문자열 내용만 다른 곳까지 함께 보고할지 여부 (기본값: False).
            대부분은 출력 문구의 표기 차이라 실행에 영향이 없으므로 기본으로는
            빼고 보여 준다. 딕셔너리 키나 비교 대상 문자열까지 훑어보려면 True.
            시그니처의 기본값은 이 설정과 무관하게 항상 대조한다.

    Returns:
        CompareResult: 대조 결과 객체.

    Raises:
        FileNotFoundError: 제출 파일이나 모듈 지문이 없는 경우.
        SyntaxError: 제출 파일에 구문 오류가 있는 경우.
    """
    target = Path(path)

    if not target.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    # 준비 단계만 먼저 잡고, 함수 수는 대조를 시작할 때 더한다
    bar = _Progress(total=4, enabled=progress)

    try:
        # --- 1) 원본 지문과 제출 파일의 지문을 준비 ---
        bar.step("원본 분석 중", 0)
        fingerprint = load_fingerprint(module, source_dir=source_dir)
        bar.step("제출 파일 읽는 중")

        # 보고서에서 문제 지점의 코드를 그대로 보여 주려면 원문이 필요하다
        source = target.read_text(encoding="utf-8")

        try:
            submitted = analyze_source(source, module=target.stem)
        except SyntaxError as e:
            raise SyntaxError(
                f"제출 파일에 구문 오류가 있어 대조할 수 없습니다.\n"
                f"  {path}:{e.lineno}줄 → {e.msg}") from None

        bar.step("함수 대조 중")
        result = _build_result(module, path, fingerprint, submitted, source,
                               bar, force)
    finally:
        bar.close()

    if report:
        result.report()

    return result


def _build_result(module, path, fingerprint, submitted, source, bar, force):
    """준비된 두 지문을 대조해 결과 객체를 만든다.

    Args:
        module (str): 원본 모듈 이름.
        path (str): 제출 파일 경로.
        fingerprint (dict): 원본 지문.
        submitted (dict): 제출 파일 지문.
        source (str): 제출 파일 원문.
        bar (_Progress): 진행률 표시줄.
        force (bool): 문자열 내용만 다른 곳까지 결과에 담을지 여부.

    Returns:
        CompareResult: 대조 결과 객체.
    """

    ref_funcs = fingerprint["functions"]
    sub_funcs = submitted["functions"]

    # --- 2) 함수마다 시그니처와 본문을 대조 ---
    defaults, params, rows = [], [], []
    details = {}
    suppressed = 0        # 문자열 차이라서 보고서에서 뺀 곳의 수

    # 함수 수만큼 칸을 늘려 대조가 진행되는 것이 보이게 한다
    bar.grow(len(ref_funcs))

    for name, ref in ref_funcs.items():
        sub = sub_funcs.get(name)
        bar.step(f"함수 대조 중 · {name}")

        # 제출 파일에 아예 없는 함수
        if sub is None:
            rows.append({"함수": name, "판정": "미작성", "본문": "-", "시그니처": "-"})
            continue

        # 시그니처 대조
        sig_ok = _compare_params(ref["params"], sub["params"], name, defaults, params)
        sig_ok = sig_ok and not any(d["함수"] == name for d in defaults)

        # 본문 대조 (전체 해시가 같으면 문장 단위 비교를 생략한다)
        if ref["code"] == sub["code"]:
            body_items = []
        else:
            body_items = _compare_statements(ref["statements"], sub["statements"])

        # 문자열 내용만 다른 곳은 기본적으로 빼고 몇 군데였는지만 세어 둔다
        if not force:
            hidden = [i for i in body_items if i["kind"] == _KIND_STRING]
            suppressed += sum(i.get("count", 1) for i in hidden)
            body_items = [i for i in body_items if i["kind"] != _KIND_STRING]

        if body_items:
            details[name] = body_items

        rows.append({
            "함수": name,
            "판정": "일치" if (sig_ok and not body_items) else "불일치",
            "본문": f"{len(body_items)}곳" if body_items else "일치",
            "시그니처": "일치" if sig_ok else "불일치",
        })

    # --- 3) 원본에 없는 함수 ---
    for name in sub_funcs:
        if name not in ref_funcs:
            rows.append({"함수": name, "판정": "원본에 없음", "본문": "-", "시그니처": "-"})

    # --- 4) 임포트 대조 ---
    bar.step("임포트 대조 중")
    ref_imports = set(fingerprint.get("imports", []))
    sub_imports = set(submitted.get("imports", []))
    import_rows = []

    for n in sorted(ref_imports - sub_imports):
        import_rows.append({"구분": "제출에 없는 임포트", "이름": n})

    for n in sorted(sub_imports - ref_imports):
        import_rows.append({"구분": "원본에 없는 임포트", "이름": n})

    # --- 5) 결과 객체 생성 ---
    result = CompareResult(
        module=module,
        path=str(path),
        fingerprint=fingerprint,
        defaults=DataFrame(defaults, columns=["함수", "파라미터", "원본", "제출"]),
        params=DataFrame(params, columns=["함수", "구분", "파라미터"]),
        functions=DataFrame(rows, columns=["함수", "판정", "시그니처", "본문"]),
        imports=DataFrame(import_rows, columns=["구분", "이름"]),
        details=details,
        source=source.splitlines(),
        force=force,
        suppressed=suppressed,
    )

    bar.step("보고서 만드는 중")

    return result
