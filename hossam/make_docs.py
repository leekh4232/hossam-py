# -*- coding: utf-8 -*-
"""API 레퍼런스 문서 생성 모듈

학생이 자신의 `helpers` 폴더에 작성한 소스 코드의 docstring 을 읽어
HTML API 레퍼런스 문서를 만들어 준다. 이 저장소가 GitHub Actions 에서
문서를 배포할 때 쓰는 방식(mkdocs + mkdocstrings)을 그대로 개인 PC 에서
실행하는 것이므로, 결과물의 모양은 공식 문서와 같다.

mkdocs 설정 파일이나 문서 페이지를 직접 만들 필요는 없다. 소스 폴더를 훑어
공개 모듈(`_` 로 시작하지 않는 최상위 `*.py`)마다 페이지와 목차를 자동으로
생성하므로, 파일을 추가하거나 지우면 문서도 따라 바뀐다.

사용 방법 (학생)

    from hossam import make_api_docs

    make_api_docs("helpers", "helpers-docs")

문서 생성에 필요한 패키지(mkdocs 계열)가 없으면 처음 한 번 자동으로 설치한다.
미리 설치해 두려면 다음과 같이 한다.

    pip install "hossam[docs]"
"""

import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

from ._config import PACKAGE_NAME

# 문서 생성에 필요한 패키지: (import 이름, pip 설치 이름)
_REQUIRED_PACKAGES = [
    ("mkdocs", "mkdocs"),
    ("material", "mkdocs-material"),
    ("mkdocstrings", "mkdocstrings[python]"),
    ("mkdocs_gen_files", "mkdocs-gen-files"),
    ("mkdocs_literate_nav", "mkdocs-literate-nav"),
]

# 이 모듈이 만든 출력 폴더임을 표시하는 파일 (다른 폴더를 덮어쓰지 않기 위한 표식)
_MARKER_NAME = f".{PACKAGE_NAME}-api-docs"


# -------------------------------------------------------------
def _find_missing_packages() -> list:
    """설치되지 않은 필수 패키지의 pip 이름 목록을 반환한다.

    Returns:
        list: 설치가 필요한 pip 패키지 이름 목록
    """
    from importlib.util import find_spec

    missing = []

    for import_name, pip_name in _REQUIRED_PACKAGES:
        if find_spec(import_name) is None:
            missing.append(pip_name)

    return missing


# -------------------------------------------------------------
def _install_packages(packages: list) -> None:
    """pip 로 패키지를 설치한다.

    Args:
        packages (list): 설치할 pip 패키지 이름 목록

    Raises:
        RuntimeError: 설치에 실패한 경우
    """
    print(f"[설치] 문서 생성에 필요한 패키지를 설치합니다: {', '.join(packages)}")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            "필수 패키지 설치에 실패했습니다. 아래 명령을 터미널에서 직접 실행한 뒤 다시 시도하세요.\n"
            f"    {' '.join(command)}"
        )

    print("[설치] 완료\n")


# -------------------------------------------------------------
def _find_modules(src_dir: Path) -> list:
    """문서화 대상 모듈 이름 목록을 반환한다.

    Args:
        src_dir (Path): 소스 폴더 경로

    Returns:
        list: `_` 로 시작하지 않는 최상위 `*.py` 의 모듈 이름 목록 (정렬됨)
    """
    modules = []

    for path in sorted(src_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue

        modules.append(path.stem)

    return modules


# -------------------------------------------------------------
def _check_output_dir(src_dir: Path, out_dir: Path, force: bool) -> None:
    """출력 폴더가 안전하게 사용할 수 있는 위치인지 검사한다.

    mkdocs 는 빌드 직전에 출력 폴더를 비우므로, 소스 폴더나 이미 다른 파일이
    들어 있는 폴더를 출력 폴더로 지정하면 그 내용이 지워질 수 있다.

    Args:
        src_dir (Path): 소스 폴더 경로
        out_dir (Path): 출력 폴더 경로
        force (bool): 비어 있지 않은 폴더도 사용할지 여부

    Raises:
        ValueError: 출력 폴더가 소스를 지울 수 있는 위치인 경우
    """
    if out_dir == src_dir:
        raise ValueError(
            f"출력 폴더가 소스 폴더와 같습니다: {out_dir}\n"
            "문서를 만들면서 소스가 지워지므로 다른 폴더를 지정하세요."
        )

    if out_dir in src_dir.parents:
        raise ValueError(
            f"출력 폴더가 소스 폴더의 상위 폴더입니다: {out_dir}\n"
            "문서를 만들면서 소스가 지워지므로 다른 폴더를 지정하세요."
        )

    if not out_dir.exists() or force:
        return

    # 이 모듈이 만든 폴더라면 그대로 덮어쓴다
    if (out_dir / _MARKER_NAME).exists():
        return

    if any(out_dir.iterdir()):
        raise ValueError(
            f"출력 폴더에 이미 다른 파일이 들어 있습니다: {out_dir}\n"
            "문서를 만들면서 그 파일들이 지워집니다. 비어 있는 폴더를 지정하거나,\n"
            "그래도 진행하려면 force=True 를 전달하세요."
        )


# -------------------------------------------------------------
def _write_gen_script(build_dir: Path, package: str, src_dir: Path) -> Path:
    """빌드 중 실행될 페이지 생성 스크립트를 만든다.

    Args:
        build_dir (Path): 빌드용 임시 폴더 경로
        package (str): 문서화할 패키지(폴더) 이름
        src_dir (Path): 소스 폴더의 실제 경로

    Returns:
        Path: 생성한 스크립트 경로
    """
    script = f'''"""빌드 시 소스에서 API 레퍼런스 페이지와 목차를 자동 생성한다."""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = {package!r}
src_dir = Path({str(src_dir)!r})

nav = mkdocs_gen_files.Nav()

# 패키지 개요 페이지
with mkdocs_gen_files.open(Path("api", "index.md"), "w") as fd:
    fd.write(f"---\\ntitle: {{PACKAGE}} Package\\n---\\n\\n")
    fd.write(f"# {{PACKAGE}} 패키지\\n\\n::: {{PACKAGE}}\\n")
nav["Package"] = "index.md"

# 최상위 모듈별 페이지 (하위 폴더/언더스코어 모듈 제외)
for path in sorted(src_dir.glob("*.py")):
    module = path.stem
    if module.startswith("_"):
        continue

    identifier = f"{{PACKAGE}}.{{module}}"

    with mkdocs_gen_files.open(Path("api", f"{{module}}.md"), "w") as fd:
        fd.write(f"---\\ntitle: {{identifier}}\\n---\\n\\n")
        fd.write(f"# {{identifier}}\\n\\n::: {{identifier}}\\n")

    # 문서의 'edit' 링크가 실제 소스 파일을 가리키도록 설정
    mkdocs_gen_files.set_edit_path(Path("api", f"{{module}}.md"), path)
    nav[module] = f"{{module}}.md"

# literate-nav 가 읽을 목차 파일 생성
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
'''
    script_path = build_dir / "gen_ref_pages.py"
    script_path.write_text(script, encoding="utf-8")

    return script_path


# -------------------------------------------------------------
def _write_index_page(docs_dir: Path, package: str, modules: list) -> None:
    """문서 첫 화면에 보일 개요 페이지를 만든다.

    Args:
        docs_dir (Path): 문서 소스 폴더 경로
        package (str): 패키지(폴더) 이름
        modules (list): 문서화 대상 모듈 이름 목록
    """
    lines = [
        f"# {package} API 레퍼런스",
        "",
        f"`{package}` 폴더의 소스 코드에 작성된 docstring 으로 만든 API 문서입니다.",
        "위쪽 **API Reference** 메뉴에서 모듈을 선택하세요.",
        "",
        f"## 모듈 목록 ({len(modules)}개)",
        "",
        "| 모듈 | 문서 |",
        "| --- | --- |",
    ]

    for module in modules:
        lines.append(f"| `{package}.{module}` | [바로가기](api/{module}.md) |")

    lines += [
        "",
        "## 문서를 다시 만들려면",
        "",
        "소스 코드를 고친 뒤 아래 코드를 다시 실행하면 문서가 갱신됩니다.",
        "",
        "```python",
        f"from {PACKAGE_NAME} import make_api_docs",
        "",
        f'make_api_docs("{package}", "...")',
        "```",
        "",
    ]

    (docs_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------
def _write_mkdocs_config(build_dir: Path, docs_dir: Path, out_dir: Path,
                         package: str, project_root: Path, gen_script: Path,
                         site_name: str) -> Path:
    """mkdocs 설정 파일을 만든다.

    Args:
        build_dir (Path): 빌드용 임시 폴더 경로 (mkdocs.yml 이 놓이는 곳)
        docs_dir (Path): 문서 소스 폴더 경로
        out_dir (Path): 완성된 HTML 문서를 저장할 폴더 경로
        package (str): 패키지(폴더) 이름
        project_root (Path): 패키지를 import 할 수 있는 기준 경로
        gen_script (Path): 페이지 생성 스크립트 경로
        site_name (str): 문서 상단에 표시할 제목

    Returns:
        Path: 생성한 mkdocs.yml 경로
    """
    config = f"""site_name: {site_name}
site_description: {package} 폴더의 모듈별 API 레퍼런스
docs_dir: {docs_dir.as_posix()}
site_dir: {out_dir.as_posix()}
use_directory_urls: false

theme:
  name: material
  language: ko
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-night
        name: 어두운 테마로 전환
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/weather-sunny
        name: 밝은 테마로 전환
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.indexes
    - navigation.top
    - content.code.copy
    - search.suggest
    - search.highlight

plugins:
  - search:
      lang: ko
  - autorefs
  - gen-files:
      scripts:
        - {gen_script.as_posix()}
  - literate-nav:
      nav_file: SUMMARY.md
  - mkdocstrings:
      handlers:
        python:
          paths:
            - {project_root.as_posix()}
          options:
            show_root_heading: true
            show_root_full_path: true
            show_source: true
            members_order: source
            docstring_style: google
            separate_signature: true
            filters:
              - "!^_"
            show_docstring_parameters: true
            show_docstring_returns: true
            show_docstring_raises: true
            show_docstring_examples: true

markdown_extensions:
  - admonition
  - footnotes
  - toc:
      permalink: true
  - attr_list
  - md_in_html
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.tabbed

nav:
  - Overview: index.md
  # API Reference 하위는 gen_ref_pages.py 가 만드는 api/SUMMARY.md 로 자동 구성됨
  - API Reference: api/
"""
    config_path = build_dir / "mkdocs.yml"
    config_path.write_text(config, encoding="utf-8")

    return config_path


# -------------------------------------------------------------
def _run_mkdocs(config_path: Path, project_root: Path, verbose: bool) -> int:
    """mkdocs build 명령을 실행한다.

    Args:
        config_path (Path): mkdocs.yml 경로
        project_root (Path): PYTHONPATH 에 추가할 경로
        verbose (bool): 빌드 상세 로그 출력 여부

    Returns:
        int: mkdocs 프로세스의 종료 코드
    """
    env = os.environ.copy()

    # mkdocstrings 가 대상 패키지를 찾을 수 있도록 경로를 추가한다
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{python_path}" if python_path else str(project_root)
    )

    # mkdocs / mkdocs-material 이 출력하는 공지성 배너를 끈다 (문서 생성과 무관)
    env["DISABLE_MKDOCS_2_WARNING"] = "true"
    env["NO_MKDOCS_2_WARNING"] = "true"

    command = [sys.executable, "-m", "mkdocs", "build", "-f", str(config_path)]

    # 상세 모드가 아니면 docstring 형식 경고 등을 감춘다
    if not verbose:
        command.append("-q")

    return subprocess.run(command, env=env).returncode


# -------------------------------------------------------------
def _show_link(index_file: Path) -> None:
    """노트북 환경이면 결과 문서로 가는 링크를 출력한다.

    Args:
        index_file (Path): 문서 시작 페이지(index.html) 경로
    """
    try:
        from IPython.display import HTML, display
        from IPython import get_ipython

        if get_ipython() is None:
            return

        display(HTML(
            f'<a href="{index_file.as_uri()}" target="_blank">'
            f'📄 생성된 API 문서 열기 ({index_file.name})</a>'
        ))
    except Exception:
        # 노트북이 아니거나 IPython 이 없으면 조용히 넘어간다
        pass


# -------------------------------------------------------------
def make_api_docs(src_dir: str, out_dir: str, site_name: str = None,
                  open_browser: bool = False, force: bool = False,
                  install: bool = True, verbose: bool = False) -> str:
    """소스 폴더의 docstring 을 읽어 HTML API 레퍼런스 문서를 생성한다.

    Args:
        src_dir (str): 문서화할 소스 폴더 경로 (예: `helpers`)
        out_dir (str): 문서가 생성될 폴더 경로 (예: `helpers-docs`)
        site_name (str): 문서 상단에 표시할 제목. 생략하면 `<폴더명> API Docs`
        open_browser (bool): 생성 후 기본 브라우저로 문서를 열지 여부
        force (bool): 출력 폴더에 다른 파일이 있어도 진행할지 여부
        install (bool): 필수 패키지가 없을 때 자동으로 설치할지 여부
        verbose (bool): 빌드 상세 로그를 모두 출력할지 여부

    Returns:
        str: 생성된 문서의 시작 페이지(index.html) 경로

    Raises:
        FileNotFoundError: 소스 폴더가 없거나 문서화할 `*.py` 가 없는 경우
        ValueError: 출력 폴더가 소스를 지울 수 있는 위치인 경우
        RuntimeError: 패키지 설치 또는 문서 빌드에 실패한 경우
    """
    # -------------------------------------
    # 경로 확인
    # -------------------------------------
    src_path = Path(src_dir).expanduser().resolve()

    if not src_path.is_dir():
        raise FileNotFoundError(
            f"소스 폴더를 찾을 수 없습니다: {src_path}\n"
            "노트북이 있는 위치를 기준으로 한 상대 경로이거나 절대 경로여야 합니다."
        )

    package = src_path.name
    project_root = src_path.parent
    out_path = Path(out_dir).expanduser().resolve()

    if not (src_path / "__init__.py").exists():
        print(f"[경고] {src_path}/__init__.py 가 없어 패키지 개요가 비어 보일 수 있습니다.")

    modules = _find_modules(src_path)

    if not modules:
        raise FileNotFoundError(f"{src_path} 안에 문서화할 .py 파일이 없습니다.")

    _check_output_dir(src_path, out_path, force)

    # -------------------------------------
    # 필수 패키지 확인 및 설치
    # -------------------------------------
    missing = _find_missing_packages()

    if missing:
        if not install:
            raise RuntimeError(
                "문서 생성에 필요한 패키지가 없습니다: " + ", ".join(missing) + "\n"
                f"    {sys.executable} -m pip install {' '.join(missing)}"
            )

        _install_packages(missing)

    # -------------------------------------
    # 설정 파일을 임시 폴더에 만들고 빌드
    # -------------------------------------
    print(f"[준비] 모듈 {len(modules)}개: {', '.join(modules)}")
    build_dir = Path(tempfile.mkdtemp(prefix=f"{PACKAGE_NAME}-docs-"))

    try:
        docs_dir = build_dir / "docs"
        docs_dir.mkdir()

        gen_script = _write_gen_script(build_dir, package, src_path)
        _write_index_page(docs_dir, package, modules)
        config_path = _write_mkdocs_config(
            build_dir, docs_dir, out_path, package, project_root,
            gen_script, site_name or f"{package} API Docs",
        )

        print("[빌드] 문서를 생성하는 중입니다...")
        code = _run_mkdocs(config_path, project_root, verbose)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    if code != 0:
        raise RuntimeError(
            "문서 빌드에 실패했습니다. verbose=True 로 다시 실행하면 원인을 볼 수 있습니다."
        )

    # 이 모듈이 만든 폴더임을 표시 (다음 실행 때 덮어쓰기 확인용)
    (out_path / _MARKER_NAME).write_text(
        f"이 폴더는 {PACKAGE_NAME}.make_api_docs() 가 생성한 문서 폴더입니다.\n", encoding="utf-8"
    )

    index_file = out_path / "index.html"
    print(f"[완료] 문서가 생성되었습니다: {out_path}")
    print(f"       {index_file} 파일을 브라우저로 열어보세요.")
    _show_link(index_file)

    if open_browser:
        webbrowser.open(index_file.as_uri())

    return str(index_file)
