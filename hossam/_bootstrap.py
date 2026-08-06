# -*- coding: utf-8 -*-
"""패키지 공통 초기화 로직.

패키지마다 달라지는 값(패키지 이름, 안내 메시지)은 전부 `_config.py` 에 들어 있다.
따라서 이 파일은 다른 패키지(hossam ↔ jussam)에 **그대로 복사해 붙여넣어도** 된다.
"""

import importlib.metadata
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import requests

from ._config import MESSAGES, PACKAGE_NAME

# 이 파일들이 패키지 상위 폴더에 있으면 설치본이 아니라 소스 트리를 참조 중인 것으로 본다.
_SOURCE_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")

# 그래프 이미지 선명도(100~300)
_FIGURE_DPI = 200


# -------------------------------------
# 버전 확인
# -------------------------------------
def resolve_version(package_dir: Path) -> str:
    """설치된 배포판의 버전을 반환한다. 소스코드를 직접 참조 중이면 "develop"."""
    if any((package_dir.parent / marker).exists() for marker in _SOURCE_MARKERS):
        return "develop"

    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "develop"


def check_pypi_latest(package_name: str = PACKAGE_NAME) -> dict:
    """설치된 버전과 PyPI 최신 버전을 비교한다.

    Args:
        package_name (str, optional): 조회할 패키지 이름. Defaults to 이 패키지.

    Returns:
        dict: package, installed, latest, outdated 키를 갖는 딕셔너리
    """
    installed = None
    latest = None

    try:
        # 설치된 버전
        installed = importlib.metadata.version(package_name)
        # PyPI 최신 버전
        url = f"https://pypi.org/pypi/{package_name}/json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
    except Exception:
        latest = None

    return {
        "package": package_name,
        "installed": installed,
        "latest": latest,
        "outdated": installed is not None and latest is not None and installed != latest,
    }


# -------------------------------------
# 출력 옵션 설정
# -------------------------------------
def _setup_pandas() -> None:
    # 각 열의 넓이 제한 없음
    pd.set_option("display.max_colwidth", None)
    # 출력 너비 제한 없음 (가로 스크롤될 수 있음)
    pd.set_option("display.width", None)
    # 컬럼 생략 금지
    pd.set_option("display.max_columns", None)
    # 행 최대 출력 수 100개로 수정
    pd.set_option("display.max_rows", 100)
    # 소수점 자리수 3자리로 설정
    pd.options.display.float_format = "{:.3f}".format


def _setup_matplotlib(package_dir: Path) -> None:
    """matplotlib 이 설치되어 있으면 한글 폰트와 그래프 기본값을 설정한다.

    matplotlib 은 선택적 의존성으로 취급한다. 설치되어 있지 않은 패키지에서는
    아무 것도 하지 않고 넘어가므로, 이 파일을 그대로 복사해도 import 가 깨지지 않는다.
    """
    try:
        from matplotlib import font_manager as fm
        from matplotlib import pyplot as plt
    except ImportError:
        return

    # 패키지 안에 fonts 폴더가 있으면 한글 폰트를 등록한다.
    font_dir = package_dir / "fonts"

    if font_dir.is_dir():
        for font_file in sorted(font_dir.glob("*.ttf")):
            fm.fontManager.addfont(str(font_file))            # 폰트 등록
            fprop = fm.FontProperties(fname=str(font_file))   # 폰트의 속성을 읽어옴
            plt.rcParams["font.family"] = fprop.get_name()    # 그래프에 한글 폰트 적용

    plt.rcParams["font.size"] = 12                # 기본 폰트 크기
    plt.rcParams["axes.unicode_minus"] = False    # 그래프에 마이너스 깨짐 방지
    plt.rcParams["figure.dpi"] = _FIGURE_DPI      # 그래프의 dpi 설정
    plt.rcParams["savefig.dpi"] = _FIGURE_DPI     # 저장되는 그래프의 dpi 설정
    plt.rcParams["lines.linewidth"] = 2           # 그래프 선 굵기 설정
    plt.rcParams["axes.axisbelow"] = True         # 그래프의 축과 격자선을 뒤에 배치


# -------------------------------------
# 안내 메시지
# -------------------------------------
def _print_banner(version: str) -> None:
    for msg in MESSAGES:
        print(msg)

    print(f"🔖 Version: {version}")

    # 개발 모드에서는 PyPI 조회(네트워크 대기)를 건너뛴다.
    if version == "develop":
        return

    version_info = check_pypi_latest()

    if version_info["outdated"]:
        print(
            f"\n⚠️  '{PACKAGE_NAME}' 패키지의 최신 버전이 출시되었습니다! "
            f"(설치된 버전: {version_info['installed']}, 최신 버전: {version_info['latest']})"
        )
        print("   최신 버전으로 업데이트하려면 다음 명령어를 실행하세요:")
        print(f"   pip install --upgrade {PACKAGE_NAME}\n")


# -------------------------------------
# 초기화 진입점
# -------------------------------------
def init(package_dir: Path) -> str:
    """패키지 초기화를 수행하고 확정된 버전 문자열을 반환한다.

    Args:
        package_dir (Path): 패키지 폴더 경로. 보통 `Path(__file__).resolve().parent`.

    Returns:
        str: 설치된 버전. 소스 트리에서 실행 중이면 "develop".
    """
    version = resolve_version(package_dir)

    _setup_pandas()
    _setup_matplotlib(package_dir)

    # 멀티프로세싱 워커마다 안내 메시지가 중복 출력되지 않도록 메인 프로세스에서만 출력
    if mp.current_process().name == "MainProcess":
        _print_banner(version)

    return version
