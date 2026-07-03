import os
import glob as gl
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from pathlib import Path

import importlib.metadata
import requests
import pandas as pd
from importlib.resources import files
from importlib.metadata import version

# -------------------------------------
# 내보낼 모듈 임포트
# -------------------------------------
from . import my_qtcheck
from . import my_plot
from . import my_stats
from . import my_prep
from . import my_pipeline
from . import my_util
from .my_util import load_info
from .my_util import load_data

# -------------------------------------
# 한글 폰트 설정
# -------------------------------------
# 모듈의 현재 위치 (fonts 폴더 접근용)
MODULE_DIR = Path(__file__).resolve().parent

# 한글을 지원하는 폰트 파일의 경로
fpath = f"{MODULE_DIR}/fonts"
font_files = gl.glob(os.path.join(fpath, "*.ttf")) # 폰트 파일 검색

for f in font_files:
    fm.fontManager.addfont(f)             # 폰트 등록
    fprop = fm.FontProperties(fname=f)    # 폰트의 속성을 읽어옴
    fname = fprop.get_name()              # 읽어온 속성에서 폰트이름 추출
    plt.rcParams['font.family'] = fname   # 그래프에 한글 폰트 적용

# -------------------------------------
# 그래프 기본 설정
# -------------------------------------
my_dpi = 200                               # 이미지 선명도(100~300)
plt.rcParams['font.size']   = 12           # 기본 폰트 크기
plt.rcParams['axes.unicode_minus'] = False # 그래프에 마이너스 깨짐 방지
plt.rcParams['figure.dpi'] = my_dpi        # 그래프의 dpi 설정
plt.rcParams['savefig.dpi'] = my_dpi       # 저장되는 그래프의 dpi 설정
plt.rcParams['lines.linewidth'] = 2        # 그래프 선 굵기 설정
plt.rcParams['axes.axisbelow'] = True      # 그래프의 축과 격자선을 뒤에 배치

# -------------------------------------
# 버전체크
# -------------------------------------
def _resolve_version() -> str:
    project_root = MODULE_DIR.parent
    source_markers = ("pyproject.toml", "setup.py", "setup.cfg")
    if any((project_root / marker).exists() for marker in source_markers):
        # 소스코드를 직접 참조 중(개발 모드)
        return "develop"

    try:
        return version("hossam")
    except importlib.metadata.PackageNotFoundError:
        return "develop"


__version__ = _resolve_version()

# -------------------------------------
# pandas 출력 옵션 설정
# -------------------------------------
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



# -------------------------------------
# 버전체크 함수
# -------------------------------------
def check_pypi_latest(package_name: str):
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
        "outdated": installed is not None and latest is not None and installed != latest
    }



# -------------------------------------
# 초기화
# -------------------------------------
messages = [
    "📦 아이티윌 이광호 강사가 제작한 라이브러리를 사용중입니다.",
    "📚 자세한 사용 방법은 https://py.hossam.kr 을 참고하세요.",
    "📧 Email: leekh4232@gmail.com",
    "🎬 Youtube: https://www.youtube.com/@hossam-codingclub",
    "📝 Blog: https://blog.hossam.kr/",
    f"🔖 Version: {__version__}",
]

for msg in messages:
    print(f"{msg}")

version_info = check_pypi_latest("hossam")

if __version__ != "develop" and version_info["outdated"]:
    print(
        f"\n⚠️  'hossam' 패키지의 최신 버전이 출시되었습니다! (설치된 버전: {version_info['installed']}, 최신 버전: {version_info['latest']})"
    )
    print("   최신 버전으로 업데이트하려면 다음 명령어를 실행하세요:")
    print("   pip install --upgrade hossam\n")
