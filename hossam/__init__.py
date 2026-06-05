import os
IMPORT_VIA = os.environ.get("_HOSSAM_IMPORT_VIA", "direct")

import importlib.metadata
import multiprocessing as mp
import requests
from pathlib import Path

# submodules
from . import my_classroom
from . import my_util
from . import my_plot
from . import my_qtcheck
from .my_util import load_info
from .my_util import _load_data_remote as load_data

# py-modules
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
from importlib.metadata import version

try:
    __version__ = version("hossam")
except Exception:
    __version__ = "develop"

my_dpi = my_plot.config.dpi # type: ignore

__all__ = [
    "load_data",
    "load_info",
    "my_util",
    "my_plot",
    "my_classroom",
    "my_qtcheck"
]


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

def _init():

    if IMPORT_VIA == "direct":
        # 안내 메시지 (블릿 리스트)
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

    # 모듈의 현재 위치 (fonts 폴더 접근용)
    MODULE_DIR = Path(__file__).resolve().parent

    # 한글을 지원하는 폰트 파일의 경로
    fpath = f"{MODULE_DIR}/fonts"
    font_files = os.listdir(fpath)

    try:
        fname = None

        for f in font_files:
            # 폰트 파일의 전체 경로
            font_path = os.path.join(fpath, f)
            fm.fontManager.addfont(str(font_path))
            fprop = fm.FontProperties(fname=str(font_path)) # type: ignore
            
            if not fname:
                fname = fprop.get_name()
            
        plt.rcParams.update(
            {
                "font.family": fname,
                "font.size": my_plot.config.font_size,
                "font.weight": my_plot.config.font_weight,
                "xtick.labelsize": my_plot.config.font_size,
                "ytick.labelsize": my_plot.config.font_size,
                "legend.fontsize": my_plot.config.font_size,
                "legend.title_fontsize": my_plot.config.font_size,
                "axes.titlesize": my_plot.config.title_font_size,
                "axes.titlepad": my_plot.config.title_pad,
                "figure.titlesize": my_plot.config.title_font_size,
                "axes.labelsize": my_plot.config.label_font_size,
                "axes.unicode_minus": False,
                "text.antialiased": True,
                "lines.antialiased": True,
                "patch.antialiased": True,
                "figure.dpi": my_plot.config.dpi,
                "savefig.dpi": my_plot.config.dpi,
                "text.hinting": "auto",
                "text.hinting_factor": 8,
                "axes.axisbelow": True
            }
        )
    except Exception as e:
        raise Warning(f"\n한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}).")

if mp.current_process().name == "MainProcess":
    _init()
