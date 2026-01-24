import importlib.metadata
import requests

# submodules
from . import hs_classroom
from . import hs_gis
from . import hs_plot
from . import hs_prep
from . import hs_stats
from . import hs_timeserise
from . import hs_util
from . import hs_cluster
from .hs_util import load_info
from .hs_util import _load_data_remote as load_data
from .hs_plot import visualize_silhouette

# py-modules
import sys
import warnings
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
from importlib.metadata import version

try:
    __version__ = version("hossam")
except Exception:
    __version__ = "develop"

my_dpi = hs_plot.config.dpi

__all__ = ["my_dpi", "load_data", "load_info", "hs_classroom", "hs_gis", "hs_plot", "hs_prep", "hs_stats", "hs_timeserise", "hs_util", "hs_cluster", "visualize_silhouette"]


def check_pypi_latest(package_name: str):
    # 설치된 버전
    installed = importlib.metadata.version(package_name)

    try:
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
        "outdated": installed != latest
    }


def _init_korean_font():
    """
    패키지에 포함된 한글 폰트를 기본 폰트로 설정합니다.
    """
    font_file = "NotoSansKR-Regular.ttf"
    try:
        # 패키지 리소스에서 폰트 파일 경로 확보
        with as_file(files("hossam") / font_file) as font_path:
            fm.fontManager.addfont(str(font_path))
            fprop = fm.FontProperties(fname=str(font_path))
            fname = fprop.get_name()

            plt.rcParams.update({
                "font.family": fname,
                "font.size": hs_plot.config.font_size,
                "font.weight": hs_plot.config.font_weight,
                "axes.unicode_minus": False,
                "text.antialiased": True,
                "lines.antialiased": True,
                "patch.antialiased": True,
                "figure.dpi": hs_plot.config.dpi,
                "savefig.dpi": hs_plot.config.dpi * 2,
                "text.hinting": "auto",
                "text.hinting_factor": 8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            })
            if sys.stdout.isatty():
                print(
                    "\n✅ 시각화를 위한 한글 글꼴(NotoSansKR-Regular)이 자동 적용되었습니다."
                )
            return
    except Exception as e:
        warnings.warn(f"\n한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}).")


def _init():

    version_info = check_pypi_latest("hossam")

    if version_info["outdated"]:
        print(
            f"\n⚠️  'hossam' 패키지의 최신 버전이 출시되었습니다! (설치된 버전: {version_info['installed']}, 최신 버전: {version_info['latest']})"
        )
        print("   최신 버전으로 업데이트하려면 다음 명령어를 실행하세요:")
        print("   pip install --upgrade hossam\n")

        raise Warning("hossam 패키지가 최신 버전이 아닙니다.")

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

    _init_korean_font()


_init()
