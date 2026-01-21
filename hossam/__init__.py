# submodules
from . import hs_classroom
from . import hs_gis
from . import hs_plot
from . import hs_prep
from . import hs_stats
from . import hs_timeserise
from . import hs_util
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

__all__ = ["my_dpi", "load_data", "load_info", "hs_classroom", "hs_gis", "hs_plot", "hs_prep", "hs_stats", "hs_timeserise", "hs_util", "visualize_silhouette"]

# 내부 모듈에서 hs_fig를 사용할 때는 아래와 같이 import 하세요.
# from hossam import hs_fig


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
