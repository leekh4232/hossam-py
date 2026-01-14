from .data_loader import load_data, load_info
from .hs_stats import oneway_anova
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
from importlib.metadata import version
from types import SimpleNamespace
import sys
import warnings

try:
    __version__ = version("hossam")
except Exception:
    __version__ = "develop"


hs_fig = SimpleNamespace(
    dpi=200,
    width=800,
    height=480,
    font_size=9.5,
    font_weight="light",
    frame_width=0.5,
    line_width=1.5,
    grid_alpha=0.3,
    grid_width=0.5,
    fill_alpha=0.3
)

__all__ = ["load_data", "load_info", "hs_classroom", "hs_gis", "hs_plot", "hs_prep", "hs_stats", "hs_timeserise", "hs_util", "hs_fig"]


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
                "font.size": hs_fig.font_size,
                "font.weight": hs_fig.font_weight,
                "axes.unicode_minus": False,
                "text.antialiased": True,
                "lines.antialiased": True,
                "patch.antialiased": True,
                "figure.dpi": hs_fig.dpi,
                "savefig.dpi": hs_fig.dpi * 2,
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

    # MCP/stdio 환경에서는 배너를 출력하지 않음 (stdout TTY일 때만 출력)
    if sys.stdout.isatty():
        for msg in messages:
            print(f"{msg}")

    _init_korean_font()


_init()
