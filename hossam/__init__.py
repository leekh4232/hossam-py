from .data_loader import load_data, load_info
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
from importlib.metadata import version
import warnings

try:
    __version__ = version("hossam")
except Exception:
    __version__ = "develop"

__all__ = ["load_data", "load_info"]

my_dpi = 200  # 이미지 선명도(100~300)
default_font_size = 6


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
            plt.rcParams["font.family"] = fname
            plt.rcParams["font.size"] = default_font_size
            plt.rcParams["axes.unicode_minus"] = False
            print(
                "\n✅ 시각화를 위한 한글 글꼴(NotoSansKR-Regular)이 자동 적용되었습니다."
            )
            return
    except Exception as e:
        warnings.warn(f"\n한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}).")


def _init():
    # Jupyter Notebook 환경에서 로고 이미지 표시
    try:
        # IPython 환경인지 확인
        get_ipython()
        # Jupyter Notebook 환경
        from IPython.display import display, Image

        try:
            with as_file(files("hossam") / "leekh.png") as img_path:
                display(Image(filename=str(img_path)))
        except Exception:
            pass  # 이미지 로드 실패 시 무시하고 메시지만 출력
    except NameError:
        # IPython이 아닌 환경 (일반 Python 스크립트)
        pass

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
