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
from . import hs_ml
from . import hs_cluster
from . import hs_study
from .hs_util import load_info
from .hs_util import _load_data_remote as load_data
from .hs_plot import visualize_silhouette
from .hs_stats import ttest_ind as hs_ttest_ind
from .hs_stats import outlier_table as hs_outlier_table
from .hs_stats import oneway_anova as hs_oneway_anova
from .hs_ml import learning_cv as hs_learning_cv
from .hs_ml import reg_scores as hs_get_scores
from .hs_ml import cls_bin_scores as hs_cls_bin_scores
from .hs_ml import score_cv as hs_get_score_cv
from .hs_ml import feature_importance as hs_feature_importance
from .hs_ml import shap_analysis as hs_shap_analysis
from .hs_ml import shap_dependence_analysis as hs_shap_dependence_analysis
from .hs_stats import describe as hs_describe
from .hs_stats import category_describe as hs_category_describe
from .VIFSelector import VIFSelector
from .hs_util import tune_image as hs_tune_image
from .hs_util import load_image as hs_load_image

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

my_dpi = hs_plot.config.dpi # type: ignore

__all__ = [
    "my_dpi",
    "load_data",
    "load_info",
    "hs_classroom",
    "hs_gis",
    "hs_plot",
    "hs_prep",
    "hs_stats",
    "hs_timeserise",
    "hs_util",
    "hs_cluster",
    "hs_ml",
    "hs_study",
    "visualize_silhouette",
    "hs_ttest_ind",
    "hs_outlier_table",
    "hs_oneway_anova",
    "hs_learning_cv",
    "hs_get_scores",
    "hs_get_score_cv",
    "hs_cls_bin_scores",
    "hs_feature_importance",
    "hs_shap_analysis",
    "hs_shap_dependence_analysis",
    "VIFSelector",
    "init_pyplot",
    "hs_describe",
    "hs_category_describe",
    "hs_tune_image",
    "hs_load_image"
]


def check_pypi_latest(package_name: str):
    # 설치된 버전
    installed = importlib.metadata.version(package_name)
    print(f"현재 설치된 '{package_name}' 패키지 버전: {installed}")

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
        "outdated": installed < latest, # type: ignore
    }


def init_pyplot():
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

    # matplotlib 기본값으로 복원
    plt.rcParams.update(plt.rcParamsDefault)    # type: ignore

    # seaborn 스타일 제거
    sb.reset_defaults()
    sb.reset_orig()

    # 현재 figure에도 반영
    plt.rcdefaults()

    font_file = "NotoSansKR-Regular.ttf"
    try:
        # 패키지 리소스에서 폰트 파일 경로 확보
        with as_file(files("hossam") / font_file) as font_path:
            fm.fontManager.addfont(str(font_path))
            fprop = fm.FontProperties(fname=str(font_path)) # type: ignore
            fname = fprop.get_name()

            plt.rcParams.update(
                {
                    "font.family": fname,
                    "font.size": hs_plot.config.font_size,
                    "font.weight": hs_plot.config.font_weight,
                    #"text.fontsize": hs_plot.config.text_font_size,
                    "xtick.labelsize": hs_plot.config.font_size,
                    "ytick.labelsize": hs_plot.config.font_size,
                    "legend.fontsize": hs_plot.config.font_size,
                    "legend.title_fontsize": hs_plot.config.font_size,
                    "axes.titlesize": hs_plot.config.title_font_size,
                    "axes.titlepad": hs_plot.config.title_pad,
                    "figure.titlesize": hs_plot.config.title_font_size,
                    "axes.labelsize": hs_plot.config.label_font_size,
                    "axes.unicode_minus": False,
                    "text.antialiased": True,
                    "lines.antialiased": True,
                    "patch.antialiased": True,
                    "figure.dpi": hs_plot.config.dpi,
                    "savefig.dpi": hs_plot.config.dpi,
                    "text.hinting": "auto",
                    "text.hinting_factor": 8
                }
            )

            print(
                "\n✅ 시각화를 위한 한글 글꼴(NotoSansKR-Regular)이 자동 적용되었습니다."
            )
            return
    except Exception as e:
        raise Warning(f"\n한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}).")

    from IPython.display import display, HTML   # type: ignore

    display(
        HTML(
            """
            <style>      
            .dataframe tr:hover {
                background-color: #ffff99 !important;
                border: 1px solid #ffcc00;
            }
            </style>
            """
            )
        )


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

    version_info = check_pypi_latest("hossam")

    if __version__ != "develop" and version_info["outdated"]:
        print(
            f"\n⚠️  'hossam' 패키지의 최신 버전이 출시되었습니다! (설치된 버전: {version_info['installed']}, 최신 버전: {version_info['latest']})"
        )
        print("   최신 버전으로 업데이트하려면 다음 명령어를 실행하세요:")
        print("   pip install --upgrade hossam\n")

        raise Warning("hossam 패키지가 최신 버전이 아닙니다.")

    init_pyplot()

import multiprocessing as mp

def is_parallel_worker():
    return mp.current_process().name != "MainProcess"

if not is_parallel_worker():
    _init()
