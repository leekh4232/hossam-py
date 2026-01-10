from .data_loader import load_data, load_info
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
from importlib.metadata import version
import warnings

try:
    __version__ = version('hossam')
except Exception:
    __version__ = 'develop'
__all__ = ['load_data', 'load_info']

my_dpi = 200  # 이미지 선명도(100~300)
default_font_size = 6

def _init_korean_font():
	"""
	패키지에 포함된 한글 폰트를 기본 폰트로 설정합니다.
	"""
	font_file = 'NotoSansKR-Regular.ttf'
	try:
		# 패키지 리소스에서 폰트 파일 경로 확보
		with as_file(files('hossam') / font_file) as font_path:
			fm.fontManager.addfont(str(font_path))
			fprop = fm.FontProperties(fname=str(font_path))
			fname = fprop.get_name()
			plt.rcParams['font.family'] = fname
			plt.rcParams['font.size'] = default_font_size
			plt.rcParams['axes.unicode_minus'] = False
			return
	except Exception as e:
		warnings.warn(f"한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}).")


# 모듈 임포트 시점에 폰트 초기화 수행
_init_korean_font()