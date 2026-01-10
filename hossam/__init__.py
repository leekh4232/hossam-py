from .data_loader import load_data, load_info
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from importlib.resources import files, as_file
import os
import warnings

__version__ = '0.3.0'
__all__ = ['load_data', 'load_info']

my_dpi = 200  # 이미지 선명도(100~300)


def _init_korean_font():
	"""패키지에 포함된 한글 폰트를 자동 등록하고 기본 폰트로 설정합니다.

	- 패키지 내부의 NotoSansKR-Regular.ttf를 우선적으로 사용합니다.
	- 폰트 파일을 찾지 못하면 말굽고딕 등 시스템 폰트로 폴백합니다.
	"""
	# 환경 변수 제어
	env_disable = os.environ.get('HOSSAM_FONT_DISABLE', '').lower() in {'1', 'true', 'yes'}
	env_family = os.environ.get('HOSSAM_FONT_FAMILY')
	env_size = os.environ.get('HOSSAM_FONT_SIZE')
	env_path = os.environ.get('HOSSAM_FONT_PATH')

	if env_disable:
		return

	# 우선순위 1: 사용자 지정 폰트 경로
	if env_path:
		try:
			fm.fontManager.addfont(env_path)
			fprop = fm.FontProperties(fname=env_path)
			fname = fprop.get_name()
			plt.rcParams['font.family'] = env_family or fname
			plt.rcParams['font.size'] = int(env_size) if env_size else 6
			plt.rcParams['axes.unicode_minus'] = False
			return
		except Exception as e:
			warnings.warn(f"환경 변수 경로 폰트 사용 실패 ({e}). 패키지/시스템 폰트로 폴백합니다.")

	font_file = 'NotoSansKR-Regular.ttf'
	try:
		# 패키지 리소스에서 폰트 파일 경로 확보
		with as_file(files('hossam') / font_file) as font_path:
			fm.fontManager.addfont(str(font_path))
			fprop = fm.FontProperties(fname=str(font_path))
			fname = fprop.get_name()
			plt.rcParams['font.family'] = fname
			plt.rcParams['font.size'] = int(env_size) if env_size else 6
			plt.rcParams['axes.unicode_minus'] = False
			return
	except Exception as e:
		warnings.warn(f"한글 폰트 초기화: 패키지 폰트 사용 실패 ({e}). 시스템 폰트로 폴백합니다.")

	# 시스템 폰트 폴백 (Windows 우선: Malgun Gothic, 그 외 대안)
	fallback_fonts = [
		'Malgun Gothic',
		'NanumGothic',
		'AppleGothic',
		'Noto Sans CJK KR',
		'Noto Sans KR',
		'DejaVu Sans'
	]
	for ff in ([env_family] if env_family else []) + fallback_fonts:
		try:
			plt.rcParams['font.family'] = ff
			plt.rcParams['font.size'] = int(env_size) if env_size else 6
			plt.rcParams['axes.unicode_minus'] = False
			return
		except Exception:
			continue

	warnings.warn('한글 폰트 초기화 실패: 적절한 폰트를 찾지 못했습니다.')


# 모듈 임포트 시점에 폰트 초기화 수행
_init_korean_font()