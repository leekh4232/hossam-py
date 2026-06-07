"""
hossam 명령줄 인터페이스(CLI)

설치 후 터미널에서 다음과 같이 사용한다.

    hossam -m qtcheck mydataset.xlsx     # 로컬 파일(확장자 있음)
    hossam -m qtcheck iris               # 원격 데이터셋(확장자 없는 키)

-m(--mode)으로 실행할 기능(모드)을 선택하고, 그 뒤에 데이터셋을 전달한다.
- 확장자가 있으면 로컬 파일로 읽는다: Excel(.xlsx, .xls), CSV(.csv), Parquet(.parquet)
- 확장자 없이 단어만 주면 원격 데이터셋 키로 보고 my_util._load_data_remote 로 로드한다.

새로운 기능은 아래 MODES 딕셔너리에 핸들러를 등록하는 것만으로 추가할 수 있다.
"""

import argparse
import builtins
import os
import sys

from pandas import read_csv, read_excel, read_parquet


# 지원하는 파일 확장자별 로더
_LOADERS = {
    ".xlsx": read_excel,
    ".xls": read_excel,
    ".csv": read_csv,
    ".parquet": read_parquet,
}


def _load_dataset(path):
    """
    데이터셋을 읽어 DataFrame으로 반환한다.

    - 확장자가 있으면 로컬 파일로 읽는다(.xlsx, .xls, .csv, .parquet).
    - 확장자가 없으면(예: "iris") 원격 데이터셋 키로 보고
      my_util._load_data_remote 로 로드한다.

    Args:
        path (str): 로컬 파일 경로 또는 원격 데이터셋 키

    Returns:
        DataFrame: 로드된 데이터프레임
    """
    ext = os.path.splitext(path)[1].lower()

    # 확장자가 없으면 원격 데이터셋 키로 간주하고 로드
    if ext == "":
        from .my_util import _load_data_remote

        print(f"🌐 원격 데이터셋을 키로 불러옵니다: {path}")
        data = _load_data_remote(path)
        if data is None:
            raise ValueError(
                f"원격 데이터셋을 찾을 수 없습니다: '{path}' "
                "(메타데이터에 등록된 키인지 확인하세요)"
            )
        return data

    # 확장자가 있으면 로컬 파일로 로드
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    loader = _LOADERS.get(ext)
    if loader is None:
        supported = ", ".join(sorted(_LOADERS.keys()))
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: '{ext}' (지원 형식: {supported})"
        )

    print(f"📂 데이터셋을 불러옵니다: {path}")
    return loader(path)


def _ensure_display():
    """
    터미널에는 Jupyter의 display()가 없으므로, 없을 때만 print로 대체한다.
    (categorical_summary 등 라이브러리 함수가 display를 직접 호출하는 경우 대비)
    """
    if not hasattr(builtins, "display"):
        builtins.display = print


def _run_qtcheck(args):
    """
    품질 점검(qtcheck) 모드: 데이터셋을 읽어 대화형 품질 점검을 수행한다.
    """
    from .qtcheck_cli import auto_qtcheck

    data = _load_dataset(args.dataset)

    # 저장 파일명 접두어로 사용할 데이터셋 이름(확장자 제외)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    _ensure_display()
    auto_qtcheck(data, dataset_name=dataset_name)


# 모드 이름 -> (핸들러, 설명) 매핑. 새 기능은 여기에 추가한다.
MODES = {
    "qtcheck": (_run_qtcheck, "데이터 품질 점검(대화형)을 수행합니다."),
}


def _build_parser():
    """argparse 파서를 구성한다."""
    mode_help = "실행할 기능(모드). 사용 가능: " + ", ".join(
        f"{name} - {desc}" for name, (_, desc) in MODES.items()
    )

    parser = argparse.ArgumentParser(
        prog="hossam",
        description="hossam 데이터 도우미 명령줄 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  hossam -m qtcheck mydataset.xlsx   # 로컬 파일\n"
               "  hossam -m qtcheck iris             # 원격 데이터셋 키",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=list(MODES.keys()),
        metavar="MODE",
        help=mode_help,
    )
    parser.add_argument(
        "dataset",
        help="로컬 파일 경로(.xlsx, .xls, .csv, .parquet) 또는 "
             "확장자 없는 원격 데이터셋 키(예: iris)",
    )
    return parser


def main(argv=None):
    """CLI 진입점."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler, _ = MODES[args.mode]

    try:
        handler(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
