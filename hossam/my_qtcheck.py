import numpy as np
from pandas import to_datetime, DataFrame, ExcelWriter

def set_type(data, as_int=[], as_float=[], as_string=[], 
             as_category=[], as_datetime=[]):
    """
    데이터프레임의 컬럼 타입을 변경하고 
    변경된 데이터프레임의 정보를 출력하는 함수

    Args:
        data (DataFrame): 타입을 변경할 데이터프레임
        as_int (list): int 타입으로 변경할 컬럼 리스트
        as_float (list): float 타입으로 변경할 컬럼 리스트
        as_string (list): string 타입으로 변경할 컬럼 리스트
        as_category (list): category 타입으로 변경할 컬럼 리스트
        as_datetime (list): datetime 타입으로 변경할 컬럼 리스트

    Returns:
        DataFrame: 타입이 변경된 데이터프레임
    """
    df = data.copy()
    
    for col in as_int:
        df[col] = df[col].astype(int)
    for col in as_float:
        df[col] = df[col].astype(float)
    for col in as_string:
        df[col] = df[col].astype(str)
    for col in as_category:
        df[col] = df[col].astype('category')
    for col in as_datetime:
        df[col] = to_datetime(df[col])

    df.info()

    return df

def get_number_column_names(data):
    """
    데이터프레임에서 숫자형 컬럼의 이름을 리스트로 반환하는 함수

    Args:
        data (DataFrame): 숫자형 컬럼의 이름을 추출할 데이터프레임

    Returns:
        list: 숫자형 컬럼의 이름 리스트
    """
    return data.select_dtypes(include="number").columns.to_list()

def get_categorical_column_names(data):
    """
    데이터프레임에서 범주형 컬럼의 이름을 리스트로 반환하는 함수

    Args:
        data (DataFrame): 범주형 컬럼의 이름을 추출할 데이터프레임

    Returns:
        list: 범주형 컬럼의 이름 리스트
    """
    return data.select_dtypes(include="category").columns.to_list()
    

def check_duplicates(data, drop=True):
    """
    데이터프레임에서 행 단위 중복을 검사하고, 중복된 행을 제거하는 함수

    Args:
        data (DataFrame): 중복을 검사할 데이터프레임
        drop (bool): 중복된 행을 제거할지 여부 (기본값: True)

    Returns:
        DataFrame: 중복이 제거된 데이터프레임
    """
    df = data.copy()
    duplicate_rows = df.duplicated()
    num_duplicates = duplicate_rows.sum() 
    print(f"중복된 행의 수: {num_duplicates}")
    
    if drop and num_duplicates > 0:
        df = df.drop_duplicates()
        print("중복된 행이 제거되었습니다.")
    
    return df


def check_missing_values(data):
    """
    데이터프레임에서 컬럼별 결측치 개수와 비율을 계산하여 데이터 프레임으로 반환하는 함수

    Args:
        data (DataFrame): 결측치를 점검할 데이터프레임

    Returns:
        DataFrame: 컬럼별 결측치 개수와 비율이 포함된 데이터프레임
    """
    na_count = data.isna().sum()
    na_ratio = (na_count / len(data)) * 100

    return DataFrame({
        'Missing Count': na_count,
        'Missing Ratio (%)': na_ratio
    })


def categorical_summary(data, columns=None, value_counts=True, save_path=None):
    """
    데이터프레임의 범주형 컬럼에 대한 요약 통계를 반환하는 함수

    Args:
        data (DataFrame): 범주형 컬럼의 요약 통계를 출력할 데이터프레임
        columns (list): 요약 통계를 출력할 범주형 컬럼 리스트
        value_counts (bool): 각 범주형 컬럼의 value_counts()를 출력할지 여부 (기본값: True)
        save_path (str): 요약 통계 결과를 CSV 파일로 저장할 경로 (기본값: None, 저장하지 않음)

    Returns:
        DataFrame: 범주형 컬럼에 대한 요약 통계가 포함된 데이터프레임
    """
    # columns가 비어있으면 데이터프레임에서 범주형 컬럼의 이름을 가져옴
    if not columns:
        columns = get_categorical_column_names(data)

    # 대상 컬럼으로 데이터프레임 생성
    df = data[columns].copy()

    # 명목형 변수의 기술 통계량 계산
    desc_df = df.describe(include="category")

    # 저장될 파일 경로가 전달된 경우 기술 통계량을 Excel 파일로 저장
    if save_path:
        desc_df.to_excel(save_path, sheet_name='Summary', index=True)
    
    # 각 범주형 컬럼의 value_counts()를 출력해야 한다면?
    if value_counts:
        for col in columns:
            cdf = DataFrame(data[col].value_counts())
            cdf.index.name = col
            cdf.sort_index(inplace=True)
            print(f"📊 컬럼 '{col}'의 value_counts():")
            display(cdf)

            # 저장될 파일 경로가 전달된 경우 value_counts 결과를 Excel 파일로 저장
            if save_path:
                # 기존 파일에 이어 쓰기를 수행하기 위해 ExcelWriter를 사용하여 시트별로 저장
                with ExcelWriter(save_path, mode='a') as excel_writer:
                    cdf.to_excel(excel_writer, sheet_name=col, index=True)

    return desc_df

def numerical_summary(data, columns=None, save_path=None):
    """
    데이터프레임의 숫자형 컬럼에 대한 요약 통계를 반환하는 함수

    Args:
        data (DataFrame): 숫자형 컬럼의 요약 통계를 출력할 데이터프레임
        columns (list): 요약 통계를 출력할 숫자형 컬럼 리스트
        save_path (str): 요약 통계 결과를 CSV 파일로 저장할 경로 (기본값: None, 저장하지 않음)

    Returns:
        DataFrame: 숫자형 컬럼에 대한 요약 통계가 포함된 데이터프레임
    """
    #-----------------------------------------------------
    # 1) columns가 비어있으면 데이터프레임에서 숫자형 컬럼의 이름을 가져옴
    #-----------------------------------------------------
    if not columns:
        columns = get_number_column_names(data)

    desc_df = data[columns].describe().T

    #-----------------------------------------------------
    # 2) 평균-중앙값의 상대 차이율을 계산하여 중심 수준 파악
    #-----------------------------------------------------
    # "평균-중앙값 상대 차이율 = |평균 - 중앙값| / 중앙값" 컬럼 추가
    desc_df['rel_diff'] = abs(desc_df['mean'] - desc_df['50%']) / desc_df['50%']

    # 상대 차이율 의미 컬럼 추가
    conditions = [desc_df['rel_diff'] < 0.1, desc_df['rel_diff'] < 0.5]
    choices = ['similar', 'diff']
    desc_df['rdiff_flag'] = np.select(conditions, choices, default='large_diff')

    #-----------------------------------------------------
    # 3) IQR, 이상치 경계값 계산
    #-----------------------------------------------------
    # iqr
    desc_df['iqr'] = desc_df['75%'] - desc_df['25%']

    # 상한 이상치 경계
    desc_df['upper_bound'] = desc_df['75%'] + 1.5 * desc_df['iqr']

    # 하한 이상치 경계
    desc_df['lower_bound'] = desc_df['25%'] - 1.5 * desc_df['iqr']

    #-----------------------------------------------------
    # 4) 명목형 변수를 제외한 데이터 프레임 생성
    #-----------------------------------------------------
    df = data[columns].copy()

    #-----------------------------------------------------
    # 5) 상한 이상치 탐지
    #-----------------------------------------------------
    # 상한 이상치 수
    desc_df['upper_outliers'] = ((df > desc_df['upper_bound'])).sum()

    # 상한 이상치 수 비율
    desc_df['upper_outliers_ratio'] = desc_df['upper_outliers'] / df.shape[0]

    #-----------------------------------------------------
    # 6) 하한 이상치 탐지
    #-----------------------------------------------------
    # 하한 이상치 수
    desc_df['lower_outliers'] = ((df < desc_df['lower_bound'])).sum()

    # 하한 이상치 수 비율
    desc_df['lower_outliers_ratio'] = desc_df['lower_outliers'] / df.shape[0]

    #-----------------------------------------------------
    # 7) 전체 이상치 집계
    #-----------------------------------------------------
    # 통합 이상치 수
    desc_df['outliers'] = desc_df['upper_outliers'] + desc_df['lower_outliers']

    # 통합 이상치 수 비율
    desc_df['outliers_ratio'] = desc_df['outliers'] / df.shape[0]

    #-----------------------------------------------------
    # 8) 왜도 점검
    #-----------------------------------------------------
    # 왜도 계산
    desc_df['skew'] = df.skew()

    # 왜도를 통한 분포 형태 해석
    conditions_skew = [(desc_df['skew'] < -0.5), (desc_df['skew'] > 0.5)]
    choices_skew = ['left tail', 'right tail']
    desc_df['skew_interpret'] = np.select(conditions_skew, choices_skew, default='symmetric')

    #-----------------------------------------------------
    # 9) 첨도 점검
    #-----------------------------------------------------
    # 첨도 계산
    desc_df['kurt'] = df.kurt()

    # 첨도를 통한 분포 형태 해석
    conditions_kurt = [(desc_df['kurt'] < 0), (desc_df['kurt'] > 0)]
    choices_kurt = ['platykurtic', 'leptokurtic']
    desc_df['kurt_interpret'] = np.select(conditions_kurt, choices_kurt, default='mesokurtic')

    #-----------------------------------------------------
    # 10) 로그 변환 필요성 판단 함수 정의 (inner function)
    #-----------------------------------------------------
    def judge_log_transform(skew, kurt):
        if skew >= 1:                        # 강한 우측 꼬리 분포
            return "log1p"
        elif skew > 0.5 and kurt > 0:       # 우측 꼬리 분포이면서 첨도가 높은 경우
            return "log1p"
        elif skew <= -1:                    # 강한 좌측 꼬리 분포
            return "reverse_log1p"
        elif skew < -0.5 and kurt > 0:      # 좌측 꼬리 분포이면서 첨도가 높은 경우
            return "reverse_log1p"
        else:                               # 대칭 분포
            return "none"

    #-----------------------------------------------------
    # 11) 로그 변환 필요성 판정
    #-----------------------------------------------------
    desc_df['log_need'] = desc_df.apply(lambda row: judge_log_transform( row['skew'], row['kurt']), axis=1)

    #-----------------------------------------------------
    # 12) 기술 통계량 표 저장
    #-----------------------------------------------------
    # 저장 경로 파라미터가 전달되었다면 기술 통계량 표를 Excel 파일로 저장
    if save_path:
        desc_df.to_excel(save_path, index=True)

    #-----------------------------------------------------
    # 13) 결과 리턴
    #-----------------------------------------------------
    return desc_df


# =====================================================================
# 대화형 데이터 품질 점검 자동화
# =====================================================================

def _prompt(message, default=""):
    """
    기본값을 지원하는 입력 프롬프트. 엔터만 누르면 기본값을 사용한다.

    Args:
        message (str): 사용자에게 보여줄 안내 문구
        default (str): 입력이 없을 때 사용할 기본값

    Returns:
        str: 사용자 입력값(없으면 기본값)
    """
    hint = f" [{default}]" if default != "" else ""
    try:
        answer = input(f"{message}{hint}\n> ").strip()
    except EOFError:
        answer = ""
    return answer if answer else default


def _prompt_yes_no(message, default=True):
    """
    예/아니오 입력 프롬프트.

    Args:
        message (str): 질문 문구
        default (bool): 입력이 없을 때의 기본 선택

    Returns:
        bool: True(예) / False(아니오)
    """
    d = "Y/n" if default else "y/N"
    answer = _prompt(f"{message} ({d})", "").lower()
    if answer in ("y", "yes", "예", "ㅇ"):
        return True
    if answer in ("n", "no", "아니오", "ㄴ"):
        return False
    return default


def _show(obj):
    """Jupyter 환경이면 display, 아니면 print로 출력한다."""
    try:
        display(obj)
    except NameError:
        print(obj)


def _parse_columns(raw, data):
    """
    콤마로 구분된 컬럼 문자열을 데이터프레임에 실제 존재하는 컬럼 리스트로 변환한다.

    Args:
        raw (str): "cut, color, clarity" 형태의 문자열
        data (DataFrame): 컬럼 존재 여부를 검사할 데이터프레임

    Returns:
        list: 유효한 컬럼명 리스트
    """
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    valid, invalid = [], []
    for c in cols:
        (valid if c in data.columns else invalid).append(c)
    if invalid:
        print(f"⚠️  존재하지 않는 컬럼은 무시합니다: {invalid}")
    return valid


def auto_qtcheck(data, dataset_name="dataset"):
    """
    데이터 품질 점검 프로세스(a.ipynb)를 대화형으로 자동 실행하는 함수.

    호출하면 아래 단계를 순서대로 진행하며, 각 단계의 분기점마다 사용자 입력을
    받아 my_qtcheck의 기존 기능들을 자동으로 호출한다. 모든 질문은 엔터만 누르면
    대괄호 안의 기본값이 적용된다.

        1. 자료형 확인            → DataFrame.info()
        2. 자료형 변환            → set_type()
        3. 중복 점검/제거         → check_duplicates()
        4. 결측치 점검/삭제       → check_missing_values() (발견 시 삭제 여부 질문)
        5. 명목형 기술 통계량     → categorical_summary()
        6. 연속형 기술 통계량     → numerical_summary()
        7. 이상치 점검/삭제       → IQR 1.5배 경계 기준 (발견 시 삭제 여부 질문)
        8. 품질 검사 결과 저장    → DataFrame.to_excel() (정제 완료본)

    Args:
        data (DataFrame): 품질 점검 대상 데이터프레임 (예: load_data로 불러온 origin)
        dataset_name (str): 저장 파일명 접두어로 사용할 데이터셋 이름

    Returns:
        DataFrame: 모든 점검/정제를 마친 데이터프레임
    """
    print("=" * 60)
    print(f"🔍 [{dataset_name}] 데이터 품질 점검을 시작합니다.")
    print("   (각 질문에 엔터만 누르면 대괄호 안의 기본값이 적용됩니다.)")
    print("=" * 60)

    #-----------------------------------------------------
    # 1) 자료형 확인
    #-----------------------------------------------------
    print("\n## #02. 자료형 점검 — 1. 자료형 확인\n")
    data.info()

    #-----------------------------------------------------
    # 2) 자료형 변환
    #-----------------------------------------------------
    print("\n## #02. 자료형 점검 — 2. 자료형 변환\n")
    object_cols = data.select_dtypes(include="object").columns.to_list()
    default_cat = ", ".join(object_cols)
    raw = _prompt("범주형(category)으로 변환할 컬럼을 콤마로 구분해 입력하세요. "
                  "(변환하지 않으려면 none 입력)", default_cat)

    if raw.lower() == "none" or raw == "":
        df = data.copy()
        print("➡️  자료형 변환을 건너뜁니다.")
    else:
        category_cols = _parse_columns(raw, data)
        df = set_type(data, as_category=category_cols)

    #-----------------------------------------------------
    # 3) 중복 점검 / 제거
    #-----------------------------------------------------
    print("\n## #03. 데이터 중복 점검\n")
    drop = _prompt_yes_no("중복된 행을 제거할까요?", default=True)
    df = check_duplicates(df, drop=drop)

    #-----------------------------------------------------
    # 4) 결측치 점검 / 삭제
    #-----------------------------------------------------
    print("\n## #04. 결측치 점검\n")
    missing = check_missing_values(df)
    _show(missing)

    total_missing = int(missing["Missing Count"].sum())
    if total_missing == 0:
        print("✅ 결측치가 없습니다.")
    else:
        print(f"⚠️  총 {total_missing}개의 결측치가 발견되었습니다.")
        if _prompt_yes_no("결측치가 포함된 행을 삭제할까요?", default=True):
            before = df.shape[0]
            df = df.dropna()
            print(f"🧹 결측치가 포함된 행 {before - df.shape[0]}개를 삭제했습니다. "
                  f"(현재 {df.shape[0]}행)")
        else:
            print("➡️  결측치를 유지합니다.")

    #-----------------------------------------------------
    # 5) 명목형 변수 기술 통계량
    #-----------------------------------------------------
    print("\n## #05. 기술 통계량 — 명목형 변수\n")
    if get_categorical_column_names(df):
        cat_save = None
        if _prompt_yes_no("명목형 요약을 Excel로 저장할까요?", default=True):
            cat_save = f"{dataset_name}_category_summary.xlsx"   # 용도에 맞는 파일명 자동 지정
        categorical_summary(df, save_path=cat_save)
        if cat_save:
            print(f"💾 저장 완료: {cat_save}")
    else:
        print("ℹ️  범주형(category) 컬럼이 없어 건너뜁니다. "
              "(2단계에서 자료형을 변환했는지 확인하세요.)")

    #-----------------------------------------------------
    # 6) 연속형 변수 기술 통계량
    #-----------------------------------------------------
    print("\n## #06. 기술 통계량 — 연속형 변수\n")
    num_save = None
    if _prompt_yes_no("연속형 요약을 Excel로 저장할까요?", default=True):
        num_save = f"{dataset_name}_numerical_summary.xlsx"      # 용도에 맞는 파일명 자동 지정
    desc_df = numerical_summary(df, save_path=num_save)
    _show(desc_df)
    if num_save:
        print(f"💾 저장 완료: {num_save}")

    #-----------------------------------------------------
    # 7) 이상치 점검 / 삭제
    #    numerical_summary가 계산한 IQR(1.5배) 경계값을 이용해 이상치 행을 탐지
    #-----------------------------------------------------
    print("\n## #07. 이상치 점검\n")
    num_cols = get_number_column_names(df)
    lower = desc_df.loc[num_cols, "lower_bound"]
    upper = desc_df.loc[num_cols, "upper_bound"]
    outlier_mask = ((df[num_cols] < lower) | (df[num_cols] > upper)).any(axis=1)
    n_outlier_rows = int(outlier_mask.sum())

    if n_outlier_rows == 0:
        print("✅ 이상치가 없습니다.")
    else:
        print(f"⚠️  이상치가 포함된 행이 {n_outlier_rows}개 발견되었습니다. "
              "(IQR 1.5배 경계 기준)")
        if _prompt_yes_no("이상치가 포함된 행을 삭제할까요?", default=False):
            df = df[~outlier_mask]
            print(f"🧹 이상치가 포함된 행 {n_outlier_rows}개를 삭제했습니다. "
                  f"(현재 {df.shape[0]}행)")
        else:
            print("➡️  이상치를 유지합니다.")

    #-----------------------------------------------------
    # 8) 정제 결과 저장 (결측치·이상치 처리가 모두 반영된 최종 데이터셋)
    #-----------------------------------------------------
    print("\n## #08. 품질 검사 결과 저장\n")
    if _prompt_yes_no("정제된 데이터셋을 Excel로 저장할까요?", default=True):
        path = f"{dataset_name}_qtcheck.xlsx"          # 용도에 맞는 파일명 자동 지정
        df.to_excel(path, index=False)
        print(f"💾 저장 완료: {path}")
    else:
        print("➡️  저장을 건너뜁니다.")

    #-----------------------------------------------------
    # 9) 완료
    #-----------------------------------------------------
    print("\n" + "=" * 60)
    print("🎉 데이터 품질 점검이 완료되었습니다.")
    print("=" * 60)

    return df