import numpy as np
from IPython.display import display
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
                # xlsxwriter는 이어 쓰기(mode='a')를 지원하지 않으므로 openpyxl을 명시
                with ExcelWriter(save_path, mode='a', engine='openpyxl') as excel_writer:
                    cdf.to_excel(excel_writer, sheet_name=col, index=True)

    return desc_df

def numerical_summary(data, columns=None, save_path=None):
    """데이터프레임의 숫자형 컬럼에 대한 요약 통계를 반환하는 함수

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
    def judge_log_transform(skew, kurt, min_value):
        right = skew >= 1 or (skew > 0.5 and kurt > 0)    # 우측 꼬리
        left = skew <= -1 or (skew < -0.5 and kurt > 0)   # 좌측 꼬리

        if right:
            if min_value > 0:   return "log"        # 0이 없으면 순수 log 가 정의된다
            if min_value == 0:  return "log1p"      # 0을 피하려고 +1 한다
            return "none"                           # 음수가 섞이면 로그 계열을 쓸 수 없다
        if left:
            return "reverse_log1p"                  # max-x >= 0 이라 항상 안전하다
        return "none"

    #-----------------------------------------------------
    # 11) 로그 변환 필요성 판정
    #-----------------------------------------------------
    desc_df['log_need'] = desc_df.apply(
        lambda row: judge_log_transform(row['skew'], row['kurt'], row['min']), axis=1)

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


def set_analysis_plan(data, target, exclude=[], ordinal_map={}, column_means={},
                      target_is_continuous=None, verbose=True):
    """분석 대상 변수를 유형별로 분류하고 EDA 범위를 정리하는 함수

    종속변수가 연속형이면 예측(회귀) 모형, 명목형이면 분류 모형으로 판정하여
    그에 맞는 분석 기법만 적용 대상으로 표시한다. 수행 내용은 다음 네 가지다.
        1. 서열척도 적용 (명목형 범주의 순서 지정)
        2. 변수 유형 분류 (종속변수·제외 대상을 뺀 명목형/연속형 목록)
        3. 컬럼 의미 딕셔너리 정리 (설명이 없는 컬럼은 컬럼명으로 채운다)
        4. EDA 범위 정리표 생성 (종속유형 x 독립유형별 분석 기법과 적용 여부)

    Args:
        data (DataFrame): 분석 대상 데이터프레임
        target (str): 종속변수 컬럼명
        exclude (list): 분석 대상에서 제외할 컬럼 리스트 (기본값: [])
        ordinal_map (dict): 서열척도를 적용할 {컬럼명: 범주 순서 리스트} (기본값: {})
        column_means (dict): 컬럼의 의미를 정리한 {컬럼명: 설명} (기본값: {})
        target_is_continuous (bool): 종속변수 연속형 여부. None이면 dtype으로 판정 (기본값: None)
        verbose (bool): 분류 결과와 EDA 범위 정리표를 출력할지 여부 (기본값: True)

    Returns:
        tuple: (DataFrame, dict)
            - DataFrame: 서열척도가 적용된 데이터프레임
            - dict: 분류 결과. 다음 키를 갖는다.
                target(str), target_type(str), model_type(str), target_groups(int or None),
                continuous(list), nominal(list), nominal_2(list), nominal_n(list),
                exclude(list), column_means(dict), plan(DataFrame)

    Raises:
        KeyError: `target` 이 데이터프레임에 없는 경우
        ValueError: `ordinal_map` 의 범주 목록이 실제 값과 일치하지 않는 경우
    """
    df = data.copy()

    #-----------------------------------------------------
    # 1) 종속변수 확인
    #-----------------------------------------------------
    if target not in df.columns:
        raise KeyError(f"종속변수 '{target}' 가 데이터프레임에 없습니다.")

    #-----------------------------------------------------
    # 2) 서열척도 적용
    #-----------------------------------------------------
    # 데이터셋에 없는 컬럼은 건너뛰므로 다른 데이터셋에서도 오류가 나지 않는다.
    ordinal_applied = []

    for col, order in ordinal_map.items():
        if col not in df.columns:
            continue

        # 범주형이 아니면 순서를 지정할 수 없으므로 먼저 변환한다
        if str(df[col].dtype) != 'category':
            df[col] = df[col].astype('category')

        # 실제 값과 지정한 순서가 다르면 누락된 범주가 결측치로 바뀌므로 미리 막는다
        actual = set(df[col].cat.categories)
        if actual != set(order):
            raise ValueError(f"'{col}' 의 서열척도 목록이 실제 값과 다릅니다. "
                             f'(실제: {sorted(actual)} / 지정: {list(order)})')

        df[col] = df[col].cat.reorder_categories(order, ordered=True)
        ordinal_applied.append(col)

    #-----------------------------------------------------
    # 3) 종속변수 유형 판정
    #-----------------------------------------------------
    # 파라미터가 주어지지 않으면 dtype 으로 판정한다 (범주형 -> 분류, 숫자형 -> 예측)
    if target_is_continuous is None:
        target_is_continuous = target not in get_categorical_column_names(df)

    # 분류 모형이라면 종속변수의 집단 수에 따라 적용 가능한 기법이 달라진다
    target_groups = None if target_is_continuous else int(df[target].nunique())

    if target_is_continuous:
        target_type = '연속형'
    elif target_groups == 2:
        target_type = '명목형(2집단)'
    else:
        target_type = '명목형(3집단+)'

    model_type = '예측(회귀)' if target_is_continuous else '분류'

    #-----------------------------------------------------
    # 4) 변수 유형 분류
    #-----------------------------------------------------
    nominal_cols = get_categorical_column_names(df)
    continuous_cols = get_number_column_names(df)

    # 종속변수와 제외 대상을 두 목록에서 함께 걷어낸다.
    # -> 어느 쪽에 속하는지 따지지 않으므로 target_is_continuous 를 직접 지정한 경우에도 안전하다
    drop = set(exclude) | {target}
    nominal_cols = [c for c in nominal_cols if c not in drop]
    continuous_cols = [c for c in continuous_cols if c not in drop]

    # 명목형은 집단 수에 따라 적용할 기법이 갈리므로 미리 나눠 둔다
    nominal_2 = [c for c in nominal_cols if df[c].nunique() == 2]
    nominal_n = [c for c in nominal_cols if df[c].nunique() >= 3]

    #-----------------------------------------------------
    # 5) 컬럼 의미 딕셔너리 정리
    #-----------------------------------------------------
    # 설명이 없는 컬럼은 컬럼명으로 채운다 (그래프 제목 등에서 KeyError 가 나지 않도록)
    means = dict(column_means)
    undefined = [c for c in df.columns if c not in means]

    for c in undefined:
        means[c] = c

    #-----------------------------------------------------
    # 6) EDA 범위 정리표 생성
    #-----------------------------------------------------
    # 종속유형 x 독립유형 조합별로 적용할 분석 기법
    methods = {
        ('연속형',       '연속형'):       '상관분석',
        ('연속형',       '명목형(2집단)'):  'T검정',
        ('연속형',       '명목형(3집단+)'): 'ANOVA',
        ('명목형(2집단)',  '연속형'):       'T검정',
        ('명목형(2집단)',  '명목형(2집단)'):  '교차분석',
        ('명목형(2집단)',  '명목형(3집단+)'): '교차분석',
        ('명목형(3집단+)', '연속형'):       'ANOVA',
        ('명목형(3집단+)', '명목형(2집단)'):  '교차분석',
        ('명목형(3집단+)', '명목형(3집단+)'): '교차분석',
    }

    # 독립유형별 변수 목록
    groups = {'연속형': continuous_cols,
              '명목형(2집단)': nominal_2,
              '명목형(3집단+)': nominal_n}

    rows = []

    for (y_type, x_type), method in methods.items():
        columns = groups[x_type]

        # 종속변수 유형이 맞아야 하고, 해당 독립유형 변수가 실제로 있어야 적용된다
        if y_type != target_type:
            applied, note = '미적용', f'종속변수 유형이 {target_type}'
        elif not columns:
            applied, note = '미적용', f'{x_type} 독립변수 없음'
        else:
            applied, note = '적용', f'{x_type} 독립변수 {len(columns)}종'

        rows.append({
            '종속유형': y_type,
            '독립유형': x_type,
            '분석 유형': method,
            '대상 변수': ', '.join(columns) if applied == '적용' else '-',
            '적용 여부': applied,
            '비고': note
        })

    plan = DataFrame(rows)

    #-----------------------------------------------------
    # 7) 분류 결과 출력
    #-----------------------------------------------------
    if verbose:
        print(f'종속변수  : {target} ({means[target]})')
        print(f'종속유형  : {target_type}' +
              (f' / {target_groups}개 집단' if target_groups else ''))
        print(f'모형유형  : {model_type}')
        print(f'명목형    : {nominal_cols}')
        print(f'연속형    : {continuous_cols}')
        print(f'제외      : {list(exclude)}')

        if ordinal_applied:
            print(f'서열척도  : {ordinal_applied}')

        if undefined:
            print(f'의미 미정의: {undefined}')

        print()
        display(plan)

    #-----------------------------------------------------
    # 8) 결과 리턴
    #-----------------------------------------------------
    info = {
        'target': target,
        'target_type': target_type,
        'model_type': model_type,
        'target_groups': target_groups,
        'continuous': continuous_cols,
        'nominal': nominal_cols,
        'nominal_2': nominal_2,
        'nominal_n': nominal_n,
        'exclude': list(exclude),
        'column_means': means,
        'plan': plan
    }

    return df, info
