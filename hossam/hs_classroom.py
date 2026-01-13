# -*- coding: utf-8 -*-
# ===================================================================
# 패키지 참조
# ===================================================================
import math
from pandas import DataFrame, qcut, concat, to_numeric
from kmodes.kmodes import KModes
from matplotlib import pyplot as plt
import seaborn as sns
from hossam.hs_util import load_data, pretty_table

# ===================================================================
# 학생들을 관심사와 성적으로 균형잡힌 조로 편성한다
# ===================================================================
def cluster_students(
    df,
    n_groups: int,
    score_cols: list = None,
    interest_col: str = None,
    max_iter: int = 200,
    score_metric: str = 'total'
) -> DataFrame:
    """학생들을 균형잡힌 조로 편성하는 함수.

    관심사 기반 1차 군집과 점수/인원 균형 조정을 통해 동질성 있고
    균형잡힌 조를 구성합니다.

    Args:
        df: 학생 정보를 담은 데이터프레임 또는 엑셀/CSV 파일 경로.
            데이터프레임의 경우: 반드시 '학생번호' 컬럼 포함.
            파일 경로의 경우: 자동으로 hs_load_data 함수를 사용하여 로드.
            interest_col이 지정된 경우 해당 컬럼 필수.
            score_cols이 지정된 경우 해당 컬럼들 필수.
        n_groups: 목표 조의 개수.
        score_cols: 성적 계산에 사용할 점수 컬럼명 리스트.
            예: ['과목1점수', '과목2점수', '과목3점수']
            None일 경우 점수 기반 균형 조정을 하지 않습니다. 기본값: None
        interest_col: 관심사 정보가 있는 컬럼명.
            None일 경우 관심사 기반 군집화를 하지 않습니다. 기본값: None
        max_iter: 균형 조정 최대 반복 횟수. 기본값: 200
        score_metric: 점수 기준 선택 ('total' 또는 'average').
            'total'이면 총점, 'average'이면 평균점수 기준. 기본값: 'total'

    Returns:
        '조' 컬럼이 추가된 데이터프레임. 관심사와 점수로 균형잡힌 조 배치 완료.

    Raises:
        ValueError: 필수 컬럼이 없거나 입력값이 유효하지 않은 경우.

    Examples:
        >>> df = read_csv('students.csv')
        >>> result = cluster_students(
        ...     df=df,
        ...     n_groups=5,
        ...     score_cols=['국어', '영어', '수학'],
        ...     interest_col='관심사'
        ... )
    """

    # 파일 경로인 경우 데이터프레임으로 로드
    if isinstance(df, str):
        df = load_data(df, info=False)

    # 입력 검증
    if df is None or len(df) == 0:
        raise ValueError("데이터프레임이 비어있습니다")

    if n_groups < 2:
        raise ValueError("조의 개수는 최소 2 이상이어야 합니다")

    if score_cols is not None and not isinstance(score_cols, list):
        raise ValueError("score_cols은 리스트여야 합니다")

    if interest_col is not None and not isinstance(interest_col, str):
        raise ValueError("interest_col은 문자열이어야 합니다")

    # 필수 컬럼 확인
    if '학생번호' not in df.columns:
        raise ValueError("데이터프레임에 '학생번호' 컬럼이 필요합니다")

    # 선택적 컬럼 확인
    if score_cols is not None:
        missing_cols = [col for col in score_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"점수 컬럼이 없습니다: {missing_cols}")

    if interest_col is not None and interest_col not in df.columns:
        raise ValueError(f"관심사 컬럼 '{interest_col}'이 없습니다")

    df = df.copy()

    # ===== 1단계: 점수 기반 처리 =====
    if score_cols is not None:
        # 총점/평균점수 계산
        df['총점'] = df[score_cols].sum(axis=1)
        df['평균점수'] = df[score_cols].mean(axis=1)

        # 사용 점수 기준 결정
        metric_col = '총점' if (score_metric or '').lower() != 'average' else '평균점수'

        # 성적사분위 분류 (선택한 기준 사용)
        df['성적사분위'] = qcut(
            df[metric_col],
            q=[0, 0.25, 0.50, 0.75, 1.0],
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'  # 중복된 값 처리
        )

        # 성적그룹 매핑
        df['성적그룹'] = df['성적사분위'].map({
            'Q1': '하', 'Q2': '중',
            'Q3': '중', 'Q4': '상'
        })

        # 극단값 분리 (선택한 기준 사용)
        Q1 = df[metric_col].quantile(0.25)
        Q3 = df[metric_col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['극단여부'] = (df[metric_col] < lower) | (df[metric_col] > upper)

        df_outlier = df[df['극단여부']].copy()
        df_main = df[~df['극단여부']].copy()
    else:
        df_main = df.copy()
        df_outlier = None

    # ===== 2단계: 조 개수 결정 =====
    main_size = len(df_main)
    actual_n_groups = min(n_groups, main_size)

    if actual_n_groups < 2:
        actual_n_groups = 2

    # ===== 3단계: 관심사 기반 1차 군집 =====
    if interest_col is not None:
        X_interest = df_main[[interest_col]].to_numpy()

        kmodes_interest = KModes(
            n_clusters=actual_n_groups,
            init='Cao',
            random_state=42,
            verbose=0
        )

        df_main['조'] = kmodes_interest.fit_predict(X_interest) + 1
    else:
        # 관심사가 없으면 단순 번호 할당
        df_main['조'] = (df_main.index % actual_n_groups) + 1

    # ===== 4단계: 조 인원 & 성적 균형 조정 =====
    if score_cols is not None:
        df_main = _balance_groups(
            df_main,
            actual_n_groups,
            score_cols,
            interest_col,
            max_iter
        )
    else:
        # score_cols가 없는 경우 최소한 인원 균형만 조정
        total = len(df_main)
        min_size = total // actual_n_groups
        max_size = min_size + 1
        df_main = _balance_group_sizes_only(df_main, actual_n_groups, min_size, max_size)

    # ===== 5단계: 극단값 포함 병합 =====
    if df_outlier is not None and len(df_outlier) > 0:
        # '조'는 숫자형 유지: 극단값은 0으로 표시
        df_outlier['조'] = 0
        result = concat([df_main, df_outlier], ignore_index=True)
    else:
        result = df_main

    # 평균점수는 이미 계산됨 (score_cols 있을 때)

    # 임시 컬럼 제거
    cols_to_drop = ['성적사분위', '성적그룹', '극단여부']
    result = result.drop(
        columns=[col for col in cols_to_drop if col in result.columns]
    )

    # 컬럼 순서 조정
    if score_cols is not None and '총점' in result.columns and '조' in result.columns and '평균점수' in result.columns:
        # 기존 컬럼 (총점, 조, 평균점수 제외)
        other_cols = [col for col in result.columns if col not in ['총점', '조', '평균점수']]
        # 원하는 순서: 총점, 조, 평균점수, 나머지
        result = result[['총점', '조', '평균점수'] + other_cols]

    # '조' 컬럼을 숫자 타입으로 강제 변환 (결과는 pandas nullable Int64)
    if '조' in result.columns:
        result['조'] = to_numeric(result['조'], errors='coerce').astype('Int64')

    return result


# ===================================================================
# 조 내 인원과 성적 균형을 반복 조정하여 최적화한다
# ===================================================================
def _balance_groups(
    df: DataFrame,
    n_groups: int,
    score_cols: list,
    interest_col: str = None,
    max_iter: int = 200
) -> DataFrame:
    """조 내 인원과 성적 균형을 조정하는 내부 함수.

    Args:
        df: '조' 컬럼이 있는 데이터프레임.
        n_groups: 조의 개수.
        score_cols: 성적 컬럼명 리스트.
        interest_col: 관심사 컬럼명. 기본값: None
        max_iter: 최대 반복 횟수. 기본값: 200

    Returns:
        균형이 조정된 데이터프레임.
    """

    df = df.copy()

    total = len(df)
    min_size = total // n_groups
    max_size = min_size + 1

    grade_levels = ['상', '중', '하']
    available_grades = [g for g in grade_levels if g in df['성적그룹'].unique()]

    if not available_grades:
        # 성적 데이터가 없으면 인원 균형만 조정
        return _balance_group_sizes_only(df, n_groups, min_size, max_size)

    grade_totals = df['성적그룹'].value_counts()
    grade_bounds = {
        g: (grade_totals[g] // n_groups, math.ceil(grade_totals[g] / n_groups))
        for g in available_grades
    }

    def dominant_interest(sub):
        """부분 데이터프레임에서 가장 많은 관심사 반환"""
        if interest_col is None or interest_col not in sub.columns:
            return None
        if sub[interest_col].mode().empty:
            return None
        return sub[interest_col].mode().iloc[0]

    for iteration in range(max_iter):
        changed = False

        # ===== 성적 균형 (우선순위: 높음) =====
        grade_counts = (
            df.groupby('조')['성적그룹']
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=available_grades, fill_value=0)
        )

        for g in sorted(df['조'].unique()):
            group = df[df['조'] == g]

            for grade in available_grades:
                count = grade_counts.loc[g, grade]
                min_g, max_g = grade_bounds[grade]

                if count <= max_g:
                    continue

                donors = group[group['성적그룹'] == grade]

                need_groups = []
                for og in sorted(df['조'].unique()):
                    if og == g:
                        continue
                    other_count = grade_counts.loc[og, grade]
                    if other_count >= min_g:
                        continue
                    other_group = df[df['조'] == og]

                    og_interest = dominant_interest(other_group)
                    need_groups.append((min_g - other_count, og, og_interest))

                need_groups.sort(reverse=True)

                for _, og, og_interest in need_groups:
                    if len(df[df['조'] == g]) - 1 < min_size:
                        continue

                    # 관심사 일치를 선호하지만, 성적 균형이 우선
                    if interest_col is not None and og_interest is not None:
                        donor_match = donors[donors[interest_col] == og_interest]
                    else:
                        donor_match = donors

                    # 관심사 일치하는 학생이 없으면 상관없이 이동
                    if len(donor_match) == 0:
                        donor_match = donors

                    if len(donor_match) == 0:
                        continue

                    donor_idx = donor_match.index[0]
                    df.loc[donor_idx, '조'] = og
                    changed = True
                    break

                if changed:
                    break

            if changed:
                break

        if changed:
            continue

        # ===== 인원 균형 (우선순위: 낮음) =====
        for g in sorted(df['조'].unique()):
            group = df[df['조'] == g]
            if len(group) > max_size:
                target = df['조'].value_counts()
                small = target[target < min_size].index
                if len(small) > 0:
                    df.loc[group.index[0], '조'] = small[0]
                    changed = True
                    break

        if not changed:
            break

    return df


# ===================================================================
# 성적 데이터가 없을 때 각 조의 인원 수만 균형조정한다
# ===================================================================
def _balance_group_sizes_only(
    df: DataFrame,
    n_groups: int,
    min_size: int,
    max_size: int
) -> DataFrame:
    """성적 데이터가 없을 때 인원만 균형조정합니다.

    Args:
        df: '조' 컬럼이 있는 데이터프레임.
        n_groups: 조의 개수.
        min_size: 조의 최소 인원.
        max_size: 조의 최대 인원.

    Returns:
        인원 균형이 조정된 데이터프레임.
    """
    df = df.copy()

    for _ in range(200):
        changed = False

        for g in sorted(df['조'].unique()):
            group = df[df['조'] == g]
            if len(group) > max_size:
                target = df['조'].value_counts()
                small = target[target < min_size].index
                if len(small) > 0:
                    df.loc[group.index[0], '조'] = small[0]
                    changed = True
                    break

        if not changed:
            break

    return df


# ===================================================================
# 조 편성 결과의 인원, 관심사, 점수 분포를 시각화한다
# ===================================================================
def report_summary(df: DataFrame, figsize: tuple = (20, 4.2), dpi: int = None) -> None:
    """조 편성 결과의 요약 통계를 시각화합니다.

    조별 인원 분포, 관심사 분포, 평균점수 분포를 나타냅니다.

    Args:
        df: cluster_students 함수의 반환 결과 데이터프레임.
        figsize: 그래프 크기 (width, height). 기본값: (20, 4.2)
        dpi: 그래프 해상도. None이면 my_dpi 사용. 기본값: None

    Examples:
        >>> from hossam.classroom import cluster_students, report_summary
        >>> df_result = cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
        >>> report_summary(df_result)
    """

    if dpi is None:
        dpi = my_dpi

    if df is None or len(df) == 0:
        print("데이터프레임이 비어있습니다")
        return

    if '조' not in df.columns:
        print("데이터프레임에 '조' 컬럼이 없습니다")
        return

    # 극단값(0조) 제외
    df = df[df['조'] != 0].copy()

    # 필요한 컬럼 확인
    has_score = '총점' in df.columns
    has_avg = '평균점수' in df.columns
    has_interest = '관심사' in df.columns

    # 혼합 타입 안전 정렬 라벨 준비
    labels = df['조'].unique().tolist()
    def _sort_key(v):
        try:
            return (0, int(v))
        except (ValueError, TypeError):
            return (1, str(v))
    ordered_labels = sorted(labels, key=_sort_key)

    # 플롯 개수 결정
    n_plots = 1  # 인원 분포는 항상 표시
    if has_interest:
        n_plots += 1
    if has_score and has_avg:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)

    # axes를 배열로 변환 (단일 플롯인 경우 대비)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # ===== 1. 조별 인원 분포 =====
    group_sizes = df['조'].value_counts()
    group_sizes = group_sizes.reindex(ordered_labels, fill_value=0)
    plot_df = DataFrame({'조': group_sizes.index, '인원': group_sizes.values})
    sns.barplot(data=plot_df, x='조', y='인원', hue='조', ax=axes[plot_idx], palette='Set2', legend=False)
    axes[plot_idx].set_title("조별 인원 분포", fontsize=12, fontweight='bold')
    axes[plot_idx].set_xlabel("조")
    axes[plot_idx].set_ylabel("인원")
    for i, v in enumerate(group_sizes.values):
        axes[plot_idx].text(i, v + 0.1, str(int(v)), ha='center', fontsize=10)
    plot_idx += 1

    # ===== 2. 조별 관심사 분포 =====
    if has_interest:
        interest_dist = (
            df.groupby(['조', '관심사'])
            .size()
            .unstack(fill_value=0)
        )
        interest_dist = interest_dist.reindex(index=ordered_labels)

        # 비율로 변환 (각 조가 100%가 되도록)
        interest_pct = interest_dist.div(interest_dist.sum(axis=1), axis=0) * 100

        # stacked bar chart로 그리기 (각 막대의 높이가 100%)
        interest_pct.plot(
            kind='bar',
            stacked=True,
            ax=axes[plot_idx],
            colormap='Set3',
            width=0.8
        )
        axes[plot_idx].set_title("조별 관심사 분포 (%)", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("조")
        axes[plot_idx].set_ylabel("비율 (%)")
        axes[plot_idx].set_xticklabels(axes[plot_idx].get_xticklabels(), rotation=0)
        axes[plot_idx].legend(title='관심사', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[plot_idx].set_ylim(0, 100)
        plot_idx += 1

    # ===== 3. 조별 평균점수 분포 =====
    if has_score and has_avg:
        avg_by_group = df.groupby('조')['평균점수'].agg(['mean', 'std']).reset_index()
        sns.barplot(
            x='조', y='mean', data=avg_by_group, hue='조',
            ax=axes[plot_idx], palette='coolwarm', legend=False,
            errorbar='sd', err_kws={'linewidth': 2}
        )
        axes[plot_idx].set_title("조별 평균점수 분포 (±1 std)", fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel("조")
        axes[plot_idx].set_ylabel("평균점수")

        # ylim 고정: 0~100
        axes[plot_idx].set_ylim(0, 100)

        for i, row in avg_by_group.iterrows():
            axes[plot_idx].text(i, row['mean'] + 2, f"{row['mean']:.1f}", ha='center', fontsize=10)
        plot_idx += 1

    plt.tight_layout()
    plt.show()


# ===================================================================
# 조별 점수 분포를 커널 밀도 추정(KDE) 그래프로 시각화한다
# ===================================================================
def report_kde(df: DataFrame, metric: str = 'average', figsize: tuple = (20, 8), dpi: int = None) -> None:
    """조별 점수 분포를 KDE(Kernel Density Estimation)로 시각화합니다.

    각 조의 점수 분포를 커널 밀도 추정으로 표시하고 평균 및 95% 신뢰구간을 나타냅니다.

    Args:
        df: cluster_students 함수의 반환 결과 데이터프레임.
        metric: 점수 기준 선택 ('total' 또는 'average').
            'total'이면 총점, 'average'이면 평균점수. 기본값: 'average'
        figsize: 그래프 크기 (width, height). 기본값: (20, 8)
        dpi: 그래프 해상도. None이면 my_dpi 사용. 기본값: None

    Examples:
        >>> from hossam.classroom import cluster_students, report_kde
        >>> df_result = cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
        >>> report_kde(df_result, metric='average')
    """

    if dpi is None:
        dpi = my_dpi

    if df is None or len(df) == 0:
        print("데이터프레임이 비어있습니다")
        return

    if '조' not in df.columns:
        print("데이터프레임에 '조' 컬럼이 없습니다")
        return

    # 필요한 컬럼 확인
    has_score = '총점' in df.columns
    has_avg = '평균점수' in df.columns

    if not has_score:
        print("점수 데이터가 없습니다")
        return

    # 혼합 타입 안전 정렬 라벨 준비
    labels = df['조'].unique().tolist()
    def _sort_key(v):
        try:
            return (0, int(v))
        except (ValueError, TypeError):
            return (1, str(v))
    ordered_labels = sorted(labels, key=_sort_key)

    n_groups = len(ordered_labels)

    # 레이아웃 결정 (3열 기준)
    cols = 3
    rows = (n_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # axes를 1D 배열로 평탄화
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    plot_idx = 0

    # 메트릭 컬럼 결정
    metric_col = '평균점수' if (metric or '').lower() == 'average' else '총점'
    if metric_col not in df.columns:
        print(f"'{metric_col}' 컬럼이 없습니다")
        return

    # ===== 각 조별 KDE =====
    for group in ordered_labels:
        group_series = df[df['조'] == group][metric_col].dropna()
        n = group_series.size
        if n == 0:
            continue

        ax_kde = axes[plot_idx]
        sns.kdeplot(group_series, ax=ax_kde, fill=True)

        # 평균선 및 95% 신뢰구간(평균에 대한) 표시
        mean_val = float(group_series.mean())
        std_val = float(group_series.std(ddof=1)) if n > 1 else float('nan')
        se = (std_val / (n ** 0.5)) if (n > 1 and std_val == std_val) else None
        if se is not None:
            ci_low = mean_val - 1.96 * se
            ci_high = mean_val + 1.96 * se
            ax_kde.axvspan(ci_low, ci_high, color='red', alpha=0.12, label='95% CI', zorder=9)
        # 평균 세로선 (최상위로)
        ax_kde.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label='Mean', zorder=10)

        # 제목/레이블
        ax_kde.set_title(f"{group}조 {metric_col} KDE", fontsize=12, fontweight='bold')
        ax_kde.set_xlabel(metric_col)
        ax_kde.set_ylabel("밀도")

        # 범례 표시 (Mean/CI가 있으면 노출)
        handles, labels_ = ax_kde.get_legend_handles_labels()
        if handles:
            ax_kde.legend(fontsize=9)

        plot_idx += 1

    # 불필요한 서브플롯 제거
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# ===================================================================
# 조별로 학생 목록과 평균 점수를 요약하여 데이터프레임으로 반환한다
# ===================================================================
def group_summary(df: DataFrame, name_col: str = '학생번호') -> DataFrame:
    """조별로 학생 목록과 평균 점수를 요약합니다.

    Args:
        df: cluster_students 함수의 반환 결과 데이터프레임.
            '조' 컬럼이 필수로 포함되어야 함.
        name_col: 학생 이름이 들어있는 컬럼명. 기본값: '학생번호'

    Returns:
        조별 요약 정보가 담긴 데이터프레임.
        컬럼: '조', '학생', '총점평균', '평균점수평균'

    Examples:
        >>> from hossam.classroom import cluster_students, group_summary
        >>> df_result = cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
        >>> summary = group_summary(df_result, name_col='이름')
        >>> print(summary)
    """

    if df is None or len(df) == 0:
        print("데이터프레임이 비어있습니다")
        return DataFrame()

    if '조' not in df.columns:
        print("데이터프레임에 '조' 컬럼이 없습니다")
        return DataFrame()

    if name_col not in df.columns:
        print(f"데이터프레임에 '{name_col}' 컬럼이 없습니다")
        return DataFrame()

    # 혼합 타입 안전 정렬
    def _sort_key(v):
        try:
            return (0, int(v))
        except (ValueError, TypeError):
            return (1, str(v))

    # 조별로 그룹화하여 정보 수집
    result_data = []

    for group_name in sorted(df['조'].unique(), key=_sort_key):
        group_df = df[df['조'] == group_name]

        # 학생 이름들을 콤마로 구분하여 결합
        students = ', '.join(group_df[name_col].astype(str).tolist())

        # 총점과 평균점수가 있는지 확인하고 평균 계산
        row = {'조': group_name, '학생': students}

        if '총점' in df.columns:
            row['총점평균'] = round(group_df['총점'].mean(), 2)

        if '평균점수' in df.columns:
            row['평균점수평균'] = round(group_df['평균점수'].mean(), 2)

        result_data.append(row)

    result_df = DataFrame(result_data)

    return result_df


# ===================================================================
# 학생 조 편성부터 시각화까지의 전체 분석 프로세스를 일괄 실행한다
# ===================================================================
def analyze_classroom(
    df,
    n_groups: int,
    score_cols: list = None,
    interest_col: str = None,
    max_iter: int = 200,
    score_metric: str = 'average',
    name_col: str = '학생번호',
    show_summary: bool = True,
    show_kde: bool = True
) -> DataFrame:
    """학생 조 편성부터 시각화까지 전체 프로세스를 일괄 실행합니다.

    다음 순서로 실행됩니다:
    1. cluster_students: 학생들을 균형잡힌 조로 편성
    2. group_summary: 조별 학생 목록과 평균 점수 요약
    3. report_summary: 조 편성 결과 요약 시각화 (선택적)
    4. report_kde: 조별 점수 분포 KDE 시각화 (선택적)

    Args:
        df: 학생 정보를 담은 데이터프레임 또는 파일 경로.
        n_groups: 목표 조의 개수.
        score_cols: 성적 계산에 사용할 점수 컬럼명 리스트. 기본값: None
        interest_col: 관심사 정보가 있는 컬럼명. 기본값: None
        max_iter: 균형 조정 최대 반복 횟수. 기본값: 200
        score_metric: 점수 기준 선택 ('total' 또는 'average'). 기본값: 'average'
        name_col: 학생 이름 컬럼명. 기본값: '학생번호'
        show_summary: 요약 시각화 표시 여부. 기본값: True
        show_kde: KDE 시각화 표시 여부. 기본값: True

    Returns:
        조별 요약 정보 (group_summary의 결과).

    Examples:
        >>> from hossam.classroom import analyze_classroom
        >>> summary = analyze_classroom(
        ...     df='students.csv',
        ...     n_groups=5,
        ...     score_cols=['국어', '영어', '수학'],
        ...     interest_col='관심사',
        ...     name_col='이름'
        ... )
        >>> print(summary)
    """

    print("=" * 60)
    print("1. 학생 조 편성 중...")
    print("=" * 60)

    # 1. 조 편성
    df_result = cluster_students(
        df=df,
        n_groups=n_groups,
        score_cols=score_cols,
        interest_col=interest_col,
        max_iter=max_iter,
        score_metric=score_metric
    )

    print(f"\n✓ 조 편성 완료: {len(df_result)}명의 학생을 {n_groups}개 조로 배정\n")

    print("=" * 60)
    print("2. 조별 요약 생성 중...")
    print("=" * 60)

    # 2. 조별 요약
    summary = group_summary(df_result, name_col=name_col)
    print("\n✓ 조별 요약:")
    pretty_table(summary, tablefmt="pretty")
    print()

    # 3. 요약 시각화
    if show_summary:
        print("=" * 60)
        print("3. 조 편성 요약 시각화 중...")
        print("=" * 60)
        report_summary(df_result)
        print("\n✓ 요약 시각화 완료\n")

    # 4. KDE 시각화
    if show_kde:
        print("=" * 60)
        print(f"4. 조별 {score_metric} KDE 시각화 중...")
        print("=" * 60)
        report_kde(df_result, metric=score_metric)
        print("\n✓ KDE 시각화 완료\n")

    print("=" * 60)
    print("전체 분석 완료!")
    print("=" * 60)

    return summary
