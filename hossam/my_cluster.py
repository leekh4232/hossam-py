import numpy as np
import seaborn as sb
from pandas import DataFrame, crosstab
from IPython.display import display

# K-평균 군집분석, 계층적(병합형) 군집분석, 밀도기반 군집분석
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# 계층적 군집이 합쳐온 과정을 나무 모양으로 그려주는 함수
from scipy.cluster.hierarchy import dendrogram

# 각 데이터에서 가까운 순서로 k개의 이웃을 찾아주는 클래스 (k-distance plot 에 사용)
from sklearn.neighbors import NearestNeighbors

# 군집 품질 평가 지표
from sklearn.metrics import silhouette_samples, silhouette_score

# 곡선이 꺾이는 지점(엘보우 포인트)을 계산해주는 패키지
from kneed import KneeLocator

# 정규성 검정 (대표값으로 평균을 쓸지 중앙값을 쓸지 판단한다)
from scipy.stats import normaltest

from . import RANDOM_STATE
from . import my_plot
from . import my_prep
from . import my_qtcheck


# ===================================================================
# K-평균 군집분석 — 데이터를 k개의 그룹으로 나누고 결과를 시각화한다
# ===================================================================
def kmeans(data, k, columns=None, scaling='standard', cluster_name='그룹번호',
           random_state=RANDOM_STATE, n_init=10, verbose=True, plot=True, x=None, y=None,
           title=None, palette='tab10', size=100, edgecolor='#ffffff', linewidth=1.5, alpha=1, outline=True, center=True, center_marker='X', center_size=150, 
           center_color='#ff0000',center_edgecolor='#000000', center_linewidth=1.5,
           equal_scale=False, width=1280, height=640, save_path=None, ax=None):
    """데이터를 k개의 군집으로 나누고, 군집 결과와 중심점을 시각화하는 함수

    K-Means는 데이터 사이의 '거리'로 그룹을 나누므로, 변수마다 값의 범위가 다르면
    범위가 큰 변수 하나가 거리를 지배하게 된다. 그래서 스케일링을 기본으로 수행한다.

    Args (기본값은 위의 함수 정의 참고):
        data, k: 군집화할 데이터프레임과 나눌 군집의 개수
        columns, cluster_name, random_state: 사용할 컬럼(None이면 수치형 전체),
            군집 번호를 저장할 컬럼명, 중심점의 초기 위치를 결정하는 랜덤시드
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 중 가장 좋은 결과를 채택한다
            (sklearn의 기본값 'auto'는  1회만 시도하므로 초기값 운에 따라 나쁜 국소최적해에 빠질 수 있다.)
        scaling: 스케일러 이름('standard'/'minmax'/'robust'/'maxabs', None이면 원본 값)
        verbose: 스케일링 전후의 값의 범위를 출력할지 여부
        plot, x, y, title: 시각화 여부, 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개),
            그래프 제목(None이면 군집 개수를 포함하여 자동 생성)
        palette, size, edgecolor, linewidth, alpha: 데이터 포인트의 색상 팔레트,
            마커 크기, 테두리 색상, 테두리 두께, 투명도
        outline: 군집의 외곽선(ConvexHull)을 표시할지 여부
        center: 모델이 찾은 중심점을 표시할지 여부
        center_marker, center_size, center_color, center_edgecolor, center_linewidth:
            중심점의 마커 모양, 크기, 색상, 테두리 색상, 테두리 두께
        equal_scale: x축의 범위를 y축과 동일하게 맞출지 여부
            (모델이 실제로 본 거리 관계를 그대로 확인할 때 사용한다)
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (estimator, df, center_df) — 학습이 완료된 모델,
            군집 번호 컬럼이 추가된 데이터(스케일링 적용 후),
            각 군집의 중심점 좌표(컬럼명은 대상 컬럼과 동일)
    """
    # --- 1) 군집화에 사용할 컬럼 결정 ---
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # --- 2) 스케일링 적용 ---
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # --- 3) 모델 생성 및 학습 (중심점을 찾는 과정) ---
    # n_init: 시작 위치를 여러 번 바꿔 시도한 뒤 가장 좋은 결과를 채택한다
    estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    estimator.fit(df)

    # --- 4) 각 데이터가 몇 번 그룹인지 예측하여 컬럼으로 추가 ---
    df[cluster_name] = estimator.predict(df)

    # --- 5) 모델이 찾은 중심점을 데이터프레임으로 구성 ---
    center_df = DataFrame(estimator.cluster_centers_, columns=columns)

    # --- 6) 군집 결과 시각화 ---
    if plot:
        # 축으로 사용할 컬럼 결정 (지정이 없으면 대상 컬럼의 앞에서 두 개)
        if x is None:       x = columns[0]
        if y is None:       y = columns[1]

        # 제목을 지정하지 않은 경우 군집 개수를 포함한 제목을 자동으로 생성
        if title is None:   title = f'K-Means 군집 결과 (k={k})'

        # 그래프 초기화 (ax를 전달받은 경우에는 그 위에 겹쳐 그린다)
        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                   xlabel=x, ylabel=y)

        # 군집별 산점도 (outline=True이면 각 군집의 외곽선까지 표시)
        my_plot.scatterplot(data=df, x=x, y=y, hue=cluster_name,
                            palette=palette, size=size, edgecolor=edgecolor,
                            linewidth=linewidth, alpha=alpha, outline=outline, ax=ax)

        # 모델이 찾은 중심점을 덧그리기 (중심점의 x좌표에 대한 y좌표 산점도)
        if center:
            my_plot.scatterplot(data=center_df, x=x, y=y,
                                marker=center_marker, size=center_size,
                                color=center_color, edgecolor=center_edgecolor,
                                linewidth=center_linewidth, outline=False, ax=ax)

        # x축의 범위를 y축과 동일하게 맞춰 실제 거리 관계를 확인
        if equal_scale:
            ax.set_xlim(ax.get_ylim())

        # 그래프 표시 (ax를 전달받은 경우에는 호출한 쪽에서 표시한다)
        if fig is not None:
            my_plot.show(save_path=save_path)

    # --- 7) 모델, 군집 결과, 중심점 좌표 반환 ---
    return estimator, df, center_df


# ===================================================================
# 실루엣 시각화 — 군집이 잘 나뉘었는지를 데이터 하나하나 단위로 확인한다
# 원본: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# ===================================================================
def visualize_silhouette(estimator, data, columns=None, cluster_name='그룹번호',
                         x=None, y=None, title=None, palette='tab10',
                         size=50, edgecolor='#ffffff', linewidth=1.5, alpha=1, outline=True,
                         center=True, center_marker='X', center_size=150,
                         center_color='#ff0000', center_edgecolor='#000000', center_linewidth=1.5,
                         width=800, height=580, save_path=None):
    """군집별 실루엣 계수 막대(왼쪽)와 군집 산점도(오른쪽)를 나란히 그리는 함수

    실루엣 계수는 "내가 속한 군집의 이웃들과는 얼마나 가깝고, 가장 가까운 다른 군집과는
    얼마나 먼가"를 데이터 하나마다 -1~1 사이의 값으로 나타낸 것이다. 평균값 하나만 보면
    "군집 하나가 통째로 망가진 경우"와 "모든 군집이 고만고만한 경우"를 구분할 수 없으므로,
    군집별 분포를 막대로 펼쳐서 함께 확인한다.

    Args (기본값은 위의 함수 정의 참고):
        estimator: 학습이 끝난 군집 모델 (KMeans, AgglomerativeClustering 등)
        data (DataFrame): 군집에 사용한 데이터
            (kmeans 함수의 반환값처럼 군집 번호 컬럼이 포함되어 있어도 된다)
        columns, cluster_name: 실루엣 계산에 사용할 컬럼(None이면 군집 번호 컬럼을 제외한
            수치형 전체), 데이터에 포함된 군집 번호 컬럼의 이름
        x, y, title: 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개),
            그래프 상단 제목(None이면 군집 개수와 실루엣 스코어로 자동 생성)
        palette, size, edgecolor, linewidth, alpha, outline: 산점도의 색상 팔레트,
            마커 크기, 테두리 색상, 테두리 두께, 투명도, 외곽선 표시 여부
            (왼쪽 막대의 색상도 같은 팔레트를 사용하므로 두 그래프의 군집 색이 일치한다)
        center, center_marker, center_size, center_color, center_edgecolor, center_linewidth:
            중심점 표시 여부와 마커 모양·크기·색상·테두리 색상·테두리 두께
            (중심점을 제공하는 모델에서만 표시된다)
        width, height, save_path: 그래프 한 칸의 가로·세로 픽셀, 저장 경로
    """
    labels = estimator.labels_

    # --- 1) 실루엣 계산에 사용할 컬럼 결정 ---
    # 군집 번호 컬럼이 섞여 있으면 그 자체가 거리 계산에 반영되므로 반드시 제외한다
    if columns is None:
        columns = [c for c in data.select_dtypes(include='number').columns if c != cluster_name]

    df = data[columns]

    # --- 2) 노이즈(-1)를 제외한 실제 군집 번호 확인 ---
    # 실루엣 계수는 "다른 군집과의 거리"가 있어야 정의되므로 군집이 2개는 되어야 한다
    cluster_ids = sorted([c for c in set(labels) if c != -1])
    k = len(cluster_ids)

    if k < 2:
        print("군집이 2개 미만이므로 실루엣 계수를 계산할 수 없습니다.")
        return

    # --- 3) 전체 평균과 데이터별 실루엣 계수 계산 ---
    sil_avg = silhouette_score(X=df, labels=labels)
    sil_values = silhouette_samples(X=df, labels=labels)

    if title is None:
        title = f'군집수={k}, 실루엣 스코어={sil_avg:.3f}'

    fig, (ax1, ax2) = my_plot.init(width=width, height=height, rows=1, cols=2, title=title)

    # 전체 제목이 각 그래프의 소제목과 겹치지 않도록 위쪽으로 밀어낸다
    fig.suptitle(title, fontsize=24, fontweight=500, y=1.0)

    # --- 4) 왼쪽: 군집별 실루엣 계수 막대 ---
    # 오른쪽 산점도와 같은 팔레트를 사용해 두 그래프의 군집 색을 일치시킨다
    colors = sb.color_palette(palette, k)

    ax1.set_title('군집별 실루엣 계수', fontsize=16, pad=10)
    ax1.set_xlabel('실루엣 계수')
    ax1.set_ylabel('군집 번호')
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (k + 1) * 10])
    ax1.set_yticks([])

    y_lower = 10

    for i, c in enumerate(cluster_ids):
        # 해당 군집에 속한 데이터의 실루엣 계수를 오름차순으로 정렬 (칼날 모양의 막대가 된다)
        ith_values = np.sort(sil_values[labels == c])
        y_upper = y_lower + len(ith_values)

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * len(ith_values), str(c))

        # 다음 군집의 막대를 10칸 띄워서 그리기 위한 시작 위치
        y_lower = y_upper + 10

    # 전체 평균 실루엣 계수를 빨간 점선으로 표시 (이 선을 넘지 못하는 군집이 문제 군집)
    ax1.axvline(x=sil_avg, color='red', linestyle='--')

    # --- 5) 오른쪽: 군집 산점도 ---
    if x is None:       x = columns[0]
    if y is None:       y = columns[1]

    vdf = df.copy()
    vdf[cluster_name] = labels

    my_plot.scatterplot(data=vdf, x=x, y=y, hue=cluster_name, palette=palette,
                        size=size, edgecolor=edgecolor, linewidth=linewidth,
                        alpha=alpha, outline=outline, ax=ax2)

    ax2.set_title('군집 산점도', fontsize=16, pad=10)
    ax2.set_xlabel(x)
    ax2.set_ylabel(y)

    # 중심점을 제공하는 모델(KMeans 등)이라면 중심점도 함께 표시
    centers = getattr(estimator, 'cluster_centers_', None)

    if center and centers is not None:
        center_df = DataFrame(centers, columns=columns)
        my_plot.scatterplot(data=center_df, x=x, y=y,
                            marker=center_marker, size=center_size,
                            color=center_color, edgecolor=center_edgecolor,
                            linewidth=center_linewidth, outline=False, ax=ax2)

    # 범례를 그래프 바깥 오른쪽 위에 고정한다
    # (자동 배치에 맡기면 데이터가 없는 자리를 찾아다니느라 K마다 위치가 달라진다)
    handles, texts = ax2.get_legend_handles_labels()

    if handles:
        ax2.legend(handles, texts, title=cluster_name,
                   loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # --- 6) 그래프 표시 ---
    # 전체 제목이 들어갈 위쪽 공간(7%)을 미리 비워 둔 뒤 표시한다
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    my_plot.show(save_path=save_path)


# ===================================================================
# 최적 k 탐색 — 세 함수가 공통으로 사용하는 준비 작업
# ===================================================================
def _prepare_k_search(data, klist, columns, scaling, verbose):
    """최적 k 탐색 함수들이 공통으로 수행하는 컬럼 선택·스케일링·k 목록 정리

    여러 k를 같은 데이터로 비교해야 하므로 스케일링은 반복문 밖에서 한 번만 수행한다.

    Args:
        data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, verbose: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 스케일링 전후의 값의 범위 출력 여부

    Returns:
        tuple: (df, klist) — 스케일링이 끝난 데이터, 2 이상만 남긴 k 목록
    """

    # 대상 컬럼이 없다면 숫자형태의 컬럼만 추려낸다.
    # --> 거리기반 알고리즘이므로 문자열 형태는 처리하지 못한다.
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # 스케일링을 k마다 반복하면 매번 같은 계산을 다시 하는 셈이므로 여기서 한 번만 처리한다
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # 실루엣 계수는 "다른 군집과의 거리"가 있어야 정의되므로 k는 2부터 확인한다
    klist = list(range(2, 11)) if klist is None else [k for k in klist if k >= 2]

    # 스케일링 후의 데이터와 2 이상만 남긴 k 목록을 반환한다
    return df, klist


# ===================================================================
# 엘보우 포인트 — 이너셔의 감소폭이 꺾이는 지점으로 k의 후보를 좁힌다
# ===================================================================
def best_k_elbow(data, klist=None, columns=None, scaling='standard',
                 sensitivity=0.01, random_state=RANDOM_STATE, n_init=10, verbose=True,
                 plot=True, title=None, color='#1f77b4', marker='o', linestyle=':',
                 best_color='#ff0000', width=1280, height=640, save_path=None, ax=None):
    """이너셔의 감소폭이 꺾이는 지점(엘보우 포인트)을 찾아 최적의 k를 추정하는 함수

    이너셔(군집 중심까지 거리의 제곱합)는 k가 커지면 항상 줄어들기 때문에 가장 작은 값을
    고르는 것은 의미가 없다. 대신 "k를 하나 더 늘려도 이제 별 이득이 없어지는" 지점을 찾는다.
    꺾이는 지점은 KneeLocator가 계산하며, 이너셔는 뭉침만 보는 지표이므로
    이 결과는 최종 답이 아니라 후보다.

    Args (기본값은 위의 함수 정의 참고):
        data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
        sensitivity: KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다
        verbose, plot, title: 계산 결과 출력 여부, 시각화 여부,
            그래프 제목(None이면 자동 생성)
        color, marker, linestyle, best_color: 이너셔 선의 색상·마커 모양·선 스타일,
            엘보우 포인트를 표시할 세로선의 색상
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (best_k, result_df) — 엘보우 포인트, k별 이너셔와 감소량이 담긴 데이터프레임
    """
    # --- 0) 공통 준비 작업 ---
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # --- 1) k를 늘려 가며 이너셔 수집 ---
    inertia = []

    for k in klist:
        estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        estimator.fit(df)
        inertia.append(estimator.inertia_)

    # --- 2) k가 1 늘어날 때마다 이너셔가 얼마나 줄었는지 계산 ---
    # np.diff()는 바로 앞 값과의 차이를 구해준다 (이너셔는 줄어들므로 음수 → 절대값으로 변환)
    drop = np.abs(np.diff(inertia))

    # --- 3) 결과 정리 ---
    # 감소량은 "앞의 k에서 넘어올 때"의 값이므로 첫 행은 비워 둔다
    result_df = DataFrame({
        'k': klist,
        '이너셔': np.round(inertia, 4),
        '감소량': np.round(np.insert(drop, 0, np.nan), 4),
    })

    # --- 4) 엘보우 포인트 찾기 ---
    # convex(아래로 볼록) + decreasing(우하향)은 이너셔 곡선의 모양이다
    kl = KneeLocator(klist, inertia, curve='convex',
                     direction='decreasing', S=sensitivity)
    best_k = kl.elbow

    # 곡선이 거의 직선이면 꺾이는 지점이 없어 None이 나온다
    if best_k is None:
        print("곡선이 완만해 꺾이는 지점을 찾지 못했습니다. "
              "sensitivity 를 낮추거나 klist 의 범위를 넓혀 다시 시도해 보세요.")
        return None, result_df

    if verbose:
        print(f"[엘보우] 최적의 k = {best_k}")

    # --- 5) 시각화 ---
    if plot:
        if title is None:   title = '엘보우 포인트'

        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                   xlabel='군집 개수 (k)', ylabel='이너셔')

        my_plot.lineplot(x=klist, y=inertia, color=color, marker=marker,
                         linestyle=linestyle, ax=ax)

        # 엘보우 포인트를 세로선으로 표시
        ax.axvline(x=best_k, color=best_color, linestyle=':')
        ax.text(best_k + 0.1, max(inertia) * 0.9, f'Best k = {best_k}', color=best_color)

        if fig is not None:
            my_plot.show(save_path=save_path)

    return best_k, result_df


# ===================================================================
# 실루엣 면적 — 막대의 넓이와 두께가 고른 k를 최적으로 판단한다
# ===================================================================
def best_k_silhouette(data, klist=None, columns=None, scaling='standard',
                      random_state=RANDOM_STATE, n_init=10, verbose=True, plot=True,
                      plot_each=False, title=None, palette='tab10',
                      width=800, height=580, save_path=None):
    """실루엣 막대의 면적과 두께를 함께 평가해 최적의 k를 찾는 함수

    Args (기본값은 위의 함수 정의 참고):
        data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
        verbose, plot, plot_each: 계산 결과 출력 여부,
            선택된 k의 실루엣 막대·산점도(visualize_silhouette)를 그릴지 여부,
            선택된 k뿐 아니라 모든 k에 대해 같은 그림을 그릴지 여부
        title, palette: 그래프 제목(None이면 군집 개수와 스코어로 자동 생성),
            산점도와 막대에 함께 쓰이는 색상 팔레트
        width, height, save_path: 그래프 한 칸의 가로·세로 픽셀, 저장 경로

    Returns:
        tuple: (best_k, result_df) — 균형지수가 가장 높은 k,
            k별 스코어·전체면적·두께비·최소상대면적·균형지수가 담긴 데이터프레임
    """
    # -- 0) 공통 준비 작업 ---
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # --- 1) k를 늘려 가며 실루엣 지표 수집 ---
    items = []

    # 아래에서 선택된 k의 그림을 그릴 때 다시 학습시키지 않도록 모델을 보관해 둔다
    estimators = {}

    for k in klist:
        # 1-1) 모델 학습
        estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        estimator.fit(df)
        estimators[k] = estimator
        cluster = estimator.labels_

        # 1-2) 실루엣 스코어 계산 (군집 전체의 평균)
        # 데이터 하나하나의 실루엣 계수 (= 띠 하나의 가로 길이)
        s_samples = silhouette_samples(X=df, labels=cluster)

        # 1-3) 군집별 면적과 두께 계산
        # 군집별 면적(계수의 합)과 두께(데이터 개수)
        area = np.array([s_samples[cluster == c].sum() for c in range(k)])
        size = np.array([(cluster == c).sum() for c in range(k)])

        score = s_samples.mean()            # 계수의 평균 = 실루엣 스코어
        total_area = area.sum()             # 계수의 합 = 데이터 개수 × 스코어
        thickness = size.max() / size.min()

        # 면적의 합이 0 이하라면 군집이 사실상 무너진 상태이므로 균형을 0으로 처리한다
        min_area = (area.min() / total_area) * k if total_area > 0 else 0.0

        # 1-4) 균형지수 계산 (스코어 × 최소상대면적 ÷ 두께비)
        items.append({
            'k': k,
            '스코어': round(score, 4),
            '전체면적': round(total_area, 4),
            '두께비': round(thickness, 4),
            '최소상대면적': round(min_area, 4),
            '균형지수': round(score * min_area / thickness, 4),
        })

        # 1-5) k마다 실루엣 막대와 산점도를 확인 (눈으로 면적·두께를 함께 보기 위한 옵션)
        # k마다 실루엣 막대와 산점도를 확인 (눈으로 면적·두께를 함께 보기 위한 옵션)
        if plot_each:
            visualize_silhouette(estimator, df, palette=palette,
                                 width=width, height=height)

    # k별 지표를 데이터프레임으로 정리
    result_df = DataFrame(items)

    # --- 2) 균형지수가 가장 높은 k 선택 ---
    best = result_df.loc[result_df['균형지수'].idxmax()]
    best_k = int(best['k'])

    if verbose:
        print(f"[실루엣] 균형지수 최대 → 최적의 k = {best_k} "
              f"(스코어 {best['스코어']}, 두께비 {best['두께비']}, 균형지수 {best['균형지수']})")

    # --- 3) 시각화 ---
    # 선택된 k가 왜 뽑혔는지는 지표 숫자보다 막대의 면적과 두께를 직접 보는 편이 빠르다
    # (plot_each 로 이미 모든 k를 그렸다면 같은 그림을 두 번 그리지 않는다)
    if plot and not plot_each:
        visualize_silhouette(estimators[best_k], df, title=title, palette=palette,
                             width=width, height=height, save_path=save_path)

    return best_k, result_df


# ===================================================================
# 최적 k 결정 — 엘보우와 실루엣을 함께 확인해 최종 k를 고른다
# ===================================================================
def best_k(data, klist=None, columns=None, scaling='standard',
           sensitivity=0.01, random_state=RANDOM_STATE, n_init=10, verbose=True,
           plot=True, plot_each=False, title=None, width=1280, height=640):
    """엘보우 포인트와 실루엣 균형지수를 함께 확인해 최종 k를 결정하는 함수

    Args (기본값은 위의 함수 정의 참고):
        data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
        sensitivity: KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다
        verbose, plot, plot_each, title: 판단 과정 출력 여부, 시각화 여부,
            k마다 실루엣 막대·산점도를 그릴지 여부, 그래프 제목(None이면 자동 생성)
        width, height: 캔버스 한 칸의 가로·세로 픽셀

    Returns:
        tuple: (best_k, result_df) — 최종 선택한 k,
            k별 이너셔·감소량·실루엣 지표를 한 표로 합친 데이터프레임
    """
    # --- 1) 스케일링을 한 번만 수행한 뒤 두 함수에 같은 데이터를 넘긴다 ---
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # --- 2) 두 지표를 각각 계산 ---
    elbow_k, elbow_df = best_k_elbow(df, klist=klist, scaling=None,
                                     sensitivity=sensitivity, random_state=random_state,
                                     n_init=n_init, verbose=False, plot=plot, width=width, height=height)

    sil_k, sil_df = best_k_silhouette(df, klist=klist, scaling=None,
                                      random_state=random_state, n_init=n_init,
                                      verbose=False, plot=plot, plot_each=plot_each, width=width, height=height)

    # --- 3) 두 결과를 한 표로 합치기 ---
    result_df = elbow_df.merge(sil_df, on='k')

    # --- 4) 최종 판단 ---
    final_k = sil_k # 엘보우는 뭉침만 보므로 후보 제시용, 최종 결정은 분리와 균형까지 보는 실루엣이 맡는다

    if verbose:
        # 두 지표가 가리키는 k와 지표값을 함께 출력해 비교한다
        sil_row = sil_df[sil_df['k'] == sil_k].iloc[0]
        elbow_row = sil_df[sil_df['k'] == elbow_k]

        print(f"[1단계] 엘보우 포인트   : k = {elbow_k}  (뭉침만 확인 → 후보)")
        print(f"[2단계] 실루엣 균형지수 : k = {sil_k}  (뭉침 + 분리 + 크기 균형)")
        print('-' * 60)

        # 두 지표가 같은 k를 가리키는지 여부에 따라 최종 선택을 안내한다
        if elbow_k == sil_k:
            print("두 지표가 같은 k를 가리키므로 그대로 확정합니다.")
        else:
            print("두 지표가 다른 k를 가리킵니다.")

            if not elbow_row.empty:
                print(f"  · k = {elbow_k} : 균형지수 {elbow_row.iloc[0]['균형지수']} "
                      f"(스코어 {elbow_row.iloc[0]['스코어']}, 두께비 {elbow_row.iloc[0]['두께비']})")

            print(f"  · k = {sil_k} : 균형지수 {sil_row['균형지수']} "
                  f"(스코어 {sil_row['스코어']}, 두께비 {sil_row['두께비']})")

        print('-' * 60)
        print(f">>> 최종 선택: k = {final_k}")
        print("    (지표는 후보를 좁혀줄 뿐이므로, 목적과 도메인 지식으로 한 번 더 확인할 것)")

    return final_k, result_df


# ===================================================================
# 덴드로그램 재료 — 계층적 군집이 합쳐온 과정을 그림용 표로 바꾼다
# 원본: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# ===================================================================
def dendrogram_source(estimator):
    """계층적 군집 모델을 scipy의 dendrogram()이 읽는 linkage 행렬로 변환하는 함수

    sklearn의 AgglomerativeClustering은 "무엇과 무엇을 합쳤는지(children_)"와
    "얼마나 먼 거리에서 합쳤는지(distances_)"를 따로 들고 있을 뿐, 덴드로그램을
    바로 그려주지는 않는다. 그림을 그리려면 여기에 "그 가지 아래에 원본 데이터가
    몇 개나 들어 있는지"를 더한 네 칸짜리 표가 필요하므로 그 개수를 세어 붙인다.

    Args:
        estimator: compute_distances=True 로 학습한 AgglomerativeClustering 모델

    Returns:
        ndarray: [자식1, 자식2, 병합거리, 포함 샘플수] 형태의 linkage 행렬
    """
    # --- 1) 병합 거리가 기록되어 있는지 확인 ---
    # compute_distances=True 없이 학습하면 거리를 남기지 않아 높이를 그릴 수 없다
    if getattr(estimator, 'distances_', None) is None:
        raise ValueError("덴드로그램을 그리려면 "
                         "AgglomerativeClustering(compute_distances=True) 로 학습해야 합니다.")

    # --- 2) 병합 단계마다 그 아래에 몇 개의 원본 데이터가 들어있는지 센다 ---
    counts = np.zeros(estimator.children_.shape[0])
    n_samples = len(estimator.labels_)

    for i, merge in enumerate(estimator.children_):
        current_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1                              # 아직 병합되지 않은 개별 데이터
            else:
                current_count += counts[child_idx - n_samples]  # 이미 병합된 군집

        counts[i] = current_count

    # --- 3) 네 칸(자식1, 자식2, 병합거리, 포함 샘플수)을 옆으로 이어 붙인다 ---
    return np.column_stack([estimator.children_,
                            estimator.distances_, counts]).astype(float)


# ===================================================================
# 자른 높이 — 학습이 끝난 모델에서 "나무를 어느 높이에서 잘랐는가"를 되돌린다
# ===================================================================
def _cut_height(estimator):
    """계층적 군집 모델이 나무를 자른 높이(병합 거리)를 계산하는 내부 공용 함수

    Args:
        estimator: 학습이 끝난 AgglomerativeClustering 모델

    Returns:
        float: 자른 높이 (자를 곳이 없으면 None)
    """
    # --- 1) 거리 기준으로 잘랐다면 그 값이 곧 자른 높이 ---
    if getattr(estimator, 'distance_threshold', None) is not None:
        return estimator.distance_threshold

    # --- 2) 군집 수로 잘랐다면 병합 거리 사이에서 되돌린다 ---
    # (데이터 n개를 k개로 만들려면 n-k번 합쳐야 한다)
    distances = np.sort(estimator.distances_)
    n_merged = len(estimator.labels_) - estimator.n_clusters_

    if n_merged >= len(distances):  # 하나로 다 합친 경우에는 나무의 꼭대기 위쪽이 자른 높이
        return distances[-1] * 1.05

    if n_merged < 1:    # 아무것도 합치지 않은 경우에는 자를 높이가 없다
        return None

    return (distances[n_merged - 1] + distances[n_merged]) / 2


# ===================================================================
# 덴드로그램 시각화 — 합쳐온 과정을 나무 그림으로 그리고, 자른 높이를 표시한다
# ===================================================================
def plot_dendrogram(estimator, cut_height=None, title=None,
                         p=30, truncate_mode='lastp', leaf_rotation=0, leaf_font_size=8,
                         count_sort='ascending', cut_line=True, cut_color='#ff0000',
                         xlabel=None, ylabel='병합 거리',
                         width=1280, height=640, save_path=None, ax=None):
    """계층적 군집 모델이 합쳐온 과정을 덴드로그램으로 그리는 함수

    Args (기본값은 위의 함수 정의 참고):
        estimator: compute_distances=True 로 학습한 AgglomerativeClustering 모델
        cut_height: 나무를 자른 높이(병합 거리). 이 높이 아래의 가지를 군집별로 다른
            색으로 칠하고 가로선으로 표시한다. None이면 모델이 실제로 자른 높이를
            직접 계산해서 쓰므로, 보통은 지정할 필요가 없다
        title, xlabel, ylabel: 그래프 제목(None이면 '덴드로그램'), x축·y축 이름
            (xlabel이 None이면 가지를 묶어 그리는지에 따라 자동으로 정한다)
        p, truncate_mode: 표시할 가지의 개수와 생략 방식
            ('lastp'=마지막 p개의 가지만, None이면 데이터 전체를 그린다)
        leaf_rotation, leaf_font_size, count_sort: 아래쪽 눈금의 회전 각도, 글자 크기,
            가지의 정렬 기준('ascending'이면 작은 덩어리를 왼쪽에 둔다)
        cut_line, cut_color: 자른 높이를 가로선으로 표시할지 여부와 그 색상
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)
    """
    # --- 1) 모델이 합쳐온 과정을 dendrogram()이 읽는 표(linkage 행렬)로 변환 ---
    source = dendrogram_source(estimator)

    # --- 2) 자른 높이 확인 (지정하지 않았다면 모델이 실제로 자른 높이를 되돌린다) ---
    if cut_height is None:
        cut_height = _cut_height(estimator)

    # --- 3) 제목과 x축 이름 결정 ---
    if title is None:       title = '덴드로그램'

    # 가지를 묶어 그리는 경우에만 눈금에 (묶인 개수)가 표시되므로 축 이름을 나눠 쓴다
    if xlabel is None:
        xlabel = '데이터 (괄호 안은 묶인 개수)' if truncate_mode else '데이터'

    # --- 4) 그래프 초기화 (ax를 전달받은 경우에는 그 위에 겹쳐 그린다) ---
    fig = None
    if ax is None:
        fig, ax = my_plot.init(width=width, height=height, title=title,
                               xlabel=xlabel, ylabel=ylabel)

    # --- 5) 덴드로그램 그리기 ---
    # color_threshold: 자른 높이보다 아래쪽 가지를 군집별로 다른 색으로 칠한다
    dendrogram(source, ax=ax, p=p, truncate_mode=truncate_mode,
               leaf_rotation=leaf_rotation, leaf_font_size=leaf_font_size,
               count_sort=count_sort, color_threshold=cut_height)

    # --- 6) 나무를 자른 높이를 가로선으로 표시 ---
    # (이 선과 만나는 가지의 수가 곧 군집 수다)
    if cut_line and cut_height is not None:
        ax.axhline(y=cut_height, color=cut_color, linestyle='--')
        ax.text(ax.get_xlim()[1], cut_height, f' 자른 높이 = {cut_height:.3f}',
                color=cut_color, va='bottom', ha='right')

    # --- 7) 그래프 표시 (ax를 전달받은 경우에는 호출한 쪽에서 표시한다) ---
    if fig is not None:
        my_plot.show(save_path=save_path)


# ===================================================================
# 계층적(병합형) 군집분석 — 가까운 것끼리 차례로 합치고 그 과정을 덴드로그램으로 확인한다
# ===================================================================
def agglomerative(data, k=None, distance_threshold=None, columns=None, scaling='standard',
                  cluster_name='그룹번호', linkage='ward', metric='euclidean',
                  verbose=True, plot=True, title=None,
                  p=30, truncate_mode='lastp', leaf_rotation=0, leaf_font_size=8,
                  count_sort='ascending', cut_line=True, cut_color='#ff0000',
                  width=1280, height=640, save_path=None, ax=None):
    """데이터를 계층적으로 묶어 군집화하고, 합쳐온 과정을 덴드로그램으로 시각화하는 함수

    Args (기본값은 위의 함수 정의 참고):
        data: 군집화할 데이터프레임
        k: 나눌 군집의 개수 (distance_threshold 와 둘 중 하나만 지정한다)
        distance_threshold: 병합을 멈출 거리 기준 (이 값 이상 떨어진 덩어리는 합치지 않는다)
        columns, cluster_name: 사용할 컬럼(None이면 수치형 전체), 군집 번호를 저장할 컬럼명
        scaling: 스케일러 이름('standard'/'minmax'/'robust'/'maxabs', None이면 원본 값)
        linkage: 두 덩어리 사이의 거리를 재는 방법
            ('ward'=합쳤을 때 분산 증가가 최소, 'complete'=가장 먼 쌍,
             'average'=모든 쌍의 평균, 'single'=가장 가까운 쌍)
        metric: 데이터 사이의 거리 계산 방식 (linkage='ward'는 'euclidean'만 지원한다)
        verbose: 스케일링 전후의 값의 범위와 군집별 데이터 개수를 출력할지 여부
        plot, title: 덴드로그램 시각화 여부, 그래프 제목(None이면 자동 생성)
        p, truncate_mode: 덴드로그램에 표시할 가지의 개수와 생략 방식
            ('lastp'=마지막 p개의 가지만, None이면 데이터 전체를 그린다)
        leaf_rotation, leaf_font_size, count_sort: 아래쪽 눈금의 회전 각도, 글자 크기,
            가지의 정렬 기준('ascending'이면 작은 덩어리를 왼쪽에 둔다)
        cut_line, cut_color: 나무를 자른 높이를 가로선으로 표시할지 여부와 그 색상
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (estimator, df) — 학습이 완료된 모델,
            군집 번호 컬럼이 추가된 데이터(스케일링 적용 후)
    """
    # --- 1) 자를 기준이 하나만 지정되었는지 확인 ---
    # 군집 수와 거리 기준은 "나무를 어디서 자를까"에 대한 서로 다른 답이므로 함께 쓸 수 없다
    if (k is None) == (distance_threshold is None):
        raise ValueError("k(군집 수)와 distance_threshold(거리 기준) 중 "
                         "하나만 지정해야 합니다.")

    # --- 2) 군집화에 사용할 컬럼 결정 ---
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # --- 3) 스케일링 적용 ---
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # --- 4) 모델 생성 및 학습 (가까운 덩어리부터 차례로 합치는 과정) ---
    # compute_distances: 병합 거리를 남겨야 덴드로그램의 높이를 그릴 수 있다
    # compute_full_tree: 중간에 멈추면 나무의 윗부분이 잘려 덴드로그램이 불완전해진다
    estimator = AgglomerativeClustering(n_clusters=k, distance_threshold=distance_threshold,
                                        linkage=linkage, metric=metric,
                                        compute_distances=True, compute_full_tree=True)
    estimator.fit(df)

    # --- 5) 각 데이터가 몇 번 그룹인지 컬럼으로 추가 ---
    # 계층적 군집은 새로운 데이터를 예측하는 기능이 없으므로 학습 결과(labels_)를 그대로 쓴다
    df[cluster_name] = estimator.labels_

    # --- 6) 나무를 자른 높이 계산 ---
    cut_height = _cut_height(estimator)

    # --- 7) 군집 결과 요약 출력 ---
    if verbose:
        sizes = df[cluster_name].value_counts().sort_index()

        print(f"[계층적 군집] 군집 수 = {estimator.n_clusters_}, "
              f"연결 방법 = {linkage}, 거리 = {metric}")

        if cut_height is not None:
            print(f"  · 자른 높이(병합 거리) = {cut_height:.4f}")

        print("  · 군집별 데이터 개수 : " +
              ', '.join([f"{c}번 {n}개" for c, n in sizes.items()]))

    # --- 8) 덴드로그램 시각화 ---
    if plot:
        # 제목을 지정하지 않은 경우 자른 기준과 군집 개수를 포함한 제목을 자동으로 생성
        if title is None:
            basis = (f'거리 {distance_threshold}' if distance_threshold is not None else f'군집수 {k}')
            title = f'덴드로그램 ({basis} 기준 → {estimator.n_clusters_}개 군집)'

        # 그리는 일은 덴드로그램 함수에 맡긴다 (자른 높이를 넘겨 가로선과 색을 함께 표시)
        plot_dendrogram(estimator, cut_height=cut_height, title=title,
                             p=p, truncate_mode=truncate_mode,
                             leaf_rotation=leaf_rotation, leaf_font_size=leaf_font_size,
                             count_sort=count_sort, cut_line=cut_line, cut_color=cut_color,
                             width=width, height=height, save_path=save_path, ax=ax)

    # --- 9) 모델과 군집 결과 반환 ---
    return estimator, df


# ===================================================================
# 최적 eps 탐색 — k번째 이웃까지의 거리가 치솟는 지점을 eps 후보로 삼는다
# ===================================================================
def best_eps(data, min_samples=5, columns=None, scaling='standard',
             metric='euclidean', n_jobs=-1, sensitivity=1.0, verbose=True,
             plot=True, title=None, color='#1f77b4', linestyle='-',
             best_color='#ff0000', width=1280, height=640, save_path=None, ax=None):
    """k-distance plot 의 꺾이는 지점을 찾아 DBSCAN 의 최적 eps 를 추정하는 함수

    Args (기본값은 위의 함수 정의 참고):
        data: 군집화할 데이터프레임
        min_samples: 반경 안에 있어야 할 최소 데이터 개수 (이 값이 곧 k가 된다)
        columns, scaling: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값. 이미 스케일링한 데이터라면 None)
        metric, n_jobs: 거리 계산 방식, 사용할 CPU 수(-1이면 전부 사용)
        sensitivity: KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다
        verbose, plot, title: 계산 결과 출력 여부, 시각화 여부,
            그래프 제목(None이면 자동 생성)
        color, linestyle, best_color: 거리 곡선의 색상·선 스타일,
            꺾이는 지점을 표시할 가로·세로선의 색상
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (best_eps, result_df) — eps 후보,
            거리 순위별 k번째 이웃까지의 거리가 담긴 데이터프레임
    """
    # --- 1) 대상 컬럼 결정 ---
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # --- 2) 스케일링 적용 ---
    # 거리를 재는 계산이므로 단위가 큰 변수가 거리를 독점하지 않도록 맞춰준다
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # --- 3) 각 데이터에서 k번째 이웃까지의 거리 구하기 ---
    k = min_samples

    neighbors = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs)
    neighbors.fit(df)

    # distance[i] = i번째 데이터에서 이웃들까지의 거리 (가까운 순)
    # kneighbors() 는 자기 자신을 0번 이웃으로 포함하므로 마지막 열이 곧
    # "자기 포함 min_samples 개를 채우려면 반경을 얼마까지 벌려야 하는가"가 된다
    distance, _ = neighbors.kneighbors(df)

    # 마지막 열만 뽑아서 작은 순으로 정렬한다.
    # 정렬해야 x축이 밀도 순위가 되어 끝에서 치솟는 모양을 눈으로 확인할 수 있다
    target = np.sort(distance[:, k - 1])

    # --- 4) 결과 정리 ---
    result_df = DataFrame({
        '순위': range(1, len(target) + 1),
        f'{k}번째 이웃까지의 거리': np.round(target, 4),
    })

    # --- 5) 꺾이는 지점(엘보우 포인트) 찾기 ---
    # convex(아래로 볼록) + increasing(우상향)은 정렬된 거리 곡선의 모양이다
    kl = KneeLocator(range(len(target)), target, curve='convex',
                     direction='increasing', S=sensitivity)

    point = kl.elbow        # 꺾이는 지점의 '순서'
    eps = kl.elbow_y        # 꺾이는 지점의 '거리' → 이것이 eps 후보

    # 곡선이 거의 직선이면 꺾이는 지점이 없어 None이 나온다
    if eps is None:
        print("곡선이 완만해 꺾이는 지점을 찾지 못했습니다. "
              "sensitivity 를 낮추거나 min_samples 를 조정해 다시 시도해 보세요.")
        return None, result_df

    if verbose:
        print(f"[k-distance] min_samples = {k}, 최적의 eps = {eps:.4f}")
        print(f"  · 꺾이는 위치 = {point}번째 데이터 (전체 {len(target)}개 중)")
        print(f"  · 노이즈 후보 = {int((target > eps).sum())}개 (거리가 eps 보다 먼 데이터)")

    # --- 6) 시각화 ---
    if plot:
        if title is None:   title = f'{k}번째 이웃까지의 거리 (k-distance plot)'

        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                   xlabel='거리 순으로 정렬한 데이터', ylabel='거리')

        my_plot.lineplot(x=list(range(len(target))), y=target,
                         color=color, linestyle=linestyle, ax=ax)

        # 꺾이는 지점을 가로선(거리)과 세로선(순서)으로 표시
        ax.axhline(y=eps, color=best_color, linestyle='--', linewidth=1)
        ax.axvline(x=point, color=best_color, linestyle='--', linewidth=1)
        ax.text(0, eps, f'eps = {eps:.4f}', color=best_color, va='bottom')

        if fig is not None:
            my_plot.show(save_path=save_path)

    # --- 7) eps 후보와 거리 표 반환 ---
    return eps, result_df


# ===================================================================
# DBSCAN 군집분석 — 빽빽하게 모인 곳을 군집으로 묶고, 남는 데이터는 노이즈로 분리한다
# ===================================================================
def dbscan(data, eps=0.5, min_samples=5, columns=None, scaling='standard',
           cluster_name='그룹번호', vector_name='벡터유형', metric='euclidean', n_jobs=-1,
           verbose=True, plot=True, x=None, y=None, title=None, outline=True,
           palette='tab10', size=100, edgecolor='#ffffff', linewidth=1.5, alpha=1,
           core_marker='o', border_marker='^', border_size=120, border_alpha=0.5,
           noise_marker='X', noise_size=150, noise_color='#ff0000',
           noise_edgecolor='#000000', noise_linewidth=1.5,
           width=1280, height=640, save_path=None, ax=None):
    """반경 안의 데이터 개수(밀도)를 기준으로 군집화하고, 그 결과를 시각화하는 함수

    Args (기본값은 위의 함수 정의 참고):
        data: 군집화할 데이터프레임
        eps: 이웃으로 인정할 반경 (가장 중요한 값. 표준화 기준 0.3~1.0 에서 탐색한다)
        min_samples: 반경 안에 있어야 할 최소 데이터 개수 (변수가 2~3개면 3~6)
        columns, cluster_name, vector_name: 사용할 컬럼(None이면 수치형 전체),
            군집 번호·벡터 유형을 저장할 컬럼명
        scaling: 스케일러 이름('standard'/'minmax'/'robust'/'maxabs', None이면 원본 값)
        metric, n_jobs: 거리 계산 방식, 사용할 CPU 수(-1이면 전부 사용)
        verbose, plot: 스케일링·군집 요약의 출력 여부, 시각화 여부
        x, y, title: 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개), 그래프 제목
        outline: 군집의 외곽선(ConvexHull)을 표시할지 여부
        palette, size, edgecolor, linewidth, alpha: 군집별 색상 팔레트(외곽 벡터·외곽선에도
            같이 적용), 핵심 벡터의 마커 크기, 테두리 색상, 테두리 두께, 투명도
        core_marker, border_marker, border_size, border_alpha: 핵심·외곽 벡터의 마커 모양,
            외곽 벡터의 마커 크기와 투명도(색은 그대로 두고 농도만 낮춰 구분한다)
        noise_marker, noise_size, noise_color, noise_edgecolor, noise_linewidth:
            노이즈 마커의 모양·크기·색상·테두리 색상·테두리 두께
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (estimator, df, summary_df) — 학습이 완료된 모델,
            군집 번호·벡터 유형 컬럼이 추가된 데이터(스케일링 적용 후),
            군집별 데이터 개수·비율·벡터 유형 개수를 정리한 표(노이즈는 -1 행)
    """
    # --- 1) 군집화에 사용할 컬럼 결정 ---
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # --- 2) 스케일링 적용 ---
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # --- 3) 모델 생성 및 학습 (밀도가 높은 곳을 찾아 번호를 붙이는 과정) ---
    estimator = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=n_jobs)
    estimator.fit(df)

    # DBSCAN 에는 predict() 가 없다. 학습 결과는 labels_ 에 들어 있다
    labels = estimator.labels_

    # --- 4) 각 데이터의 군집 번호와 벡터 유형을 컬럼으로 추가 ---
    # 핵심(core) : 반경 안에 min_samples 개 이상을 거느린 데이터 (군집의 몸통)
    # 외곽(border): 스스로는 기준에 못 미치지만 핵심의 반경 안에 있는 데이터 (군집의 가장자리)
    # 노이즈(noise): 어느 쪽도 아닌 데이터 (군집 번호가 -1)

    # 'border'라는 값으로 채운, 데이터 길이와 동일한 배열 생성
    vectors = np.full(len(df), 'border', dtype=object)

    # core_sample_indices_ 는 "몇 번째 행"인지를 담은 위치 번호이므로 위치로 사용한다.
    # 이 위치에 해당하는 벡터 유형을 'core'로 바꾼다.
    vectors[estimator.core_sample_indices_] = 'core'

    # 노이즈는 labels_가 -1이므로 따로 처리한다
    vectors[labels == -1] = 'noise'

    # 원본 데이터에 군집 번호와 벡터 유형 컬럼을 추가한다
    df[cluster_name] = labels
    df[vector_name] = vectors

    # --- 5) 군집별 요약 정리 ---
    # 노이즈(-1)는 군집이 아니므로 군집 개수에서 제외한다
    cluster_ids = sorted([c for c in set(labels) if c != -1])
    n_clusters = len(cluster_ids)
    n_noise = int((labels == -1).sum())

    items = []

    for c in sorted(set(labels)):
        mask = labels == c

        items.append({
            cluster_name: c,
            '데이터수': int(mask.sum()),
            '비율(%)': round(mask.sum() / len(labels) * 100, 1),
            '핵심벡터': int((vectors[mask] == 'core').sum()),
            '외곽벡터': int((vectors[mask] == 'border').sum()),
        })

    summary_df = DataFrame(items)

    # --- 6) 군집 결과 요약 출력 ---
    if verbose:
        print(f"[DBSCAN] eps = {eps}, min_samples = {min_samples}, 거리 = {metric}")
        print(f"  · 군집 수 = {n_clusters}개 (노이즈 제외)")
        print(f"  · 노이즈 = {n_noise}개 (전체의 {n_noise / len(labels):.1%})")

        # 군집이 하나도 만들어지지 않았다면 두 값이 데이터의 밀도와 맞지 않는다는 뜻이다
        if n_clusters == 0:
            print("  · 군집이 만들어지지 않았습니다. "
                  "eps 를 키우거나 min_samples 를 줄여 다시 시도해 보세요.")

        display(summary_df)

    # --- 7) 군집 결과 시각화 ---
    if plot:
        # 7-0) 컬럼, 제목 설정
        # 축으로 사용할 컬럼 결정 (지정이 없으면 대상 컬럼의 앞에서 두 개)
        if x is None:       x = columns[0]
        if y is None:       y = columns[1]

        # 제목을 지정하지 않은 경우 두 하이퍼파라미터를 포함한 제목을 자동으로 생성
        if title is None:
            title = f'DBSCAN 군집 결과 (eps={eps:.3g}, min_samples={min_samples})'

        # 7-1) 그래프 초기화 (ax를 전달받은 경우에는 그 위에 겹쳐 그린다)
        fig = None

        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                   xlabel=x, ylabel=y)

        # 7-2) 벡터 유형에 따라 데이터를 세 덩어리로 나눈다
        # (한 번에 그리지 않고 나눠 그려야 유형마다 마커 모양과 농도를 달리할 수 있다)
        core = df[df[vector_name] == 'core']
        border = df[df[vector_name] == 'border']
        noise = df[df[vector_name] == 'noise']

        # 7-3) 핵심 벡터 --> 군집별 색상, 진한 마커
        if not core.empty:
            my_plot.scatterplot(data=core, x=x, y=y, hue=cluster_name,
                                palette=palette, marker=core_marker, size=size,
                                edgecolor=edgecolor, linewidth=linewidth,
                                alpha=alpha, outline=False, ax=ax)

        # 7-4) 외곽 벡터 --> 군집별 색상(동일), 연한 마커
        # (범례에 같은 군집이 두 번 나오므로 범례는 끈다)
        if not border.empty:
            my_plot.scatterplot(data=border, x=x, y=y, hue=cluster_name,
                                palette=palette, marker=border_marker, size=border_size,
                                edgecolor=edgecolor, linewidth=linewidth,
                                alpha=border_alpha, outline=False, ax=ax, legend=False)

        # 7-5) 외곽선은 군집 단위로 그린다
        # (핵심/외곽으로 나눠 그리면 하나의 군집이 두 개로 쪼개져 보인다)
        if outline and cluster_ids:
            my_plot.plot_hull(data=df[df[cluster_name] != -1], x=x, y=y,
                              hue=cluster_name, palette=palette, ax=ax)

        # 7-6) 노이즈 --> 군집이 아니므로 팔레트 없이 눈에 띄는 마커로 덧그린다
        # (이상치 후보를 바로 찾기 위한 표시)
        if not noise.empty:
            my_plot.scatterplot(data=noise, x=x, y=y,
                                marker=noise_marker, size=noise_size,
                                color=noise_color, edgecolor=noise_edgecolor,
                                linewidth=noise_linewidth, outline=False, ax=ax,
                                label='noise')

        # 7-7) 그래프 표시 (ax를 전달받은 경우에는 호출한 쪽에서 표시한다)
        if fig is not None:
            my_plot.show(save_path=save_path)

    # --- 8) 모델, 군집 결과, 요약 표 반환 ---
    return estimator, df, summary_df


# ===================================================================
# 페르소나 도출 — 군집 번호를 "어떤 고객인지"로 번역한다
# ===================================================================
def persona(data, labels=None, columns=None, cluster_name='ClusterID',
            num_columns=None, cat_columns=None, exclude=None, alpha=0.05,
            show_stat=True, verbose=True, plot=True, crosstab_plot=False,
            palette='tab10', cols=2, width=800, height=520, save_path=None):
    """군집별 대표값·구성비를 집계해 페르소나 표를 만들고, 군집별 분포를 시각화하는 함수

    학습은 스케일링된 값으로 하지만 해석은 사람이 읽을 수 있는 원본 값으로 해야 하므로,
    이 함수에는 스케일링 전의 원본 데이터를 넘긴다. 군집화에 쓰지 않은 변수까지 함께
    집계하는 이유는, 거기서 나온 차이가 "나눠 놓고 보니 달랐다"는 더 강한 근거이기 때문이다.

    Args (기본값은 위의 함수 정의 참고):
        data: 해석에 사용할 원본 데이터프레임 (스케일링 전의 값)
        labels: 군집 번호. 학습이 끝난 모델·배열·Series 모두 가능하며,
            None이면 data 안의 cluster_name 컬럼을 사용한다
        columns: 군집화에 실제로 사용한 컬럼 목록 (그래프 제목에 사용 여부를 표시한다)
        cluster_name: 군집 번호 컬럼명
        num_columns, cat_columns: 집계할 연속형·범주형 컬럼(None이면 자동 선택,
            값이 모두 다른 식별자 컬럼은 자동으로 제외한다)
        exclude: 자동 선택에서 빼고 싶은 컬럼 목록
        alpha: 정규성 검정의 유의수준 (p > alpha 이면 평균, 아니면 중앙값을 대표값으로 쓴다)
        show_stat: 평균과 중앙값을 모두 표에 넣고 채택한 쪽에 별표(*)를 붙일지 여부
            (붙이면 해당 컬럼이 문자열이 되므로, 값을 계산에 쓸 때는 False로 지정한다.
             False면 채택한 대표값 하나만 숫자로 담는다)
        verbose: 대표값 선택 근거와 범주형 교차표를 출력할지 여부
        plot, crosstab_plot: 군집별 상자그림을 그릴지 여부,
            범주형 구성비를 히트맵으로도 그릴지 여부
        palette, cols: 색상 팔레트, 상자그림을 배치할 열의 개수
        width, height, save_path: 그래프 한 칸의 가로·세로 픽셀, 상자그림 저장 경로

    Returns:
        tuple: (persona_df, ratio_dict) — 군집별 대표값 표(군집 번호 순),
            범주형 컬럼별 구성비(%) 표의 딕셔너리(마지막 '전체' 행은 데이터 전체의 비율)
    """
    # --- 1) 원본 데이터에 군집 번호 붙이기 ---
    df = data.copy()

    if labels is not None:
        # 모델을 그대로 넘긴 경우에는 학습 결과(labels_)를 꺼내 쓴다
        df[cluster_name] = getattr(labels, 'labels_', labels)
    elif cluster_name not in df.columns:
        raise ValueError(f"군집 번호를 찾을 수 없습니다. labels 를 넘기거나 "
                         f"'{cluster_name}' 컬럼이 있는 데이터를 사용하세요.")

    # --- 2) 집계에서 뺄 컬럼 추리기 ---
    # 고객ID처럼 값이 모두 다른 컬럼은 대표값을 구해도 의미가 없으므로 자동으로 걸러낸다
    exclude = list(exclude) if exclude else []

    # 값이 행의 개수만큼 전부 다른 컬럼은 식별자로 보고 제외한다 (고객ID, 주문번호 등)
    id_columns = [c for c in df.columns
                  if c != cluster_name and df[c].nunique() == len(df)]

    if id_columns and verbose:
        print(f"[페르소나] 식별자로 판단해 제외한 컬럼 : {', '.join(id_columns)}")

    drop = set(exclude + id_columns + [cluster_name])

    # --- 3) 집계할 연속형 컬럼 결정 (수치형 전체에서 군집 번호·식별자를 제외) ---
    if num_columns is None:
        num_columns = [c for c in my_qtcheck.get_number_column_names(df) if c not in drop]

    # --- 4) 집계할 범주형 컬럼 결정 ---
    # category 타입 전체 + 문자열(object) 컬럼
    # (set_type() 으로 타입을 지정하지 않은 데이터도 그대로 쓸 수 있게 한다)
    if cat_columns is None:
        # 4-1) category 로 지정한 컬럼과 문자열 컬럼을 차례로 모아 후보를 만든다
        candidates = my_qtcheck.get_categorical_column_names(df)
        candidates += df.select_dtypes(include='object').columns.to_list()

        cat_columns = []

        # 4-2) 두 목록에 겹쳐 들어온 컬럼과 군집 번호·식별자는 담지 않는다
        for column in candidates:
            if column not in drop and column not in cat_columns:
                cat_columns.append(column)

    # --- 5) 컬럼 목록 정리 ---
    # 넘겨받은 목록을 그대로 쓰면 아래에서 순서를 바꿀 때 호출부의 리스트까지 바뀐다
    num_columns, cat_columns = list(num_columns), list(cat_columns)

    # 군집화에 사용한 컬럼을 앞쪽에 두어 표에서 먼저 읽히게 한다
    if columns:
        used = [c for c in columns if c in num_columns]
        num_columns = used + [c for c in num_columns if c not in used]

    # --- 6) 군집 번호 목록 확인 (노이즈(-1)는 군집이 아니므로 제외한다) ---
    cluster_ids = sorted([c for c in df[cluster_name].unique() if c != -1])
    total = len(df)

    if verbose:
        print(f"[페르소나] 군집 수 = {len(cluster_ids)}, "
              f"연속형 = {num_columns}, 범주형 = {cat_columns}")
        print('-' * 60)

    # --- 7) 군집별 대표값 집계 ---
    persona_list = []

    for c in cluster_ids:
        # 7-1) 현재 군집에 속한 데이터만 추출
        cluster_data = df[df[cluster_name] == c]

        # 7-2) 군집의 크기와 전체에서 차지하는 비중
        persona_item = {
            cluster_name: c,
            '데이터수': len(cluster_data),
            '비율(%)': round(len(cluster_data) / total * 100, 1),
        }

        # 7-3) 연속형 변수: 정규분포면 평균, 아니면 중앙값
        for column in num_columns:
            # 정규성 검정 — normaltest 는 왜도·첨도를 함께 보므로 표본이 8개는 되어야 하고,
            # 값이 모두 같으면 계산 자체가 되지 않는다 → 이런 경우는 검정을 건너뛴다
            values = cluster_data[column].dropna()
            p = normaltest(values)[1] if len(values) >= 8 and values.nunique() >= 2 else None

            # 대표값 선택 — 평균은 한쪽으로 치우친 분포에서 꼬리에 끌려가므로 정규분포일 때만
            # 쓰고, 검정을 못 한 경우에도 안전한 쪽인 중앙값을 쓴다
            method = '평균' if p is not None and p > alpha else '중앙값'

            # 두 값을 나란히 두면 "평균과 중앙값이 얼마나 벌어져 있는가"까지 함께 읽힌다
            stats = {
                '평균': round(values.mean(), 3),
                '중앙값': round(values.median(), 3),
            }

            # 표에 담기 — 두 값을 모두 넣을지, 채택한 값 하나만 넣을지
            if show_stat:
                # 같은 컬럼이라도 군집마다 채택되는 쪽이 달라지므로 별표로 표시한다
                for name, v in stats.items():
                    persona_item[f'{column}({name})'] = f'{v}*' if name == method else f'{v}'
            else:
                persona_item[column] = stats[method]

            # 선택 근거 출력
            if verbose:
                reason = f"p={p:.3f}" if p is not None else "검정 불가"
                print(f"  군집 {c} - {column} : {method} 사용 ({reason})")

        # 7-4) 범주형 변수: 최빈값
        for column in cat_columns:
            persona_item[column] = cluster_data[column].value_counts().idxmax()

            if verbose:
                print(f"  군집 {c} - {column} : 최빈값 = {persona_item[column]}")

        # 7-5) 군집별 대표값을 리스트에 추가
        persona_list.append(persona_item)

    # --- 8) 페르소나 표 완성 (군집 번호 순으로 정렬) ---
    persona_df = DataFrame(persona_list).sort_values(by=cluster_name).reset_index(drop=True)

    # --- 9) 범주형 구성비 계산 ---
    # 최빈값만 보면 모든 군집이 같은 값으로 나오는 경우가 많으므로 구성비까지 함께 본다
    ratio_dict = {}

    for column in cat_columns:
        # 9-1) 군집 × 범주 교차표 (건수)
        ct = crosstab(df[cluster_name], df[column])

        # 9-2) 비교 기준이 되는 전체 비율은 노이즈를 빼기 전에 구한다 (실제 데이터 전체의 구성비)
        overall = (ct.sum(axis=0) / ct.to_numpy().sum() * 100).round(1)

        # 9-3) 노이즈를 뺀 뒤, 군집마다 행 합이 100%가 되도록 환산
        ct = ct.loc[[c for c in ct.index if c in cluster_ids]]
        ratio = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)

        # 9-4) 데이터 전체의 비율을 마지막 행에 붙여 "이 군집만 다른가"를 바로 비교하게 한다
        ratio.loc['전체'] = overall
        ratio_dict[column] = ratio

        # 9-5) 교차표와 구성비 출력
        if verbose:
            print('-' * 60)
            print(f"[구성비] {cluster_name} × {column}")
            display(ct)
            display(ratio)

        # 9-6) 군집 수와 범주 수가 많을 때는 표보다 히트맵이 빠르게 읽힌다
        if crosstab_plot:
            my_plot.heatmap(data=ratio, fmt='0.1f', palette=palette,
                            title=f'{column} 구성비 (%)',
                            xlabel=column, ylabel=cluster_name,
                            width=width * cols, height=height)

    # --- 10) 군집별 분포 시각화 ---
    if plot and num_columns:
        # 10-1) 컬럼 수에 맞춰 그래프 칸을 몇 행 몇 열로 놓을지 계산
        n = len(num_columns)
        ncols = min(cols, n)
        nrows = int(np.ceil(n / ncols))

        # 10-2) 그래프 초기화
        fig, ax = my_plot.init(width=width, height=height, rows=nrows, cols=ncols,
                               title='군집별 분포')

        # 한 칸짜리 그래프는 배열이 아닌 Axes 하나가 오므로 배열로 맞춰준다
        axes = np.atleast_1d(ax)

        # 10-3) 노이즈(-1)는 군집이 아니므로 상자그림에서도 뺀다
        vdf = df[df[cluster_name].isin(cluster_ids)]

        # 10-4) 연속형 컬럼마다 군집별 상자그림
        for i, column in enumerate(num_columns):
            my_plot.boxplot(data=vdf, x=cluster_name, y=column, hue=cluster_name,
                            palette=palette, ax=axes[i])

            # 군집화에 쓴 변수인지 표시한다 (쓰지 않은 변수에서 나온 차이가 더 강한 근거다)
            mark = '군집화 변수' if columns and column in columns else '군집화에 쓰지 않은 변수'
            axes[i].set_title(f'{column} ({mark})', fontsize=16, pad=10)
            axes[i].set_xlabel('군집 번호')
            axes[i].set_ylabel(column)

        # 10-5) 컬럼 수가 칸 수보다 적으면 남는 칸은 숨긴다
        for a in axes[n:]:
            a.set_visible(False)

        # 10-6) 전체 제목이 들어갈 위쪽 공간(7%)을 미리 비워 둔 뒤 표시한다
        #       (칸이 하나면 전체 제목이 붙지 않으므로 공간을 비우지 않는다)
        if len(axes) > 1:
            fig.tight_layout(rect=[0, 0, 1, 0.93])
        my_plot.show(save_path=save_path)

    # --- 11) 페르소나 표와 범주형 구성비 반환 ---
    return persona_df, ratio_dict
