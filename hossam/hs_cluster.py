# -*- coding: utf-8 -*-
# ===================================================================
# 파이썬 기본 패키지 참조
# ===================================================================
import numpy as np
import concurrent.futures as futures
from tqdm.auto import tqdm
from itertools import combinations
from typing import Literal, Callable

# ===================================================================
# 데이터 분석 패키지 참조
# ===================================================================
from kneed import KneeLocator
from pandas import Series, DataFrame, MultiIndex, concat
from matplotlib.pyplot import Axes  # type: ignore
from scipy.stats import normaltest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ===================================================================
# hossam 패키지 참조
# ===================================================================
from . import hs_plot
from .hs_util import is_2d

RANDOM_STATE = 52


# ===================================================================
# K-평균 군집화 모델을 적합하는 함수.
# ===================================================================
def kmeans_fit(
    data: DataFrame,
    n_clusters: int | None = None,
    k_range: list | tuple = [2, 11],
    random_state: int = RANDOM_STATE,
    plot: bool = False,
    fields: list[str] | tuple[str] | tuple[tuple[str]] | list[list[str]] | None = None,
    **params,
) -> tuple[KMeans, DataFrame, float]:
    """
    K-평균 군집화 모델을 적합하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        n_clusters (int | None): 군집 개수.
        random_state (int, optional): 랜덤 시드. 기본값은 RANDOM_STATE.
        plot (bool, optional): True면 결과를 시각화함. 기본값 False.
        fields (list[list[str]] | None, optional): 시각화할 필드 쌍 리스트. 기본값 None이면 수치형 컬럼의 모든 조합 사용.
        **params: KMeans에 전달할 추가 파라미터.

    Returns:
        KMeans: 적합된 KMeans 모델.
        DataFrame: 클러스터 결과가 포함된 데이터 프레임
        float: 실루엣 점수
    """
    df = data.copy()

    if n_clusters is None:
        n_clusters = kmeans_best_k(data=df, k_range=k_range, random_state=random_state, plot=False)
        print(f"Best k found: {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **params)
    kmeans.fit(data)
    df["cluster"] = kmeans.predict(df)
    score = float(silhouette_score(X=data, labels=df["cluster"]))

    if plot:

        if not is_2d(fields):
            fields = [fields]   # type: ignore

        # cluster_plot(
        #     estimator=kmeans,
        #     data=data,
        #     fields=fields,
        #     title=f"K-Means Clustering (k={n_clusters})",
        # )
        for f in fields:  # type: ignore
            hs_plot.visualize_silhouette(
                estimator=kmeans,
                data=data,
                xname=f[0],     # type: ignore
                yname=f[1],     # type: ignore
                title=f"K-Means Clustering (k={n_clusters})",
                outline=True,
            )

    return kmeans, df, score


# ===================================================================
# K-평균 군집화에서 엘보우(Elbow) 기법을 활용해 최적의 K값을 탐지하는 함수.
# ===================================================================
def kmeans_elbow(
    data: DataFrame,
    k_range: list | tuple = [2, 11],
    S: float = 0.1,
    random_state: int = RANDOM_STATE,
    plot: bool = True,
    title: str = None,
    marker: str = None,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: int = hs_plot.config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
    callback: Callable | None = None,
    **params,
) -> tuple:
    """
    K-평균 군집화에서 엘보우(Elbow) 기법을 활용해 최적의 K값을 탐지하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        k_range (list | tuple, optional): K값의 범위 지정. 기본값은 [2, 11].
        S (float, optional): KneeLocator의 민감도 파라미터. 기본값 0.1.
        random_state (int, optional): 랜덤 시드. 기본값은 RANDOM_STATE.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.
        title (str, optional): 플롯 제목.
        marker (str, optional): 마커 스타일.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (int, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.
        callback (Callable | None, optional): 플롯 후 호출할 콜백 함수.
        **params: lineplot에 전달할 추가 파라미터.

    Returns:
        tuple: (best_k, inertia_list)
            - best_k: 최적의 K값
            - inertia_list: 각 K값에 대한 inertia 리스트

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        best_k, inertia_list = hs_cluster.kmeans_elbow(data.iloc[:, :-1])
        ```
    """

    inertia_list = []

    r = range(k_range[0], k_range[1])

    for k in r:
        kmeans, _, score = kmeans_fit(
            data=data, n_clusters=k, random_state=random_state
        )
        inertia_list.append(kmeans.inertia_)

    best_k, _ = elbow_point(
        x=list(r),
        y=inertia_list,
        dir="left,down",
        S=S,
        plot=plot,
        marker=marker,
        width=width,
        height=height,
        linewidth=linewidth,
        save_path=save_path,
        title=(
            f"K-Means Elbow Method (k={k_range[0]}-{k_range[1]-1}, silhouette={score:.3f})"
            if title is None
            else title
        ),
        ax=ax,
        callback=callback,
        **params,
    )

    return best_k, inertia_list


# ===================================================================
# K-평균 군집화에서 실루엣 점수를 계산하는 함수.
# ===================================================================
def kmeans_silhouette(
    data: DataFrame,
    k_range: list | tuple = [2, 11],
    random_state: int = RANDOM_STATE,
    plot: Literal[False, "silhouette", "cluster", "both"] = "both",
    title: str = None,
    xname: str = None,
    yname: str = None,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: float = hs_plot.config.line_width,
    save_path: str | None = None,
    **params,
) -> DataFrame:
    """
    K-평균 군집화에서 실루엣 점수를 계산하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        k_range (list | tuple, optional): K값의 범위 지정. 기본값은 [2, 11].
        random_state (int, optional): 랜덤 시드. 기본값은 RANDOM_STATE.
        plot (Literal[False, "silhouette", "cluster", "both"], optional):
            플롯 옵션 지정. 기본값 "both".
        title (str, optional): 플롯 제목.
        xname (str, optional): 군집 산점도의 x축 컬럼명.
        yname (str, optional): 군집 산점도의 y축 컬럼명.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (float, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        **params: silhouette_plot에 전달할 추가 파라미터.

    Returns:
        DataFrame: 각 K값에 대한 실루엣 점수 데이터프레임.

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        silhouette_scores = hs_cluster.kmeans_silhouette(data.iloc[:, :-1], k=3)
        ```
    """

    klist = list(range(k_range[0], k_range[1]))
    total = len(klist)

    if plot is not False:
        total *= 2

    with tqdm(total=total) as pbar:
        silhouettes = []
        estimators = []

        def __process_k(k):
            estimator, cdf, score = kmeans_fit(
                data=data, n_clusters=k, random_state=random_state
            )
            return score, estimator

        with futures.ThreadPoolExecutor() as executor:
            executed = []
            for k in klist:
                pbar.set_description(f"K-Means Silhouette: k={k}")
                executed.append(executor.submit(__process_k, k))

            for e in executed:
                s_score, estimator = e.result()
                silhouettes.append(s_score)
                estimators.append(estimator)
                pbar.update(1)

        if plot is not False:
            for estimator in estimators:
                pbar.set_description(f"K-Means Plotting: k={estimator.n_clusters}")

                if plot == "silhouette":
                    hs_plot.silhouette_plot(
                        estimator=estimator,
                        data=data,
                        title=title,
                        width=width,
                        height=height,
                        linewidth=linewidth,
                        save_path=save_path,
                        **params,
                    )
                elif plot == "cluster":
                    hs_plot.cluster_plot(
                        estimator=estimator,
                        data=data,
                        xname=xname,
                        yname=yname,
                        outline=True,
                        palette=None,
                        width=width,
                        height=height,
                        title=title,
                        save_path=save_path,
                    )
                elif plot == "both":
                    hs_plot.visualize_silhouette(
                        estimator=estimator,
                        data=data,
                        xname=xname,
                        yname=yname,
                        outline=True,
                        palette=None,
                        width=width,
                        height=height,
                        title=title,
                        linewidth=linewidth,
                        save_path=save_path,
                    )

                pbar.update(1)

    silhouette_df = DataFrame({"k": klist, "silhouette_score": silhouettes})
    silhouette_df.sort_values(by="silhouette_score", ascending=False, inplace=True)
    return silhouette_df


# ===================================================================
# 엘보우(Elbow) 포인트를 자동으로 탐지하는 함수.
# ===================================================================
def elbow_point(
    x: Series | np.ndarray | list,
    y: Series | np.ndarray | list,
    dir: Literal["left,down", "left,up", "right,down", "right,up"] = "left,down",
    S: float = 0.1,
    plot: bool = True,
    title: str = None,
    marker: str = None,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: int = hs_plot.config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
    callback: Callable | None = None,
    **params,
) -> tuple:
    """
    엘보우(Elbow) 포인트를 자동으로 탐지하는 함수.

    주어진 x, y 값의 곡선에서 KneeLocator를 활용해 엘보우(혹은 니) 포인트를 탐지하고, 필요시 시각화까지 지원함.

    Args:
        x (Series | np.ndarray | list): x축 값(일반적으로 K값 등).
        y (Series | np.ndarray | list): y축 값(일반적으로 inertia, SSE 등).
        dir (Literal["left,down", "left,up", "right,down", "right,up"], optional):
            곡선의 방향 및 형태 지정. 기본값은 "left,down".
            - "left,down": 왼쪽에서 오른쪽으로 감소(볼록)
            - "left,up": 왼쪽에서 오른쪽으로 증가(오목)
            - "right,down": 오른쪽에서 왼쪽으로 감소(볼록)
            - "right,up": 오른쪽에서 왼쪽으로 증가(오목)
        S (float, optional): KneeLocator의 민감도 파라미터. 기본값 0.1.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.
        title (str, optional): 플롯 제목.
        marker (str, optional): 마커 스타일.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (int, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.
        callback (Callable | None, optional): 플롯 후 호출할 콜백 함수.
        **params: lineplot에 전달할 추가 파라미터.

    Returns:
        tuple: (best_x, best_y)
            - best_x: 엘보우 포인트의 x값(예: 최적 K)
            - best_y: 엘보우 포인트의 y값

    Examples:
        ```python
        x = [1, 2, 3, 4, 5, 6]
        y = [100, 80, 60, 45, 44, 43]
        elbow_point(x, y)
        ```

    Note:
        - KneeLocator는 kneed 패키지의 클래스로, 곡선의 형태(curve)와 방향(direction)에 따라 엘보우 포인트를 탐지함.
        - dir 파라미터에 따라 curve/direction이 자동 지정됨.
        - plot=True일 때, 엘보우 포인트에 수직/수평선과 텍스트가 표시됨.
    """

    if dir == "left,down":
        curve = "convex"
        direction = "decreasing"
    elif dir == "left,up":
        curve = "concave"
        direction = "increasing"
    elif dir == "right,down":
        curve = "convex"
        direction = "increasing"
    else:
        curve = "concave"
        direction = "decreasing"

    kn = KneeLocator(x=x, y=y, curve=curve, direction=direction, S=S)

    best_x = kn.elbow
    best_y = kn.elbow_y

    if plot:

        def hvline(ax):
            ax.axvline(best_x, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(best_y, color="red", linestyle="--", linewidth=0.7)
            ax.text(
                best_x,
                best_y + (best_y * 0.01),
                "x=%.2f, y=%.2f" % (best_x, best_y),
                fontsize=6,
                ha="center",
                va="bottom",
                color="black",
                fontweight="bold",
            )

            if callback is not None:
                callback(ax)

        hs_plot.lineplot(
            df=None,
            xname=x,
            yname=y,
            title=title,
            marker=marker,
            width=width,
            height=height,
            linewidth=linewidth,
            save_path=save_path,
            callback=hvline,
            ax=ax,
            **params,
        )

    return best_x, best_y


# ===================================================================
# 데이터프레임의 여러 필드 쌍에 대해 군집 산점도를 그리는 함수.
# ===================================================================
def cluster_plot(
    estimator: KMeans | DBSCAN | AgglomerativeClustering,
    data: DataFrame,
    hue: str | None = None,
    vector: str | None = None,
    fields: list[list] = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = True,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: float = hs_plot.config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
):
    """
    데이터프레임의 여러 필드 쌍에 대해 군집 산점도를 그리는 함수.

    Args:
        estimator (KMeans): KMeans 군집화 모델.
        data (DataFrame): 군집화할 데이터프레임.
        hue (str | None, optional): 군집 레이블 컬럼명. 지정되지 않으면 estimator의 레이블 사용.
        vector (str | None, optional): 벡터 종류를 의미하는 컬럼명(for DBSCAN)
        fields (list[list], optional): 시각화할 필드 쌍 리스트. 기본값 None이면 수치형 컬럼의 모든 조합 사용.
        title (str | None, optional): 플롯 제목.
        palette (str | None, optional): 색상 팔레트 이름.
        outline (bool, optional): True면 데이터 포인트 외곽선 표시. 기본값 False.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (float, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        estimator, cdf, score = hs_cluster.kmeans_fit(data.iloc[:, :-1], n_clusters=3)
        hs_cluster.cluster_plot(cdf, hue='cluster')
        ```
    """

    if fields is None:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("데이터프레임에 수치형 컬럼이 2개 이상 필요합니다.")

        # fields의 모든 조합 생성
        fields = [list(pair) for pair in combinations(numeric_cols, 2)]

    for field_pair in fields:
        xname, yname = field_pair

        hs_plot.cluster_plot(
            estimator=estimator,    # type: ignore
            data=data,
            xname=xname,
            yname=yname,
            hue=hue,
            title=title,
            vector=vector,
            palette=palette,
            outline=outline,
            width=width,
            height=height,
            linewidth=linewidth,
            save_path=save_path,
            ax=ax,
        )


# ===================================================================
# 군집화된 데이터프레임에서 각 군집의 페르소나(특성 요약)를 생성하는 함수.
# ===================================================================
def persona(
    data: DataFrame,
    cluster: str | Series | np.ndarray | list | dict,
    fields: list[str] | None = None,
    full: bool = False,
) -> DataFrame:
    """
    군집화된 데이터프레임에서 각 군집의 페르소나(특성 요약)를 생성하는 함수.

    Args:
        data (DataFrame): 군집화된 데이터프레임.
        cluster (str | Series | ndarray | list | dict): 군집 레이블 컬럼명 또는 배열.
        fields (list[str] | None, optional): 페르소나 생성에 사용할 필드 리스트. 기본값 None이면 수치형 컬럼 전체 사용.
        full (bool, optional): True면 모든 통계량을 포함. 기본값 False.
    Returns:
        DataFrame: 각 군집의 페르소나 요약 데이터프레임.

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        estimator, df, score = hs_cluster.kmeans_fit(data.iloc[:, :-1], n_clusters=3)
        persona_df = hs_cluster.persona(df, hue='cluster')
        print(persona_df)
        ```
    """
    df = data.copy()

    if fields is None:
        fields = df.select_dtypes(include=["number"]).columns.tolist()

    if isinstance(cluster, str):
        if cluster not in df.columns:
            raise ValueError(
                f"cluster로 지정된 컬럼 '{cluster}'이(가) 데이터프레임에 존재하지 않습니다."
            )
    else:
        df["cluster"] = cluster
        cluster = "cluster"
        fields.remove(cluster) if cluster in fields else None

    persona_list = []

    grouped = df.groupby(cluster)
    for cluster_label, group in grouped:
        persona_dict = {}
        # 군집 레이블 및 카운트는 단일 인덱스 유지
        persona_dict[(cluster, "")] = cluster_label
        persona_dict[("", f"count")] = len(group)

        for field in fields:
            if field == cluster:
                continue
            
            # 명목형일 경우 최빈값 사용
            if df[field].dtype == "object" or df[field].dtype.name == "category":
                persona_dict[(field, "mode")] = group[field].mode()[0]
            else:
                if full:
                    persona_dict[(field, "mean")] = group[field].mean()
                    persona_dict[(field, "median")] = group[field].median()
                    persona_dict[(field, "std")] = group[field].std()
                    persona_dict[(field, "min")] = group[field].min()
                    persona_dict[(field, "max")] = group[field].max()
                    persona_dict[(field, "25%")] = group[field].quantile(0.25)
                    persona_dict[(field, "50%")] = group[field].quantile(0.50)
                    persona_dict[(field, "75%")] = group[field].quantile(0.75)
                else:
                    # normaltest를 사용해서 정규분포일 경우 평균/표준편차, 비정규분포일 경우 중앙값/IQR 사용
                    stat, p = normaltest(df[field])
                    alpha = 0.05

                    if p > alpha:
                        # 정규분포
                        persona_dict[(field, "mean")] = group[field].mean()
                        persona_dict[(field, "std")] = group[field].std()
                    else:
                        # 비정규분포
                        persona_dict[(field, "median")] = group[field].median()
                        persona_dict[(field, "IQR")] = group[field].quantile(
                            0.75
                        ) - group[field].quantile(0.25)

        persona_list.append(persona_dict)

    persona_df = DataFrame(persona_list)
    # 멀티인덱스로 변환 (단일 인덱스는 그대로)
    persona_df.columns = MultiIndex.from_tuples(persona_df.columns)  # type: ignore
    # 군집 레이블(cluster)을 인덱스로 설정
    persona_df.set_index((cluster, ""), inplace=True)
    persona_df.index.name = cluster
    return persona_df


# ===================================================================
# 엘보우 포인트와 실루엣 점수를 통해 최적의 K값을 결정하는 함수.
# ===================================================================
def kmeans_best_k(
    data: DataFrame,
    k_range: list | tuple = [2, 11],
    S: float = 0.1,
    random_state: int = RANDOM_STATE,
    plot: bool = True,
) -> int:
    """
    엘보우 포인트와 실루엣 점수를 통해 최적의 K값을 결정하는 함수.
    Args:
        data (DataFrame): 군집화할 데이터프레임.
        k_range (list | tuple, optional): K값의 범위 지정. 기본값은 [2, 11].
        S (float, optional): KneeLocator의 민감도 파라미터. 기본값 0.1.
        random_state (int, optional): 랜덤 시드. 기본값은 RANDOM_STATE.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.

    Returns:
        int: 최적의 K값.

    Examples:
        ```python
        from hossam import *
        data = hs_util.load_data('iris')
        best_k = hs_cluster.kmeans_best_k(data.iloc[:, :-1])
        ```
    """

    elbow_k, _ = kmeans_elbow(
        data=data,
        k_range=k_range,
        S=S,
        random_state=random_state,
        plot=True if plot else False,
    )

    silhouette_df = kmeans_silhouette(
        data=data,
        k_range=k_range,
        random_state=random_state,
        plot="both" if plot else False,
    )

    silhouette_k = silhouette_df.sort_values(
        by="silhouette_score", ascending=False
    ).iloc[0]["k"]

    if elbow_k == silhouette_k:
        best_k = elbow_k
    else:
        best_k = min(elbow_k, silhouette_k)

    print(f"Elbow K: {elbow_k}, Silhouette K: {silhouette_k} => Best K: {best_k}")
    return best_k


# ===================================================================
# DBSCAN 군집화 모델을 적합하는 함수.
# ===================================================================
def __dbscan_fit(
    data: DataFrame, eps: float = 0.5, min_samples: int = 5, **params
) -> tuple[DBSCAN, DataFrame, DataFrame]:
    """
    DBSCAN 군집화 모델을 적합하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        eps (float, optional): 두 샘플이 같은 군집에 속하기 위한 최대 거리. 기본값 0.5.
        min_samples (int, optional): 핵심점이 되기 위한 최소 샘플 수. 기본값 5.
        **params: DBSCAN에 전달할 추가 파라미터.

    Returns:
        tuple: (estimator, df)
            - estimator: 적합된 DBSCAN 모델.
            - df: 클러스터 및 벡터 유형이 포함된 데이터 프레임.
            - result_df: 군집화 요약 통계 데이터 프레임.

    """
    df = data.copy()
    estimator = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, **params)
    estimator.fit(df)
    df["cluster"] = estimator.labels_

    # 기본적으로 모두 외곽 벡터로 지정
    df["vector"] = "border"

    # 핵심 벡터인 경우 'core'로 지정
    df.loc[estimator.core_sample_indices_, "vector"] = "core"

    # 노이즈 분류
    df.loc[df["cluster"] == -1, "vector"] = "noise"

    labels = estimator.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.mean(labels == -1)

    result_df = DataFrame(
        {
            "eps": [eps],
            "min_samples": [min_samples],
            "n_clusters": [n_clusters],
            "noise_ratio": [noise_ratio],
        }
    )

    return estimator, df, result_df


# ===================================================================
# DBSCAN 군집화에서 최적의 eps 값을 탐지하는 함수.
# ===================================================================
def dbscan_eps(
    data: DataFrame,
    min_samples: int = 5,
    delta_ratio: float = 0.3,
    step_ratio: float = 0.05,
    S: float = 0.1,
    plot: bool = True,
    title: str | None = None,
    palette: str | None = None,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: int = hs_plot.config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
) -> tuple[float, np.ndarray]:
    """
    DBSCAN 군집화에서 최적의 eps 값을 탐지하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        min_samples (int, optional): 핵심점이 되기 위한 최소 샘플 수. 기본값 5.
        delta_ratio (float, optional): eps 탐색 범위 비율. 기본값 0.3.
        step_ratio (float, optional): eps 탐색 스텝 비율. 기본값 0.05.
        S (float, optional): KneeLocator의 민감도 파라미터. 기본값 0.1.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.
        title (str | None, optional): 플롯 제목.
        palette (str | None, optional): 색상 팔레트 이름.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (float, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Returns:
        tuple: (best_eps, eps_grid)
            - best_eps: 최적의 eps 값
            - eps_grid: 탐색할 eps 값의 그리드 배열

    Examples:
        ```python
        from hossam import *
        data = hs_util.load_data('iris')
        best_eps, eps_grid = hs_cluster.dbscan_eps(data, plot=True)
        ```
    """

    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    # 각 포인트에 대해 k번째 최근접 이웃까지의 거리 추출
    k_distances = distances[:, -1]
    k_distances.sort()

    # 엘보우 포인트 탐지
    _, best_eps = elbow_point(
        x=list(range(1, len(k_distances) + 1)),
        y=k_distances,
        dir="right,down",
        S=S,
        plot=plot,
        title=title,
        marker=None,
        width=width,
        height=height,
        linewidth=linewidth,
        palette=palette,
        save_path=save_path,
        ax=ax,
    )

    eps_min = best_eps * (1 - delta_ratio)
    eps_max = best_eps * (1 + delta_ratio)
    step = best_eps * step_ratio

    eps_grid = np.arange(eps_min, eps_max + step, step)

    return best_eps, eps_grid

# ===================================================================
# DBSCAN 군집화 모델을 적합하고 최적의 eps 값을 탐지하는 함수.
# ===================================================================
def dbscan_fit(
    data: DataFrame,
    eps: float | list | np.ndarray | None = None,
    min_samples: int = 5,
    ari_threshold: float = 0.9,
    noise_diff_threshold: float = 0.05,
    plot: bool = True,
    **params,
) -> tuple[DBSCAN, DataFrame, DataFrame]:
    """
    DBSCAN 군집화 모델을 적합하고 최적의 eps 값을 탐지하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        eps (float | list | np.ndarray | None, optional): eps 값 또는 리스트.
            None이면 최적의 eps 값을 탐지함. 기본값 None.
        min_samples (int, optional): 핵심점이 되기 위한 최소 샘플수. 기본값 5.
        ari_threshold (float, optional): 안정 구간 탐지를 위한 ARI 임계값. 기본값 0.9.
        noise_diff_threshold (float, optional): 안정 구간 탐지를 위한 노이즈 비율 변화 임계값. 기본값 0.05.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.
        **params: DBSCAN에 전달할 추가 파라미터.

    Returns:
        tuple: (estimator, cluster_df, result_df)
            - estimator: 적합된 DBSCAN 모델 또는 모델 리스트(최적 eps가 여러 개인 경우).
            - cluster_df: 클러스터 및 벡터 유형이 포함된 데이터 프레임 또는 데이터 프레임 리스트(최적 eps가 여러 개인 경우).
            - result_df: eps 값에 따른 군집화 요약 통계 데이터 프레임.
    """

    # eps 값이 지정되지 않은 경우 최적의 eps 탐지
    if eps is None:
        _, eps_grid = dbscan_eps(data=data, min_samples=min_samples, plot=plot)
        eps = eps_grid

    # eps가 단일 값인 경우 리스트로 변환
    if not isinstance(eps, (list, np.ndarray)):
        eps = [eps]

    estimators = []
    cluster_dfs = []
    result_dfs: DataFrame | None = None

    with tqdm(total=len(eps) + 2) as pbar:
        pbar.set_description(f"DBSCAN Clustering")

        with futures.ThreadPoolExecutor() as executor:
            executers = []
            for i, e in enumerate(eps):
                executers.append(
                    executor.submit(
                        __dbscan_fit,
                        data=data,
                        eps=e,
                        min_samples=min_samples,
                        **params,
                    )
                )

            for i, e in enumerate(executers):
                estimator, cluster_df, result_df = e.result()
                estimators.append(estimator)
                cluster_dfs.append(cluster_df)

                if result_dfs is None:
                    result_df["ARI"] = np.nan
                    result_dfs = result_df
                else:
                    result_df["ARI"] = adjusted_rand_score(cluster_dfs[i - 1]["cluster"], cluster_df["cluster"])  # type: ignore
                    result_dfs = concat([result_dfs, result_df], ignore_index=True)

                pbar.update(1)

            result_dfs["cluster_diff"] = result_dfs["n_clusters"].diff().abs()  # type: ignore
            result_dfs["noise_ratio_diff"] = result_dfs["noise_ratio"].diff().abs()  # type: ignore
            result_dfs["stable"] = (  # type: ignore
                (result_dfs["ARI"] >= ari_threshold)  # type: ignore
                & (result_dfs["cluster_diff"] <= 0)  # type: ignore
                & (result_dfs["noise_ratio_diff"] <= noise_diff_threshold)  # type: ignore
            )

            # 첫 행은 비교 불가
            result_dfs.loc[0, "stable"] = False  # type: ignore
            pbar.update(1)

            if len(eps) == 1:
                result_dfs["group_id"] = 1  # type: ignore
                result_dfs["recommand"] = "unknown"  # type: ignore
            else:
                # 안정구간 도출하기
                # stable 여부를 0/1로 변환
                stable_flag = result_dfs["stable"].astype(int).values  # type: ignore

                # 연속 구간 구분용 그룹 id 생성
                group_id = (stable_flag != np.roll(stable_flag, 1)).cumsum()  # type: ignore
                result_dfs["group_id"] = group_id  # type: ignore

                # 안정구간 중 가장 긴 구간 선택
                stable_groups = result_dfs[result_dfs["stable"]].groupby("group_id")  # type: ignore

                # 각 구간의 길이 계산
                group_sizes = stable_groups.size()

                # 가장 긴 안정 구간 선택
                best_group_id = group_sizes.idxmax()

                result_dfs["recommand"] = "bad"  # type: ignore

                # 가장 긴 안정 구간에 해당하는 recommand 컬럼을 `best`로 변경
                result_dfs.loc[result_dfs["group_id"] == best_group_id, "recommand"] = "best"  # type: ignore

                # result_dfs에서 recommand가 best에 해당하는 인덱스와 같은 위치의 추정기만 추출
                best_indexes = list(result_dfs[result_dfs["recommand"] == "best"].index)  # type: ignore

                # for i in range(len(estimators) - 1, -1, -1):
                #     if i not in best_indexes:
                #         del estimators[i]
                #         del cluster_dfs[i]

            pbar.update(1)

    # best 모델 선정: recommand=='best'인 인덱스의 estimator/cluster_df만 반환
    if len(estimators) == 1:

        if plot:
            hs_plot.scatterplot(
                df=cluster_dfs[0],
                xname=cluster_dfs[0].columns[0],
                yname=cluster_dfs[0].columns[1],
                hue="cluster",
                vector="vector",
                title=f"DBSCAN Clustering (eps={estimators[0].eps}, min_samples={estimators[0].min_samples})",
                outline=True
            )

        return estimators[0], cluster_dfs[0], result_dfs # type: ignore
    
    # recommand=='best'인 인덱스 추출 (여러 개면 첫 번째)
    best_indexes = list(result_dfs[result_dfs["recommand"] == "best"].index) # type: ignore
    if not best_indexes:
        # fallback: 첫 번째
        best_index = 0
    else:
        best_index = best_indexes[0]

    best_estimator = estimators[best_index]
    best_cluster_df = cluster_dfs[best_index]

    if plot:
        hs_plot.scatterplot(
            df=best_cluster_df,
            xname=best_cluster_df.columns[0],
            yname=best_cluster_df.columns[1],
            hue="cluster",
            vector="vector",
            title=f"DBSCAN Clustering (eps={best_estimator.eps}, min_samples={best_estimator.min_samples})",
            outline=True
        )

    return best_estimator, best_cluster_df, result_dfs # type: ignore


# ===================================================================
# 단일 계층적 군집화 모델을 적합하는 함수.
# ===================================================================
def __agg_fit(
    data: DataFrame,
    n_clusters: int = 3,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    plot: bool = False,
    compute_distances: bool = True,
    **params,
) -> tuple[AgglomerativeClustering, DataFrame, float]:
    """
    단일 계층적 군집화 모델을 적합하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        n_clusters (int, optional): 군집 개수. 기본값 3.
        linkage (str, optional): 병합 기준. 기본값 "ward".
        compute_distances (bool, optional): 거리 행렬 계산 여부. 기본값 True.
        plot (bool, optional): True면 결과를 시각화함. 기본값 False.
        **params: AgglomerativeClustering에 전달할 추가 파라미터.

    Returns:
        tuple: (estimator, df, score)
            - estimator: 적합된 AgglomerativeClustering 모델.
            - df: 클러스터 결과가 포함된 데이터 프레임.
            - score: 실루엣 점수.

    """
    df = data.copy()
    estimator = AgglomerativeClustering(
        n_clusters=n_clusters, compute_distances=compute_distances, linkage=linkage, **params
    )
    estimator.fit(data)
    df["cluster"] = estimator.labels_
    score = float(silhouette_score(X=data, labels=df["cluster"]))

    if plot:
        hs_plot.visualize_silhouette(estimator=estimator, data=data)

    return estimator, df, score


def agg_fit(
    data: DataFrame,
    n_clusters: int | list[int] | np.ndarray = 3,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    plot: bool = False,
    **params,
) -> tuple[AgglomerativeClustering | list[AgglomerativeClustering], DataFrame | list[DataFrame], DataFrame]:
    """
    계층적 군집화 모델을 적합하는 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        n_clusters (int | list[int] | np.ndarray, optional): 군집 개수 또는 개수 리스트. 기본값 3.
        linkage (str, optional): 병합 기준. 기본값 "ward".
        plot (bool, optional): True면 결과를 시각화함. 기본값 False.
        **params: AgglomerativeClustering에 전달할 추가 파라미터.

    Returns:
        tuple: (estimator(s), df(s), score_df)
            - estimator(s): 적합된 AgglomerativeClustering 모델 또는 모델 리스트 (n_clusters가 리스트일 때 리턴도 리스트로 처리됨).
            - df(s): 클러스터 결과가 포함된 데이터 프레임 또는 데이터 프레임 리스트(n_cluseters가 리스트일 때 리턴되 리스트로 처리됨).
            - score_df: 각 군집 개수에 대한 실루엣 점수 데이터프레임.

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        estimators, cluster_dfs, score_df = hs_cluster.agg_fit(data.iloc[:, :-1], n_clusters=[2,3,4])
        ```
    """
    compute_distances = False

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
        compute_distances = True
    else:
        n_clusters = list(range(n_clusters[0], n_clusters[-1]))

    estimators = []
    cluster_dfs = []
    scores = []

    with tqdm(total=len(n_clusters)*2) as pbar:
        pbar.set_description(f"Agglomerative Clustering")

        with futures.ThreadPoolExecutor() as executor:
            executers = []
            for k in n_clusters:
                executers.append(
                    executor.submit(
                        __agg_fit,
                        data=data,
                        n_clusters=k,
                        linkage=linkage,
                        plot=False,
                        compute_distances=compute_distances,
                        **params,
                    )
                )
                pbar.update(1)

            for e in executers:
                estimator, cluster_df, score = e.result()
                estimators.append(estimator)
                cluster_dfs.append(cluster_df)
                scores.append({"k": estimator.n_clusters, "silhouette_score": score})

                if plot:
                    hs_plot.visualize_silhouette(
                        estimator=estimator,
                        data=data,
                        outline=True,
                        title=f"Agglomerative Clustering Silhouette (k={estimator.n_clusters})",
                    )

                pbar.update(1)

    score_df = DataFrame(scores)
    score_df.sort_values(by="silhouette_score", ascending=False, inplace=True)

    return (
        estimators[0] if len(estimators) == 1 else estimators,  # type: ignore
        cluster_dfs[0] if len(cluster_dfs) == 1 else cluster_dfs,
        score_df,  # type: ignore
    )
