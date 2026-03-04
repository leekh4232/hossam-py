# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import numpy as np
from typing import Literal
import concurrent.futures as futures

# -------------------------------------------------------------
from pycallgraphix.wrapper import register_method

# -------------------------------------------------------------
from pandas import DataFrame

# -------------------------------------------------------------
import seaborn as sb
import matplotlib.pyplot as plt

# -------------------------------------------------------------
from kneed import KneeLocator

# -------------------------------------------------------------
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram

# -------------------------------------------------------------
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------
from .plot import my_lineplot, my_convex_hull
from .core import get_random_state


# -------------------------------------------------------------

def my_kmeans_cluster(
    data: DataFrame,
    n_clusters: int,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=get_random_state(),
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> KMeans:
    """KMmeans 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int): 클러스터 개수
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): 알고리즘. Defaults to "lloyd".

    Returns:
        KMeans
    """
    estimator = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm=algorithm,
    )
    estimator.fit(data)

    # 속성 확장
    estimator.n_clusters = n_clusters
    estimator.silhouette = silhouette_score(data, estimator.labels_)

    if plot:
        my_cluster_plot(estimator, data, figsize=figsize, dpi=dpi)

    return estimator


# -------------------------------------------------------------

def my_elbow_point(
    x: list,
    y: list,
    dir: Literal["left,down", "left,up", "right,down", "right,up"] = "left,down",
    title: str = None,
    xname: str = None,
    yname: str = None,
    S: float = 0.1,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    marker: str = None,
    linewidth: int = 1,
) -> tuple:
    """엘보우 포인트를 찾는다.

    Args:
        x (list): x축 데이터
        y (list): y축 데이터
        dir (str, optional): 그래프가 볼록한 방향. Defaults to "left,down".
        title (str, optional): 그래프 제목. Defaults to "Elbow Method".
        xname (str, optional): x축 이름. Defaults to "n_clusters".
        yname (str, optional): y축 이름. Defaults to "inertia".
        S (float, optional): 정밀도. Defaults to 1.0.
        plot (bool, optional): 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.

    Returns:
        int: _description_
    """
    # left,down  convex, decresing
    # left,up concave, increasing
    # right,down convex, increasing
    # right,up  concave, decresing

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
            if xname:
                ax.set_xlabel(xname)

            if yname:
                ax.set_ylabel(yname)

            if title:
                ax.set_title(title)

            ax.axvline(best_x, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(best_y, color="red", linestyle="--", linewidth=0.7)

        if title:
            title = title + " (Elbow Point : %.1f x %.1f)" % (best_x, best_y)
        else:
            title = "Elbow Method (Elbow Point : %.1f x %.1f)" % (best_x, best_y)

        my_lineplot(
            df=None,
            xname=x,
            yname=y,
            marker=marker,
            linewidth=linewidth,
            figsize=figsize,
            dpi=dpi,
            callback=hvline,
        )

    return (best_x, best_y)


# -------------------------------------------------------------

def __silhouette_plot(cluster: any, data: DataFrame, ax: plt.Axes) -> None:
    """실루엣 계수를 파라미터로 전달받은 ax에 시각화 한다.

    Args:
        clusters (list): 클러스터 개수 리스트
        data (DataFrame): 원본 데이터
        ax (plt.Axes): 그래프 객체
    """
    sil_avg = silhouette_score(X=data, labels=cluster.labels_)
    sil_values = silhouette_samples(X=data, labels=cluster.labels_)

    y_lower = 10
    ax.set_title(
        "Number of Cluster : " + str(cluster.n_clusters) + ", "
        "Silhouette Score :" + str(round(sil_avg, 3))
    )
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (cluster.n_clusters + 1) * 10])
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.grid()

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    for i in range(cluster.n_clusters):
        ith_cluster_sil_values = sil_values[cluster.labels_ == i]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--")


# -------------------------------------------------------------

def my_cluster_plot(
    estimator: any, data: DataFrame, figsize: tuple = (10, 5), dpi: int = 100, silhouette: bool = True
) -> None:
    """클러스터링 결과를 시각화한다.

    Args:
        estimator (any): 클러트링 객체
        data (DataFrame): 원본 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
    """
    df = data.copy()
    df["cluster"] = estimator.labels_

    if silhouette:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(figsize[0] * 2, figsize[1]), dpi=dpi
        )
        fig.subplots_adjust(wspace=0.1)

        # 1st Plot showing the silhouette plot
        __silhouette_plot(cluster=estimator, data=data, ax=ax1)
    else:
        fig, ax2 = plt.subplots(
            nrows=1, ncols=1, figsize=(figsize[0], figsize[1]), dpi=dpi
        )

    # 2nd Plot showing the actual clusters formed
    xname = data.columns[0]
    yname = data.columns[1]

    for c in df["cluster"].unique():
        if c == -1:
            continue

        # 한 종류만 필터링한 결과에서 두 변수만 선택
        df_c = df.loc[df["cluster"] == c, [xname, yname]]

        try:
            # 외각선 좌표 계산
            hull = ConvexHull(df_c)

            # 마지막 좌표 이후에 첫 번째 좌표를 연결
            points = np.append(hull.vertices, hull.vertices[0])

            ax2.plot(
                df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":"
            )
            ax2.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)
        except:
            pass

    if hasattr(estimator, "core_sample_indices_"):
        df["vector"] = "O"
        df.loc[estimator.core_sample_indices_, "vector"] = "C"
        # 노이즈 분류
        df.loc[df["cluster"] == -1, "vector"] = "N"
        sb.scatterplot(
            data=df, x=xname, y=yname, hue="cluster", style="vector", ax=ax2, s=70
        )
    else:
        sb.scatterplot(data=df, x=xname, y=yname, hue="cluster", ax=ax2, s=70)

    # 중심점이 있을 경우 중심점 표시 --> KMeans
    if hasattr(estimator, "cluster_centers_"):
        # Labeling the clusters
        centers = estimator.cluster_centers_
        # Draw white circles at cluster centers
        sb.scatterplot(
            x=centers[:, 0],
            y=centers[:, 1],
            marker="o",
            color="white",
            alpha=1,
            s=200,
            edgecolor="r",
            ax=ax2,
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.grid()

    plt.show()
    plt.close()


# -------------------------------------------------------------

def my_silhouette_plot(
    clusters: list,
    data: DataFrame,
    figsize: tuple = (8, 6),
    dpi: int = 100,
    cols: int = 3,
) -> None:
    """실루엣 계수를 시각화한다.

    Args:
        clusters (list): 클러스터 개수 리스트
        data (DataFrame): 원본 데이터
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
        cols (int, optional): 열 개수. Defaults to 3.
    """
    rows = (len(clusters) + cols - 1) // cols
    fig, ax = plt.subplots(
        nrows=rows, ncols=cols, figsize=(figsize[0] * cols, figsize[1] * rows), dpi=dpi
    )

    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    with futures.ThreadPoolExecutor() as executor:
        for i in range(0, len(clusters)):
            cl = clusters[i]
            r = i // cols
            c = i % cols
            executor.submit(__silhouette_plot, cluster=cl, data=data, ax=ax[r, c])

    plt.show()
    plt.close()


# -------------------------------------------------------------

def my_kmeans(
    data: DataFrame,
    min_clusters: int = 3,
    max_clusters: int | list = 10,
    init: Literal["k-means++", "random"] = "k-means++",
    max_iter: int = 500,
    random_state=get_random_state(),
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd",
    scoring: Literal["elbow", "e", "silhouette", "s"] = "elbow",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DataFrame:
    """클러스터 개수에 따른 이너셔 값을 계산한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int | list, optional): 최대 클러스터 개수. 정수로 전달할 경우 `2`부터 주어진 개수까지 반복 수행한다. Defaults to 10.
        init (Literal["k-means++", "random"], optional): 초기화 방법. Defaults to "k-means++".
        max_iter (int, optional): 최대 반복 횟수. Defaults to 500.
        random_state (int, optional): 난수 시드. Defaults to 0.
        algorithm (Literal["lloyd", "elkan", "auto", "full"], optional): _description_. Defaults to "lloyd".

    Returns:
        DataFrame: _description_
    """
    n_clusters = list(range(min_clusters, max_clusters + 1))

    with futures.ThreadPoolExecutor() as executor:
        results = []
        for n in n_clusters:
            results.append(
                executor.submit(
                    my_kmeans_cluster,
                    data,
                    n_clusters=n,
                    init=init,
                    max_iter=max_iter,
                    random_state=random_state,
                    algorithm=algorithm,
                    plot=False,
                )
            )

        # 비동기처리로 생성된 군집객체들을 수집 --> 비동기이므로 먼저 종료된 순서대로 수집된다.
        kmeans_list = [r.result() for r in futures.as_completed(results)]

        # 클러스터 개수로 정렬
        kmeans_list = sorted(kmeans_list, key=lambda x: x.n_clusters)

        # 클러스터 개수만 별도로 추출
        cluster_list = [k.n_clusters for k in kmeans_list]

        # 최적 모델을 저장할 객체
        best_model = None

        if scoring == "elbow" or scoring == "e":
            inertia = [k.inertia_ for k in kmeans_list]

            best_k, best_y = my_elbow_point(
                x=cluster_list,
                y=inertia,
                dir="left,down",
                xname="n_cluster",
                yname="inertia",
                title="Elbow Method",
                plot=plot,
                marker="o",
                linewidth=2,
                figsize=figsize,
                dpi=dpi,
            )

            best_model = next(filter(lambda x: x.n_clusters == best_k, kmeans_list))
        elif scoring == "silhouette" or scoring == "s":
            best_model = max(kmeans_list, key=lambda x: x.silhouette)

            if plot:
                my_silhouette_plot(kmeans_list, data, figsize=(8, 6), dpi=dpi)

        # 최종 군집 결과를 시각화 한다.
        if plot:
            my_cluster_plot(best_model, data, figsize=figsize, dpi=dpi)

        return best_model


# -------------------------------------------------------------

def my_dbscan_cluster(
    data: DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Literal["euclidean", "manhattan", "cosine", "jaccard"] = "euclidean",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DBSCAN:
    """DBSCAN 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        eps (float, optional): 최대 이웃 거리. Defaults to 0.5.
        min_samples (int, optional): eps 내 최소 이웃 수. 일반적으로 minPts라고 함. Defaults to 5.
        metric (Literal["euclidean", "manhattan", "cosine", "jaccard"], optional): 거리 측정 방법.	. Defaults to "euclidean".
        algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"], optional): 이웃 검색 알고리즘.. Defaults to "auto".

    Returns:
        DBSCAN
    """

    estimator = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, n_jobs=-1
    )
    estimator.fit(X=data)

    # 속성 확장
    estimator.n_clusters = len(list(set(estimator.labels_)))

    # 노이즈가 포함된 경우
    if -1 in estimator.labels_:
        estimator.n_clusters -= 1

    estimator.silhouette = silhouette_score(data, estimator.labels_)

    if plot:
        my_cluster_plot(estimator, data, figsize=figsize, dpi=dpi)

    return estimator


# -------------------------------------------------------------

def my_n_neighbors(
    data: DataFrame,
    k: int = 3,
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
):
    """KNN 알고리즘을 시각화한다.

    Args:
        data (DataFrame): 원본 데이터
        k (int, optional): 이웃 수. Defaults to 3.
        plot (bool, optional): 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X=data)

    # 한 점의 최근접 이웃 찾기
    distance, indices = neighbors_fit.kneighbors(X=data)

    # 가까운 순서대로 정렬
    s_distance = np.sort(distance, axis=0)

    # 마지막 데이터와의 거리(k−1번째)만을 추출해서 1차원 리스트로 재구성하여 그래프의 y으로 삼음
    target = s_distance[:, k - 1]

    # 엘보우 포인트 계산
    best_k, best_y = my_elbow_point(
        x=list(range(1, len(target) + 1)),
        y=target,
        dir="right,down",
        title="KNN Method / k=%d" % k,
        xname="Points sorted by distance",
        yname="distance",
        plot=plot,
        figsize=figsize,
        dpi=dpi,
    )

    return best_y


# -------------------------------------------------------------

def my_knn_dbscan(
    data: DataFrame,
    k: int = 3,
    metric: Literal["euclidean", "manhattan", "cosine", "jaccard"] = "euclidean",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DBSCAN:
    """주어진 k값에 대한 eps와 minPts를 구한 후 이를 토대로 DBSCAN 알고리즘을 수행한다.

    Args:
        data (DataFrame):
        k (int, optional): _description_. Defaults to 3.
        metric (Literal["euclidean", "manhattan", "cosine", "jaccard"], optional): _description_. Defaults to "euclidean".
        algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"], optional): _description_. Defaults to "auto".
        plot (bool, optional): _description_. Defaults to True.
        figsize (tuple, optional): _description_. Defaults to (10, 5).
        dpi (int, optional): _description_. Defaults to 100.

    Returns:
        DBSCAN
    """

    eps = my_n_neighbors(data, k=k, plot=plot, figsize=figsize, dpi=dpi)

    try:
        estimator = my_dbscan_cluster(
            data=data, eps=eps, min_samples=k, metric=metric, algorithm=algorithm
        )
    except Exception as e:
        print(f"\x1b[31m클러스터링에 실패했습니다.\x1b[0m")
        print(f"\x1b[31m({e})\x1b[0m")
        return None

    return estimator


# -------------------------------------------------------------

def my_dbscan(
    data: DataFrame,
    k: int | list = 3,
    metric: Literal["euclidean", "manhattan", "cosine", "jaccard"] = "euclidean",
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> DBSCAN:
    """주어진 k값에 대한 eps와 minPts를 구한 후 이를 토대로 DBSCAN 알고리즘을 수행한다.

    Args:
        data (DataFrame):
        k (int, optional): _description_. Defaults to 3.
        metric (Literal["euclidean", "manhattan", "cosine", "jaccard"], optional): _description_. Defaults to "euclidean".
        algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"], optional): _description_. Defaults to "auto".
        plot (bool, optional): _description_. Defaults to True.
        figsize (tuple, optional): _description_. Defaults to (10, 5).
        dpi (int, optional): _description_. Defaults to 100.

    Returns:
        DBSCAN List
    """

    if isinstance(k, int):
        k = list(range(2, k + 1))

    for i in range(len(k)):
        try:
            print("=======[k=%d]=======" % k[i])
            k[i] = my_knn_dbscan(
                data=data,
                k=k[i],
                metric=metric,
                algorithm=algorithm,
                plot=plot,
                figsize=figsize,
                dpi=dpi,
            )
            print("\n\n")
        except Exception as e:
            print(f"\x1b[31m클러스터링에 실패했습니다.\x1b[0m")
            print(f"\x1b[31m({e})\x1b[0m")
            k[i] = None

    # 클러스터 개수로 정렬
    dbscan_list = sorted(k, key=lambda x: x.n_clusters)

    # 클러스터 개수만 별도로 추출
    cluster_list = [k.n_clusters for k in dbscan_list]

    # 최적 모델을 저장할 객체
    best_model = None

    # 최적 모델 선택
    best_model = max(dbscan_list, key=lambda x: x.silhouette)

    if plot:
        my_cluster_plot(best_model, data, figsize=figsize, dpi=dpi)

    return best_model


# -------------------------------------------------------------

def my_agg_cluster(
    data: DataFrame,
    n_clusters: int = 2,
    metric: Literal["euclidean", "l1", "l2", "manhattan", "cosine"] = "euclidean",
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    distance_threshold: float = None,
    compute_full_tree: Literal["auto", True, False] = "auto",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> AgglomerativeClustering:
    """AgglomerativeClustering 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int, optional): 클러스터 개수. Defaults to 2.
        metric (Literal["euclidean", "l1", "l2", "manhattan", "cosine"], optional): 군집 간의 유사도를 계산하는 방법. 'ward' 연결 방식일 때는 'euclidean'만 사용.
        linkage (Literal["ward", "complete", "average", "single"], optional): 군집 간의 거리(유사도)를 어떻게 계산할지 결정하는 기준. Defaults to "ward".
        distance_threshold (float, optional): 군집 간 최대 거리. None이 아닐 경우 `n_clusters`는 무시되고 거리 임계값에 도달할 때까지 군집화 진행. Defaults to None.
        compute_full_tree (Literal["auto", True, False], optional): _전체 트리를 계산할지 여부. 'auto'일 경우, `n_clusters`가 샘플 수보다 적으면 전체 트리 계산. Defaults to "auto".

    Returns:
        AgglomerativeClustering: _description_
    """
    estimator = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_full_tree=compute_full_tree,
        compute_distances=True,
    )
    estimator.fit(data)
    estimator.n_clusters = n_clusters
    estimator.silhouette = silhouette_score(data, estimator.labels_)

    if plot:
        my_cluster_plot(estimator, data, figsize=figsize, dpi=dpi)
        my_dendrogram(
            estimator,
            figsize=(int(figsize[0] * 1.5), int(figsize[1] * 1.5)),
            dpi=dpi,
        )

    return estimator


# -------------------------------------------------------------

def __dendrogram_source(estimator: AgglomerativeClustering) -> np.ndarray:
    """덴드로그램을 위한 데이터를 생성한다.

    Args:
        estimator (AgglomerativeClustering): AgglomerativeClustering 객체

    Returns:
        np.ndarray
    """
    counts = np.zeros(estimator.children_.shape[0])
    n_samples = len(estimator.labels_)

    for i, merge in enumerate(estimator.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [estimator.children_, estimator.distances_, counts]
    ).astype(float)

    # 시각화에 필요한 배열 리턴
    return linkage_matrix


# -------------------------------------------------------------

def my_dendrogram(
    estimator: AgglomerativeClustering,
    p: int = 0,
    leaf_rotation: int = 0,
    leaf_font_size: int = 12,
    count_sort: Literal["ascending", "descending"] = "ascending",
    truncate_mode: Literal[None, "lastp", "level"] = "lastp",
    figsize: tuple = (15, 7),
    dpi: int = 100,
) -> None:
    """덴드로그램을 시각화한다.

    Args:
        estimator (AgglomerativeClustering): AgglomerativeClustering 객체
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.
    """
    if p == 0:
        p = int(len(estimator.labels_) * 0.35)

    # 덴드로그램 데이터 생성
    Z = __dendrogram_source(estimator)

    # 덴드로그램 그리기
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(label="Hierarchical Clustering Dendrogram")
    dendrogram(
        Z=Z,
        p=p,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        truncate_mode=truncate_mode,
        count_sort=count_sort,
        show_contracted=True,
        show_leaf_counts=True,
    )
    plt.show()
    plt.close()


# -------------------------------------------------------------

def my_agg(
    data: DataFrame,
    n_clusters: int | list = 10,
    metric: Literal["euclidean", "l1", "l2", "manhattan", "cosine"] = "euclidean",
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    distance_threshold: float = None,
    compute_full_tree: Literal["auto", True, False] = "auto",
    plot: bool = True,
    figsize: tuple = (10, 5),
    dpi: int = 100,
) -> AgglomerativeClustering:
    """AgglomerativeClustering 알고리즘을 수행한다.

    Args:
        data (DataFrame): 원본 데이터
        n_clusters (int, optional): 클러스터 개수. Defaults to 2.
        metric (Literal["euclidean", "l1", "l2", "manhattan", "cosine"], optional): 군집 간의 유사도를 계산하는 방법. 'ward' 연결 방식일 때는 'euclidean'만 사용.
        linkage (Literal["ward", "complete", "average", "single"], optional): 군집 간의 거리(유사도)를 어떻게 계산할지 결정하는 기준. Defaults to "ward".
        distance_threshold (float, optional): 군집 간 최대 거리. None이 아닐 경우 `n_clusters`는 무시되고 거리 임계값에 도달할 때까지 군집화 진행. Defaults to None.
        compute_full_tree (Literal["auto", True, False], optional): _전체 트리를 계산할지 여부. 'auto'일 경우, `n_clusters`가 샘플 수보다 적으면 전체 트리 계산. Defaults to "auto".
        plot (bool, optional): 그래프 표시 여부. Defaults to True.
        figsize (tuple, optional): 그래프 크기. Defaults to (10, 5).
        dpi (int, optional): 해상도. Defaults to 100.

    Returns:
        AgglomerativeClustering: _description_
    """

    if isinstance(n_clusters, int):
        n_clusters = list(range(2, n_clusters + 1))

    with futures.ThreadPoolExecutor() as executor:
        for i in range(len(n_clusters)):
            n_clusters[i] = executor.submit(
                my_agg_cluster,
                data,
                n_clusters=n_clusters[i],
                metric=metric,
                linkage=linkage,
                distance_threshold=distance_threshold,
                compute_full_tree=compute_full_tree,
                plot=False,
            )

        # 비동기처리로 생성된 군집객체들을 수집 --> 비동기이므로 먼저 종료된 순서대로 수집된다.
        agg_list = [r.result() for r in futures.as_completed(n_clusters)]

        # 클러스터 개수로 정렬
        agg_list = sorted(agg_list, key=lambda x: x.n_clusters)

        # 클러스터 개수만 별도로 추출
        cluster_list = [k.n_clusters for k in agg_list]

        # 최적 모델을 저장할 객체
        best_model = None

        # 최적 모델 선택
        best_model = max(agg_list, key=lambda x: x.silhouette)

        if plot:
            my_cluster_plot(best_model, data, figsize=figsize, dpi=dpi)
            my_dendrogram(
                best_model,
                figsize=(int(figsize[0] * 1.5), int(figsize[1] * 1.5)),
                dpi=dpi,
            )

        return best_model
