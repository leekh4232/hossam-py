# -*- coding: utf-8 -*-
# ===================================================================
# 패키지 참조
# ===================================================================
from tqdm.auto import tqdm
import concurrent.futures as futures
from itertools import combinations

from typing import Literal, Callable
from kneed import KneeLocator
from pandas import Series, DataFrame, MultiIndex
from matplotlib.pyplot import Axes  # type: ignore
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from scipy.stats import normaltest

import numpy as np
from . import hs_plot

import numpy as np

RANDOM_STATE = 4232


def kmeans_fit(
    data: DataFrame, n_clusters: int, random_state: int = RANDOM_STATE, **params
) -> tuple[KMeans, np.ndarray]:
    """
    K-평균 군집화 모델을 적합하는 내부 함수.

    Args:
        data (DataFrame): 군집화할 데이터프레임.
        n_clusters (int): 군집 개수.
        random_state (int, optional): 랜덤 시드. 기본값은 RANDOM_STATE.
        **params: KMeans에 전달할 추가 파라미터.

    Returns:
        KMeans: 적합된 KMeans 모델.
        ndarray: 각 데이터 포인트의 클러스터 할당 배열.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **params)
    kmeans.fit(data)
    cluster = kmeans.predict(data)
    return kmeans, cluster


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
    dpi: int = hs_plot.config.dpi,
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
        dpi (int, optional): 플롯 해상도.
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
        kmeans, _ = kmeans_fit(data=data, n_clusters=k, random_state=random_state)
        inertia_list.append(kmeans.inertia_)

    best_k, _ = elbow_point(
        x=list(r),
        y=inertia_list,
        dir="left,down",
        S=S,
        plot=plot,
        title=title,
        marker=marker,
        width=width,
        height=height,
        dpi=dpi,
        linewidth=linewidth,
        save_path=save_path,
        ax=ax,
        callback=callback,
        **params,
    )

    return best_k, inertia_list


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
    dpi: int = hs_plot.config.dpi,
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
        dpi (int, optional): 플롯 해상도.
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
            estimator, cluster = kmeans_fit(
                data=data, n_clusters=k, random_state=random_state
            )
            s_score = silhouette_score(X=data, labels=cluster)
            return s_score, estimator

        with futures.ThreadPoolExecutor() as executor:
            for k in klist:
                pbar.set_description(f"K-Means Silhouette: k={k}")
                executed = executor.submit(__process_k, k)
                s_score, estimator = executed.result()
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
                            dpi=dpi,
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
                            dpi=dpi,
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
                            dpi=dpi,
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
    dpi: int = hs_plot.config.dpi,
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
        dpi (int, optional): 플롯 해상도.
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
                best_x + 0.1,
                best_y + 0.1,
                "Best K=%d" % best_x,
                fontsize=8,
                ha="left",
                va="bottom",
                color="r",
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
            dpi=dpi,
            save_path=save_path,
            callback=hvline,
            ax=ax,
            **params,
        )

    return best_x, best_y


def cluster_plot(
    estimator: KMeans,
    data: DataFrame,
    hue: str | None = None,
    fields: list[list] = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = False,
    width: int = hs_plot.config.width,
    height: int = hs_plot.config.height,
    linewidth: float = hs_plot.config.line_width,
    dpi: int = hs_plot.config.dpi,
    save_path: str | None = None,
    ax: Axes | None = None,
):
    """
    데이터프레임의 여러 필드 쌍에 대해 군집 산점도를 그리는 함수.

    Args:
        estimator (KMeans): KMeans 군집화 모델.
        data (DataFrame): 군집화할 데이터프레임.
        hue (str | None, optional): 군집 레이블 컬럼명. 지정되지 않으면 estimator의 레이블 사용.
        fields (list[list], optional): 시각화할 필드 쌍 리스트. 기본값 None이면 수치형 컬럼의 모든 조합 사용.
        title (str | None, optional): 플롯 제목.
        palette (str | None, optional): 색상 팔레트 이름.
        outline (bool, optional): True면 데이터 포인트 외곽선 표시. 기본값 False.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        linewidth (float, optional): 선 두께.
        dpi (int, optional): 플롯 해상도.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Examples:
        ```python
        from hossam import *

        data = hs_util.load_data('iris')
        data['cluster'] = hs_cluster.kmeans_fit(data.iloc[:, :-1], n_clusters=3)[1]
        hs_cluster.cluster_plot(data, hue='cluster')
        ```
    """

    if fields is None:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("데이터프레임에 수치형 컬럼이 2개 이상 필요합니다.")

        # fields의 모든 조합 생성
        fields = [list(pair) for pair in combinations(numeric_cols, 2)]

    with tqdm(total=len(fields)) as pbar:
        pbar.set_description("K-Means Cluster Plotting")

        for field_pair in fields:
            xname, yname = field_pair

            hs_plot.cluster_plot(
                estimator=estimator,
                data=data,
                xname=xname,
                yname=yname,
                hue=hue,
                title=title,
                palette=palette,
                outline=outline,
                width=width,
                height=height,
                dpi=dpi,
                save_path=save_path,
                ax=ax,
            )

            pbar.update(1)


def persona(
    data: DataFrame,
    cluster: str | Series | np.ndarray | list | dict,
    fields: list[str] | None = None,
    full: bool = False
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
        data['cluster'] = hs_cluster.kmeans_fit(data.iloc[:, :-1], n_clusters=3)[1]
        persona_df = hs_cluster.persona(data, hue='cluster')
        print(persona_df)
        ```
    """
    df = data.copy()

    if fields is None:
        fields = df.select_dtypes(include=["number"]).columns.tolist()

    if isinstance(cluster, str):
        if cluster not in df.columns:
            raise ValueError(f"cluster로 지정된 컬럼 '{cluster}'이(가) 데이터프레임에 존재하지 않습니다.")
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
