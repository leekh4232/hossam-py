

# ===================================================================
# 잔차도 (선형회귀의 선형성 검정)
# ===================================================================
def ols_residplot(
    fit,
    title: str | None = None,
    lowess: bool = False,
    mse: bool = False,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """잔차도를 그린다(선택적으로 MSE 범위와 LOWESS 포함).

    회귀모형의 선형성을 시각적으로 평가하기 위한 그래프를 생성한다.
    점들이 무작위로 흩어져 있으면 선형성 가정이 만족되며,
    특정 패턴이 보이면 비선형 관계가 존재할 가능성을 시사한다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
             fit.resid와 fit.fittedvalues를 통해 잔차와 적합값을 추출한다.
        title (str|None): 그래프 제목.
        lowess (bool): LOWESS 스무딩 적용 여부.
        mse (bool): √MSE, 2√MSE, 3√MSE 대역선과 비율 표시 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn residplot 추가 인자.

    Returns:
        None

    Examples:
        ```python
        from hossam import *
        fit = my_stats.ols(data, yname='target', report=False)
        residplot(fit, lowess=True, mse=True)
        ```
    """
    outparams = False

    # fit 객체에서 잔차와 적합값 추출
    resid = fit.resid
    y_pred = fit.fittedvalues
    y = y_pred + resid  # 실제값 = 적합값 + 잔차

    if ax is None:
        fig, ax = init(width=width + 150 if mse else width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    sb.residplot(
        x=y_pred,
        y=resid,
        lowess=True,  # 잔차의 추세선 표시
        line_kws={"color": "red", "linewidth": linewidth * 0.7},  # 추세선 스타일
        scatter_kws={"edgecolor": "white", "alpha": config.alpha},
        **params
    )

    if mse:
        mse_val = mean_squared_error(y, y_pred)
        mse_sq = np.sqrt(mse_val)

        r1 = resid[(resid > -mse_sq) & (resid < mse_sq)].size / resid.size * 100
        r2 = resid[(resid > -2 * mse_sq) & (resid < 2 * mse_sq)].size / resid.size * 100
        r3 = resid[(resid > -3 * mse_sq) & (resid < 3 * mse_sq)].size / resid.size * 100

        mse_r = [r1, r2, r3]

        xmin, xmax = ax.get_xlim()  # type: ignore

        # 구간별 반투명 색상 채우기 (안쪽부터 바깥쪽으로, 진한 색에서 연한 색으로)
        colors = ["red", "green", "blue"]
        alphas = [0.15, 0.10, 0.05]  # 안쪽이 더 진하게

        # 3σ 영역 (가장 바깥쪽, 가장 연함)
        ax.axhspan(-3 * mse_sq, 3 * mse_sq, facecolor=colors[2], alpha=alphas[2], zorder=0)  # type: ignore
        # 2σ 영역 (중간)
        ax.axhspan(-2 * mse_sq, 2 * mse_sq, facecolor=colors[1], alpha=alphas[1], zorder=1)  # type: ignore
        # 1σ 영역 (가장 안쪽, 가장 진함)
        ax.axhspan(-mse_sq, mse_sq, facecolor=colors[0], alpha=alphas[0], zorder=2)  # type: ignore

        # 경계선 그리기
        for i, c in enumerate(["red", "green", "blue"]):
            ax.axhline(mse_sq * (i + 1), color=c, linestyle="--", linewidth=linewidth / 2)  # type: ignore
            ax.axhline(mse_sq * (-(i + 1)), color=c, linestyle="--", linewidth=linewidth / 2)  # type: ignore

        target = [68, 95, 99.7]
        for i, c in enumerate(["red", "green", "blue"]):
            ax.text(  # type: ignore
                s=f"{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.05,
                y=(i + 1) * mse_sq,
                color=c,
            )
            ax.text(  # type: ignore
                s=f"-{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.05,
                y=-(i + 1) * mse_sq,
                color=c,
            )

    show(save_path)  # type: ignore


# ===================================================================
# Q-Q Plot (선형회귀의 정규성 검정)
# ===================================================================
def ols_qqplot(
    fit,
    title: str | None = None,
    line: str = "s",
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """표준화된 잔차의 정규성 확인을 위한 QQ 플롯을 그린다.

    statsmodels의 qqplot 함수를 사용하여 최적화된 Q-Q plot을 생성한다.
    이론적 분위수와 표본 분위수를 비교하여 잔차의 정규성을 시각적으로 평가한다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper 등).
             fit.resid 속성을 통해 잔차를 추출하여 정규성을 확인한다.
        title (str|None): 그래프 제목.
        line (str): 참조선의 유형. 기본값 's' (standardized).
                    - 's': 표본의 표준편차와 평균을 기반으로 조정된 선 (권장)
                    - 'r': 실제 점들에 대한 회귀선 (데이터 추세 반영)
                    - 'q': 1사분위수와 3사분위수를 통과하는 선
                    - '45': 45도 대각선 (이론적 정규분포)
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: statsmodels qqplot 추가 인자.

    Returns:
        None

    Examples:
        ```python
        from hossam import *
        # 선형회귀 모형 적합
        fit = my_stats.ols(data, yname='target', report=False)
        # 표준화된 선 (권장)
        qqplot(fit)
        # 회귀선 (데이터 추세 반영)
        qqplot(fit, line='r')
        # 45도 대각선 (전통적 방식)
        qqplot(fit, line='45')
        ```
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    # fit 객체에서 잔차(residuals) 추출
    residuals = fit.resid

    # markersize 기본값 설정 (기존 크기의 2/3)
    if "markersize" not in params:
        params["markersize"] = 2

    # statsmodels의 qqplot 사용 (더 전문적이고 최적화된 구현)
    # line 옵션으로 다양한 참조선 지원
    sm_qqplot(residuals, line=line, ax=ax, **params)

    # 점의 스타일 개선: 연한 내부, 진한 테두리
    for collection in ax.collections:  # type: ignore
        # PathCollection (scatter plot의 점들)
        collection.set_facecolor("#4A90E2")  # 연한 파란색 내부
        collection.set_edgecolor("#1E3A8A")  # 진한 파란색 테두리
        collection.set_linewidth(0.8)  # 테두리 굵기
        collection.set_alpha(0.7)  # 약간의 투명도

    # 선 굵기 조정
    for line in ax.get_lines():  # type: ignore
        line.set_linewidth(linewidth)  # type: ignore

    show(save_path)  # type: ignore



# ===================================================================
# ROC 커브를 시각화 한다.
# ===================================================================
def roc_curve_plot(
    fit,
    y: np.ndarray | Series | None = None,
    X: DataFrame | np.ndarray | None = None,
    title: str | None = None,
    width: int = config.height,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """로지스틱 회귀 적합 결과의 ROC 곡선을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        y (array-like|None): 외부 데이터의 실제 레이블. 제공 시 이를 실제값으로 사용.
        X (array-like|None): 외부 데이터의 설계행렬(독립변수). 제공 시 해당 데이터로 예측 확률 계산.
        title (str|None): 그래프 제목.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Notes:
        - 실제값: `y`가 주어지면 이를 사용, 없으면 `fit.model.endog`를 사용합니다.
        - 예측 확률: `X`가 주어지면 `fit.predict(X)`를 사용, 없으면 `fit.predict(fit.model.exog)`를 사용합니다.

    Returns:
        None
    """
    outparams = False
    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    # 실제값(y_true) 결정
    if y is not None:
        y_true = np.asarray(y)
    else:
        # 학습 데이터의 종속변수 사용
        y_true = np.asarray(fit.model.endog)

    # 예측 확률 결정
    if X is not None:
        y_pred_proba = np.asarray(fit.predict(X))
    else:
        y_pred_proba = np.asarray(fit.predict(fit.model.exog))

    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # ROC 곡선 그리기
    ax.plot(fpr, tpr, color="darkorange", lw=linewidth, label=f"ROC curve (AUC = {roc_auc:.4f})")  # type: ignore
    ax.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--", label="Random Classifier")  # type: ignore

    ax.set_xlim([0.0, 1.0])  # type: ignore
    ax.set_ylim([0.0, 1.05])  # type: ignore
    ax.set_xlabel("위양성율 (False Positive Rate)", fontsize=config.label_font_size)  # type: ignore
    ax.set_ylabel("재현율 (True Positive Rate)", fontsize=config.label_font_size)  # type: ignore
    ax.set_title("ROC 곡선", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore
    ax.legend(loc="lower right", fontsize=config.label_font_size)  # type: ignore
    show(save_path)  # type: ignore


# ===================================================================
# 혼동행렬 시각화
# ===================================================================
def confusion_matrix_plot(
    fit,
    title: str | None = None,
    threshold: float = 0.5,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """로지스틱 회귀 적합 결과의 혼동행렬을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        title (str|None): 그래프 제목.
        threshold (float): 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Returns:
        None
    """
    outparams = False
    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    # 학습 데이터 기반 실제값/예측 확률 결정
    y_true = np.asarray(fit.model.endog)
    y_pred_proba = np.asarray(fit.predict(fit.model.exog))
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 혼동행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 혼동행렬 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["음성", "양성"])
    # 가독성을 위해 텍스트 크기/굵기 조정
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        text_kw={"fontsize": 16, "weight": "bold"},
    )

    ax.set_title(f"혼동행렬 (임계값: {threshold})", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore

    show(save_path)  # type: ignore



def silhouette_plot(
    estimator: KMeans | AgglomerativeClustering,
    data: DataFrame,
    title: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """
    군집분석 결과의 실루엣 플롯을 시각화함.

    Args:
        estimator (KMeans | AgglomerativeClustering): 학습된 KMeans 또는 AgglomerativeClustering 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        title (str, optional): 플롯 제목. None이면 자동 생성.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 기준선 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.
        callback (Callable, optional): 추가 커스텀 콜백 함수.
        ax (Axes, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Returns:
        None

    Note:
        - 각 군집별 실루엣 계수 분포를 막대그래프로 시각화
        - 군집 품질(응집도/분리도) 평가에 활용
        - 붉은색 세로선은 전체 평균 실루엣 스코어를 의미
    """

    outparams = False
    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    sil_avg = silhouette_score(X=data, labels=estimator.labels_)
    sil_values = silhouette_samples(X=data, labels=estimator.labels_)

    y_lower = 10

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    n_clusters: int = 0
    if hasattr(estimator, "n_clusters") and estimator.n_clusters is not None:   # type: ignore
        n_clusters = estimator.n_clusters  # type: ignore
    elif hasattr(estimator, "n_clusters_") and estimator.n_clusters_ is not None:    # type: ignore
        n_clusters = estimator.n_clusters_  # type: ignore
    else:
        n_clusters = len(np.unique(estimator.labels_))  # type: ignore

    for i in range(n_clusters):  # type: ignore
        ith_cluster_sil_values = sil_values[estimator.labels_ == i]  # type: ignore
        ith_cluster_sil_values.sort()  # type: ignore

        size_cluster_i = ith_cluster_sil_values.shape[0]  # type: ignore
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(  # type: ignore
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,  # type: ignore
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # type: ignore
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=linewidth)  # type: ignore

    ax.set_xlabel("The silhouette coefficient values", fontsize=config.label_font_size)  # type: ignore
    ax.set_ylabel("Cluster label", fontsize=config.label_font_size)  # type: ignore
    ax.set_xlim([-0.1, 1])  # type: ignore
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # type: ignore

    if title is None:
        title = "Number of Cluster : " + str(n_clusters) + ", Silhouette Score :" + str(round(sil_avg, 3))  # type: ignore

    show(save_path)  # type: ignore


# ===================================================================
# 군집분석 결과 시각화
# ===================================================================
def cluster_plot(
    estimator: KMeans | AgglomerativeClustering | None = None,
    data: DataFrame | None = None,
    xname: str | None = None,
    yname: str | None = None,
    hue: str | None = None,
    vector: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = True,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
) -> None:
    """
    2차원 공간에서 군집분석 결과를 산점도로 시각화함.

    Args:
        estimator (KMeans): 학습된 KMeans 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        xname (str, optional): x축에 사용할 컬럼명. None이면 첫 번째 컬럼 사용.
        yname (str, optional): y축에 사용할 컬럼명. None이면 두 번째 컬럼 사용.
        hue (str, optional): 군집 구분에 사용할 컬럼명. None이면 'cluster' 자동 생성.
        vector (str, optional): 벡터 종류를 의미하는 컬럼명. None이면 사용 안함.
        title (str, optional): 플롯 제목. None이면 기본값 사용.
        palette (str, optional): 색상 팔레트.
        outline (bool, optional): 외곽선 표시 여부.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 중심점 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.
        ax (Axes, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Returns:
        None

    Example:
        ```python
        cluster_plot(estimator, data, xname='Sepal.Length', yname='Sepal.Width')
        ```

    Note:
        - 각 군집별 산점도와 중심점(빨간색 원/숫자) 표시
        - 2차원 특성 공간에서 군집 분포와 분리도 시각화
    """
    outparams = False
    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True

    df = data.copy() if data is not None else None  # type: ignore

    if not hue:
        df["cluster"] = estimator.labels_  # type: ignore
        hue = "cluster"

    if xname is None:
        xname = df.columns[0]  # type: ignore

    if yname is None:
        yname = df.columns[1]  # type: ignore

    xindex = df.columns.get_loc(xname)  # type: ignore
    yindex = df.columns.get_loc(yname)  # type: ignore

    def callback(ax: Axes) -> None:
        ax.set_xlabel("Feature space for the " + xname, fontsize=config.label_font_size)
        ax.set_ylabel("Feature space for the " + yname, fontsize=config.label_font_size)

        if hasattr(estimator, "cluster_centers_"):
            # 클러스터 중심점 표시
            centers = estimator.cluster_centers_  # type: ignore
            ax.scatter(  # type: ignore
                centers[:, xindex],
                centers[:, yindex],
                marker="o",  # type: ignore
                color="white",
                alpha=1,
                s=200,
                edgecolor="r",
                linewidth=linewidth,  # type: ignore
            )

            for i, c in enumerate(centers):
                ax.scatter(
                    c[xindex], c[yindex], marker="$%d$" % i, alpha=1, s=50, edgecolor="k"  # type: ignore
                )

    scatterplot(
        df=df,          # type: ignore
        xname=xname,
        yname=yname,
        hue=hue,
        vector=vector,
        title="The visualization of the clustered data." if title is None else title,
        outline=outline,
        palette=palette,
        width=width,
        height=height,
        linewidth=linewidth,
        save_path=save_path,
        callback=callback,
        ax=ax,
    )


# ===================================================================
# 군집분석 결과의 실루엣 플롯과 군집 산점도를 한 화면에 함께 시각화
# ===================================================================
def visualize_silhouette(
    estimator: KMeans | AgglomerativeClustering,
    data: DataFrame,
    xname: str | None = None,
    yname: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = True,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
) -> None:
    """
    군집분석 결과의 실루엣 플롯과 군집 산점도를 한 화면에 함께 시각화함.

    수업에서 사용한 visualize_silhouette 함수와 동일한 기능을 수행함.

    Args:
        estimator (KMeans | AgglomerativeClustering): 학습된 KMeans 또는 AgglomerativeClustering 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        xname (str, optional): 산점도 x축에 사용할 컬럼명. None이면 첫 번째 컬럼 사용.
        yname (str, optional): 산점도 y축에 사용할 컬럼명. None이면 두 번째 컬럼 사용.
        title (str, optional): 플롯 제목. None이면 기본값 사용.
        palette (str, optional): 색상 팔레트.
        outline (bool, optional): 산점도 외곽선 표시 여부.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 기준선 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.

    Returns:
        None

    Note:
        - 실루엣 플롯(왼쪽)과 2차원 군집 산점도(오른쪽)를 동시에 확인 가능
        - 군집 품질과 분포를 한눈에 비교·분석할 때 유용
    """
    fig, ax = init(rows=1, cols=2, width=width, height=height, title=title)

    silhouette_plot(
        estimator=estimator,
        data=data,
        ax=ax[0],  # type: ignore
        linewidth=linewidth,
        width=width,
        height=height
    )

    cluster_plot(
        estimator=estimator,
        data=data,
        xname=xname,
        yname=yname,
        ax=ax[1],  # type: ignore
        outline=outline,
        palette=palette,
        width=width,
        height=height
    )

    show(save_path)  # type: ignore



# ===================================================================
# 덴드로그램 시각화
# ===================================================================
def dandrogram(
    estimator: AgglomerativeClustering,
    p: int = 30,
    count_sort: Literal["ascending", "descending", False] = "ascending",
    title: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None
) -> None:
    """덴드로그램 시각화

    Args:
        estimator (AgglomerativeClustering): 학습된 AgglomerativeClustering 군집 모델 객체.
        p (int): 덴드로그램에서 표시할 마지막 병합된 군집 수. 기본값 30.
        count_sort (str): 'ascending' 또는 'descending'으로 병합 순서 정렬.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Returns:
        None
    """
    # 덴드로그램을 그리기 위해 linkage 행렬 생성
    counts = np.zeros(estimator.children_.shape[0]) # type: ignore
    n_samples = len(estimator.labels_)

    for i, merge in enumerate(estimator.children_): # type: ignore
        current_count = 0
        for child_idx in merge:  # type: ignore
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

        linkage_matrix = np.column_stack(
            [estimator.children_, estimator.distances_, counts]
        ).astype(float)

    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
        outparams = True


    sch_dendrogram(
        linkage_matrix,
        ax=ax,
        p=p,
        truncate_mode="lastp" if p > 0 else None,
        leaf_rotation=0,
        leaf_font_size=8,
        count_sort=count_sort,
        color_threshold=None,
        above_threshold_color="grey",
    )

    show(save_path)  # type: ignore


# ===================================================================
# PCA 분석 결과에 대한 biplot 시각화
# ===================================================================
def pca_plot(
    estimator: PCA,
    data: DataFrame,
    yname: str | None = None,
    fields: list | tuple | list[list] | tuple[list] | list[tuple] | tuple[tuple] | None = None,
    hue: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """
    PCA 분석 결과에 대한 biplot 시각화

    Args:
        estimator (PCA): 학습된 PCA 객체.
        data (DataFrame): PCA에 사용된 원본 데이터.
        yname (str | None): 종속변수 컬럼명.
        fields (list | tuple | list[list] | tuple[list] | list[tuple] | tuple[tuple] | None): 시각화할 독립변수 목록. None이면 자동 탐지.
        hue (str|None): 집단 구분 컬럼명.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """
    df = data.copy()
    df_columns = df.columns.tolist()

    # 종속변수가 지정되었다면 해당 컬럼 추출
    yfield = None
    if yname is not None and yname in data.columns:
        yfield = df[[yname]].copy()
        df = df.drop(columns=[yname])

    # PCA 변환 수행
    #display(df)
    score = estimator.transform(df)
    #print(score)

    # 추정기로부터 PCA 결과 데이터 프레임 생성
    pca_df = DataFrame(
        data=score,
        columns=[f"PC{i+1}" for i in range(estimator.n_components_)],
    )
    #display(pca_df)

    # 종속변수 컬럼 추가
    if yfield is not None:
        pca_df[yname] = yfield

    # 모든 컬럼명에 대한 조합 생성
    if fields is None:
        feature_cols = pca_df.columns.tolist()
        if yname is not None and yname in feature_cols:
            feature_cols.remove(yname)
        fields = list(combinations(feature_cols, 2))

    if not is_2d(fields):
        fields = [fields]   # type: ignore

    components = estimator.components_

    x_index: int = 0
    y_index: int = 0

    def __callable(ax) -> None:
        for i in range(n):
            ax.arrow(
                0,
                0,
                components[x_index, i],
                components[y_index, i],
                color="r",
                head_width=0.007,
                head_length=0.007,
                linewidth=linewidth * 0.75,
                alpha=0.75,
            )
            ax.text(
                components[x_index, i] * 1.15,
                components[y_index, i] * 1.15,
                f"{df_columns[i]} ({components[x_index, i]:.2f})",
                color="b",
                ha="center",
                va="center",
            )

        if callback is not None:
            callback(ax)

    for field_group in fields:  # type: ignore
        x_index = int(pca_df.columns.get_loc(field_group[0]))   # type: ignore
        y_index = int(pca_df.columns.get_loc(field_group[1]))   # type: ignore

        xs = score[:, x_index]
        ys = score[:, y_index]
        n = score.shape[1]

        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        title = "PCA Biplot"
        if field_group is not None:
            title += " - " + ", ".join(field_group)

        tdf = DataFrame({
            field_group[0]: xs * scalex,
            field_group[1]: ys * scaley,
        })

        scatterplot(
            df=tdf,
            xname=field_group[0],
            yname=field_group[1],
            hue=pca_df[hue] if hue is not None else None,
            outline=False,
            palette=palette,
            width=width,
            height=height,
            linewidth=linewidth,
            save_path=save_path,
            title=title,
            callback=__callable,
        )
