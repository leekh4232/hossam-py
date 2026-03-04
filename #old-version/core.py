# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import inspect
import sys, os
import numpy as np

# -------------------------------------------------------------
from pycallgraphix.wrapper import register_method

# -------------------------------------------------------------
from tabulate import tabulate

# -------------------------------------------------------------
from pandas import DataFrame, Series

# -------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

# -------------------------------------------------------------
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# -------------------------------------------------------------
__RANDOM_STATE__ = 0

__MAX_ITER__ = 1000

__N_JOBS__ = -1

__EARLY_STOPPING_ROUNDS__ = 10

__LINEAR_REGRESSION_HYPER_PARAMS__ = {"fit_intercept": [True, False]}

__RIDGE_HYPER_PARAMS__ = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
}

__LASSO_HYPER_PARAMS__ = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "selection": ["cyclic", "random"],
}

__KNN_REGRESSION_HYPER_PARAMS__ = {
    "n_neighbors": np.arange(2, stop=6),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

__DTREE_REGRESSION_HYPER_PARAMS__ = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    # "splitter": ["best", "random"],
    # "min_samples_split": np.arange(2, stop=10),
    # "min_samples_leaf": np.arange(1, stop=10),
    # "max_features": ["auto", "sqrt", "log2"]
}

__SVR_HYPER_PARAMS__ = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    # "C": [0.001, 0.01, 0.1, 1, 10, 100],
    # "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5],
    # "gamma": ["scale", "auto"],
}

__SGD_REGRESSION_HYPER_PARAMS__ = {
    "loss": [
        "squared_error",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ],
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.001, 0.01, 0.1],
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
}


__LOGISTIC_REGRESSION_HYPER_PARAMS__ = {
    # "C": [0.001, 0.01, 0.1, 1, 10, 100],
}

__KNN_CLASSFICATION_HYPER_PARAMS__ = {
    "n_neighbors": np.arange(2, stop=6),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

__NB_HYPER_PARAMS__ = {
    # "priors" : None,
    # "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

__DTREE_CLASSIFICATION_HYPER_PARAMS__ = {
    "criterion": ["gini", "entropy"],
    # "max_depth": np.arange(1, stop=10),
    # "min_samples_split": np.arange(2, stop=10),
    # "min_samples_leaf": np.arange(1, stop=10),
    # "max_features": ["auto", "sqrt", "log2"],
}

__LINEAR_SVC_HYPER_PARAMS__ = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
}

__SVC_HYPER_PARAMS__ = {
    # "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "kernel": ["poly", "rbf", "sigmoid"],
    # "degree": np.arange(2, stop=6),
    # "gamma": ["scale", "auto"],
}

__SGD_CLASSFICATION_HYPER_PARAMS__ = {
    # "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
    # "penalty": ["l2", "l1", "elasticnet"],
    "penalty": ["l2", "l1"],
    # "alpha": [0.001, 0.01, 0.1],
    # "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
}

__BAGGING_HYPER_PARAMS__ = {
    "bootstrap_features": [False, True],
    "n_estimators": [10, 20, 50, 100],
    "max_features": [0.5, 0.7, 1.0],
    "max_samples": [0.5, 0.7, 1.0],
}

__RANDOM_FOREST_REGRESSION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_features": ["sqrt", "log2"],
    "max_depth": [10, 20, 50, None],
}

__RANDOM_FOREST_CLASSIFICATION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "max_depth": [10, 20, 50, None],
}

__ADA_BOOSTING_CLASSIFICATION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "learning_rate": [0.001, 0.01, 0.1, 1],
}

__ADA_BOOSTING_REGRESSION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "learning_rate": [0.001, 0.01, 0.1, 1],
}

__GRADIENT_BOOSTING_CLASSIFICATION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "subsample": [0.5, 0.7, 1.0],
}

__GRADIENT_BOOSTING_REGRESSION_HYPER_PARAMS__ = {
    "n_estimators": [10, 20, 50, 100],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "subsample": [0.5, 0.7, 1.0],
}

__XGBOOST_CLASSIFICATION_HYPER_PARAMS__ = {
    "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
    "n_estimators": [100, 200, 300, 400, 500],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0, 1, 2, 3, 4, 5],
    "max_depth": [0, 2, 4, 6],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [1, 3, 5, 7, 9],
    "reg_lambda": [1, 3, 5, 7, 9],
}

__XGBOOST_REGRESSION_HYPER_PARAMS__ = {
    "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
    "n_estimators": [100, 200, 300, 400, 500],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0, 1, 2, 3, 4, 5],
    "max_depth": [0, 2, 4, 6],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [1, 3, 5, 7, 9],
    "reg_lambda": [1, 3, 5, 7, 9],
}

__LIGHTGBM_CLASSIFICATION_HYPER_PARAMS__ = {
    "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
    "n_estimators": [100, 200, 300, 400, 500],
    "min_child_weight": [1, 3, 5, 7, 9],
    "max_depth": [0, 2, 4, 6],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "lambda_l1": [0, 1, 3, 5, 7],
    "lambda_l2": [0, 1, 3, 5, 7],
}

__LIGHTGBM_REGRESSION_HYPER_PARAMS__ = {
    "learning_rate": [0.1, 0.3, 0.5, 0.7, 1],
    "n_estimators": [100, 200, 300, 400, 500],
    "min_child_weight": [1, 3, 5, 7, 9],
    "max_depth": [0, 2, 4, 6],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "lambda_l1": [0, 1, 3, 5, 7],
    "lambda_l2": [0, 1, 3, 5, 7],
}


# -------------------------------------------------------------

def get_estimator(
    classname: any,
    est: any = None,
    objective: str = None,
    eval_metric: str = None,
    **params,
) -> any:
    """분류분석 추정기 객체를 생성한다. 고정적으로 사용되는 속성들을 일괄 설정한다.

    Args:
        classname (any): 분류분석 추정기 클래스
        est (list | any, optional): Voting, Bagging에서 사용될 추정기 객체. Defaults to None.

    Returns:
        any: _description_
    """

    args = {}

    # VottingClassifier, VotingRegressor
    if "estimators" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["estimators"] = est

    # BaggingClassifier, BaggingRegressor
    if "estimator" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["estimator"] = est

    if "early_stopping_rounds" in dict(
        inspect.signature(obj=classname.__init__).parameters
    ):
        args["early_stopping_rounds"] = __EARLY_STOPPING_ROUNDS__

    if "eval_metric" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["eval_metric"] = eval_metric

    # 공통 속성들
    if "n_jobs" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["n_jobs"] = __N_JOBS__

    # if "max_iter" in dict(inspect.signature(obj=classname.__init__).parameters):
    #     args["max_iter"] = __MAX_ITER__

    if "random_state" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["random_state"] = __RANDOM_STATE__

    if "early_stopping" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["early_stopping"] = True

    if "probability" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["probability"] = True

    if "verbose" in dict(inspect.signature(obj=classname.__init__).parameters):
        args["verbose"] = False
        #args["verbose"] = True

    if classname == AdaBoostClassifier:
        args["algorithm"] = "SAMME"

    if classname == XGBClassifier or classname == XGBRegressor:
        args["device"] = "cpu"
        args["verbosity"] = 0
        args["objective"] = objective

    # if classname == LGBMClassifier or classname == LGBMRegressor:
    #     args["device"] = "cpu"
    #     #args["verbose"] = -1
    #     args["verbose"] = True

    if params:
        args.update(params)

    print(args)

    return classname(**args)


# -------------------------------------------------------------

def __ml(
    classname: any,
    x_train: DataFrame,
    y_train: Series = None,
    x_test: DataFrame = None,
    y_test: Series = None,
    cv: int = 5,
    scoring: any = None,
    est: any = None,
    is_print: bool = True,
    **params,
) -> any:
    """머신러닝 분석을 수행하고 결과를 출력한다.

    Args:
        classname (any): 분류분석 추정기 (모델 객체)
        x_train (DataFrame): 독립변수에 대한 훈련 데이터
        y_train (Series): 종속변수에 대한 훈련 데이터
        x_test (DataFrame): 독립변수에 대한 검증 데이터. Defaults to None.
        y_test (Series): 종속변수에 대한 검증 데이터. Defaults to None.
        cv (int, optional): 교차검증 횟수. Defaults to 5.
        scoring (any, optional): 교차검증 시 사용할 평가지표. Defaults to None.
        est (list | any, optional): Voting, Bagging에서 사용될 추정기 리스트. Defaults to None.
        is_print (bool, optional): 출력 여부. Defaults to True.

    Returns:
        any: 모델객체
    """
    # if cv < 2:
    #     cv = 2

    # 교차검증 설정
    if not params:
        params = {}

    objective = None
    eval_metric = None

    if classname in [XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor]:
        if classname == XGBClassifier or classname == LGBMClassifier:
            classes = y_train.unique()
            n_classes = len(classes)

            if n_classes == 2:
                objective = "binary:logistic"
                eval_metric = "error"
            else:
                objective = "multi:softmax"
                eval_metric = "merror"
        else:
            objective = "reg:squarederror"
            eval_metric = "rmse"

    prototype_estimator = get_estimator(
        classname=classname, est=est, objective=objective, eval_metric=eval_metric
    )

    if cv > 0:
        if scoring is None:
            grid = RandomizedSearchCV(
                estimator=prototype_estimator,
                param_distributions=params,
                cv=cv,
                n_jobs=__N_JOBS__,
                # n_iter=__MAX_ITER__,
                random_state=__RANDOM_STATE__,
                verbose=0,
            )
            # grid = GridSearchCV(
            #     estimator=prototype_estimator,
            #     param_grid=params,
            #     cv=cv,
            #     n_jobs=__N_JOBS__,
            # )
        else:
            grid = RandomizedSearchCV(
                estimator=prototype_estimator,
                param_distributions=params,
                cv=cv,
                n_jobs=__N_JOBS__,
                # n_iter=__MAX_ITER__,
                random_state=__RANDOM_STATE__,
                scoring=scoring,
                verbose=0,
            )
            # grid = GridSearchCV(
            #     estimator=prototype_estimator,
            #     param_grid=params,
            #     cv=cv,
            #     n_jobs=__N_JOBS__,
            #     scoring=scoring,
            # )

        try:
            if classname in [
                XGBClassifier,
                XGBRegressor,
            ]:
                grid.fit(
                    X=x_train,
                    y=y_train,
                    eval_set=[(x_train, y_train), (x_test, y_test)],
                    verbose=False,
                )
            elif classname in [
                LGBMClassifier,
                LGBMRegressor,
            ]:
                grid.fit(
                    X=x_train,
                    y=y_train,
                    eval_metric=eval_metric,
                    eval_set=[(x_train, y_train), (x_test, y_test)],
                )
            else:
                grid.fit(X=x_train, y=y_train)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(
                f"\033[91m[{fname}:{exc_tb.tb_lineno}] {str(object=exc_type)} {exc_obj}\033[0m"
            )
            return None

        result_df = DataFrame(data=grid.cv_results_["params"])

        if "mean_test_score" in grid.cv_results_:
            result_df["mean_test_score"] = grid.cv_results_["mean_test_score"]
            result_df = result_df.dropna(subset=["mean_test_score"])
            result_df = result_df.sort_values(by="mean_test_score", ascending=False)

        estimator = grid.best_estimator_
        estimator.best_params = grid.best_params_

        if is_print:
            print("[교차검증 TOP5]")
            print(
                tabulate(
                    tabular_data=result_df.head().reset_index(drop=True),
                    headers="keys",
                    #tablefmt="psql",
                    tablefmt="github",
                    showindex=True,
                    numalign="right",
                )
            )
            print("")

            print("[Best Params]")
            print(grid.best_params_)
            print("")
    else:
        print("교차검증 안함~!!!")
        estimator = prototype_estimator
        estimator.fit(X=x_train, y=y_train)

    # ------------------------------------------------------
    # 결과값 생성

    # 훈련 데이터에 대한 추정치 생성
    y_pred = (
        estimator.predict(x_test) if x_test is not None else estimator.predict(x_train)
    )

    if hasattr(estimator, "predict_proba"):
        y_pred_prob = (
            estimator.predict_proba(x_test)
            if x_test is not None
            else estimator.predict_proba(x_train)
        )

    # 도출된 결과를 모델 객체에 포함시킴
    estimator.x = x_test if x_test is not None else x_train
    estimator.y = y_test if y_test is not None else y_train
    estimator.y_pred = y_pred if y_test is not None else estimator.predict(x_train)

    estimator.train_score = (
        estimator.score(x_train, y_train)
        if x_train is not None and y_train is not None
        else None
    )
    estimator.test_score = (
        estimator.score(x_test, y_test)
        if x_test is not None and y_test is not None
        else None
    )

    if y_test is not None or y_train is not None:
        estimator.resid = (
            y_test - y_pred
            if y_test is not None
            else y_train - estimator.predict(x_train)
        )

    if hasattr(estimator, "predict_proba"):
        estimator.y_pred_proba = (
            y_pred_prob if y_test is not None else estimator.predict_proba(x_train)
        )

    return estimator


# -------------------------------------------------------------

def get_random_state() -> int:
    """랜덤 시드를 반환한다.

    Returns:
        int: 랜덤 시드
    """
    return __RANDOM_STATE__


# -------------------------------------------------------------

def get_max_iter() -> int:
    """최대 반복 횟수를 반환한다.

    Returns:
        int: 최대 반복 횟수
    """
    return __MAX_ITER__


# -------------------------------------------------------------

def get_n_jobs() -> int:
    """병렬 처리 개수를 반환한다.

    Returns:
        int: 병렬 처리 개수
    """
    return __N_JOBS__


# -------------------------------------------------------------

def get_hyper_params(classname: any, key: str = None) -> dict:
    """분류분석 추정기의 하이퍼파라미터를 반환한다.

    Args:
        classname (any): 분류분석 추정기

    Returns:
        dict: 하이퍼파라미터
    """

    params = {}

    if classname == LinearRegression:
        params = __LINEAR_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == Ridge:
        params = __RIDGE_HYPER_PARAMS__.copy()
    elif classname == Lasso:
        params = __LASSO_HYPER_PARAMS__.copy()
    elif classname == KNeighborsRegressor:
        params = __KNN_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == SVR:
        params = __SVR_HYPER_PARAMS__.copy()
    elif classname == DecisionTreeRegressor:
        params = __DTREE_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == SGDRegressor:
        params = __SGD_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == LogisticRegression:
        params = __LOGISTIC_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == KNeighborsClassifier:
        params = __KNN_CLASSFICATION_HYPER_PARAMS__.copy()
    elif classname == GaussianNB:
        params = __NB_HYPER_PARAMS__.copy()
    elif classname == DecisionTreeClassifier:
        params = __DTREE_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == LinearSVC:
        params = __LINEAR_SVC_HYPER_PARAMS__.copy()
    elif classname == SVC:
        params = __SVC_HYPER_PARAMS__.copy()
    elif classname == SGDClassifier:
        params = __SGD_CLASSFICATION_HYPER_PARAMS__.copy()
    elif classname == BaggingRegressor or classname == BaggingClassifier:
        params = __BAGGING_HYPER_PARAMS__.copy()
    elif classname == RandomForestRegressor:
        params = __RANDOM_FOREST_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == RandomForestClassifier:
        params = __RANDOM_FOREST_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == AdaBoostRegressor:
        params = __ADA_BOOSTING_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == AdaBoostClassifier:
        params = __ADA_BOOSTING_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == GradientBoostingRegressor:
        params = __GRADIENT_BOOSTING_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == GradientBoostingClassifier:
        params = __GRADIENT_BOOSTING_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == XGBClassifier:
        params = __XGBOOST_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == XGBRegressor:
        params = __XGBOOST_REGRESSION_HYPER_PARAMS__.copy()
    elif classname == LGBMClassifier:
        params = __LIGHTGBM_CLASSIFICATION_HYPER_PARAMS__.copy()
    elif classname == LGBMRegressor:
        params = __LIGHTGBM_REGRESSION_HYPER_PARAMS__.copy()

    if params:
        key_list = list(params.keys())

        if params and key is not None:
            for p in key_list:
                params[f"{key}__{p}"] = params[p]
                del params[p]

    # print(f"[{classname}] {params}")
    return params
