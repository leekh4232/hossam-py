import ast
import os
import pandas as pd
from pathlib import Path
from IPython.display import Markdown, display
from .hs_util import pretty_table


def module_list():
    """
    현재 패키지(hossam) 내의 모든 모듈(파이썬 파일) 이름을 데이터프레임으로 반환함.
    - .py 확장자만 대상, __init__.py는 제외
    - 컬럼명: 'module'
    """
    pkg_dir = os.path.dirname(__file__)
    files = [f for f in os.listdir(pkg_dir) if f.endswith(".py") and f != "__init__.py"]
    modules = [os.path.splitext(f)[0] for f in files]
    df = pd.DataFrame({"module": modules})
    df.sort_values(by='module', inplace=True)
    df.reset_index(drop=True, inplace=True)
    pretty_table(df)


def function_list(module_name: str, keyword: str | None = None):
    """
    지정한 모듈 내의 모든 함수 이름과 간단한 설명을 데이터프레임으로 반환함.

    - module_name: 모듈 이름 (예: 'hs_cluster')
    - keyword: 함수 이름 또는 설명에 포함된 키워드로 필터링 (선택 사항)
    """
    pkg_dir = os.path.dirname(__file__)
    py_path = os.path.join(pkg_dir, module_name + ".py")
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"{py_path} 파일이 존재하지 않음")

    with open(py_path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    rows = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            doc = ast.get_docstring(node) or ''
            # # 파라미터 추출
            # params = []
            # for arg in node.args.args:
            #     if arg.arg != 'self':
            #         params.append(arg.arg)
            # # 리턴 타입
            # returns = ''
            # if node.returns:
            #     try:
            #         returns = ast.unparse(node.returns)
            #     except Exception:
            #         returns = str(node.returns)
            rows.append({
                'function': func_name,
                'description': doc.split('\n')[0] if doc else '',
                # 'parameters': ', '.join(params),
                # 'returns': returns
                'reference': f"https://py.hossam.kr/api/{module_name}/#{module_name}.{func_name}"
            })
    df = pd.DataFrame(rows)
    df.sort_values(by='function', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.style.format(
        {"reference": lambda x: f"<a href='{x}' target='_blank'>view</a>"},
        escape=False    # type: ignore
    )

    if keyword:
        mask = df['function'].str.contains(keyword, case=False) | df['description'].str.contains(keyword, case=False)
        df = df[mask].reset_index(drop=True)

    pretty_table(df)
