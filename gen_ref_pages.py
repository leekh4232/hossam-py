"""빌드 시 hossam 패키지 소스에서 API 레퍼런스 md 페이지와 nav를 자동 생성한다.

docs/api/*.md 를 수동으로 관리할 필요가 없다.
소스에 최상위 모듈이 추가/삭제/변경되면 문서 페이지와 nav도 자동으로
생성/삭제/갱신되므로, 소스와 문서가 항상 동기화된다.

- `hossam/` 최상위의 공개 모듈(`_`로 시작하지 않는 *.py)마다 페이지를 만든다.
- `legacy/` 등 하위 폴더는 문서화 대상에서 제외한다.
"""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "hossam"
src_dir = Path(PACKAGE)

nav = mkdocs_gen_files.Nav()

# 패키지 개요 페이지
# nav 링크는 api/SUMMARY.md 기준 상대경로이므로 파일명만 사용한다.
with mkdocs_gen_files.open(Path("api", "hossam.md"), "w") as fd:
    fd.write(f"---\ntitle: {PACKAGE} Package\n---\n\n")
    fd.write(f"# {PACKAGE} 패키지\n\n::: {PACKAGE}\n")
nav["Package"] = "hossam.md"

# 최상위 모듈별 페이지 (하위 폴더/언더스코어 모듈 제외)
for path in sorted(src_dir.glob("*.py")):
    module = path.stem
    if module.startswith("_"):
        continue

    identifier = f"{PACKAGE}.{module}"

    with mkdocs_gen_files.open(Path("api", f"{module}.md"), "w") as fd:
        fd.write(f"---\ntitle: {identifier}\n---\n\n")
        fd.write(f"# {identifier}\n\n::: {identifier}\n")

    # 문서의 'edit' 링크가 실제 소스 파일을 가리키도록 설정
    mkdocs_gen_files.set_edit_path(Path("api", f"{module}.md"), path)
    nav[module] = f"{module}.md"

# literate-nav 가 읽을 nav 파일 생성
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
