import sys
import re

def transform_markdown(md: str) -> str:
    lines = md.splitlines()
    out = []
    out.append('# HossamPy API Reference\n')
    in_code = False
    for line in lines:
        # 코드블록 시작 (들여쓰기 제거)
        if re.match(r'^ {4}```python', line):
            out.append('```python')
            in_code = True
            continue
        # 코드블록 끝 (들여쓰기 제거)
        if in_code and re.match(r'^ {4}```', line):
            out.append('```')
            in_code = False
            continue
        # 코드블록 내부 (들여쓰기 제거)
        if in_code and line.startswith('    '):
            out.append(line[4:])
            continue
        # 4수준 제목을 3수준으로 끌어올림
        if line.startswith('#### '):
            out.append('### ' + line[5:])
        # 1수준 제목을 2수준으로 내림
        elif line.startswith('# '):
            out.append('## ' + line[2:])
        else:
            out.append(line)
    return '\n'.join(out)

if __name__ == '__main__':
    import subprocess
    # pydoc-markdown 실행
    result = subprocess.run(['pydoc-markdown'], capture_output=True, text=True)
    md = result.stdout
    #print(md)
    # 변환 적용
    md_transformed = transform_markdown(md)
    # 파일로 저장
    with open('HOSSAM_API.md', 'w', encoding='utf-8') as f:
        f.write(md_transformed)
