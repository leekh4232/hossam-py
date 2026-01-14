# VSCode + GitHub Copilot 가이드

> PyPI에서 설치한 Hossam을 VSCode의 GitHub Copilot과 함께 사용하는 방법입니다.

---

## 📋 필수 요구사항

### 1. 설치

```bash
pip install hossam
```

### 2. VSCode 확장

- **GitHub Copilot** 또는 **GitHub Copilot Chat** (필수)
  - VSCode 확장 마켓플레이스에서 설치
  - GitHub 계정으로 로그인

### 3. VSCode 설정

프로젝트 루트에 `.vscode/settings.json` 생성

---

## ⚙️ VSCode 설정

### 최소 설정

`.vscode/settings.json`:

```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    "You are a data analysis expert using the Hossam library.",
    "Key rules:",
    "1. Always use categories parameter: hs_util.load_data('data', categories=['col1'])",
    "2. Use module names: hs_stats.*, hs_plot.*, hs_prep.*",
    "3. Never use astype('category') manually"
  ]
}
```

### 권장 설정 (MCP 서버 포함)

```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    "You are a data analysis expert using the Hossam library.",
    "",
    "## 절대 규칙",
    "1. Data loading: hs_util.load_data('key', categories=['col1', 'col2'])",
    "2. Module names: hs_stats.*, hs_plot.*, hs_prep.*",
    "3. Never use astype('category')",
    "",
    "## Common tools",
    "hs_stats: missing_values, describe, outlier_table, normal_test",
    "hs_plot: distribution_plot, countplot, scatterplot, heatmap",
    "hs_prep: standard_scaler, minmax_scaler, get_dummies",
    "hs_util: load_data, load_info"
  ],
  "mcp.servers": {
    "hossam": {
      "command": "hossam-mcp"
    }
  }
}
```

---

## 💡 Copilot Chat 사용

### 방법 1: 챗 창에서 직접 입력

1. `Cmd + L` (또는 Copilot Chat 아이콘 클릭)
2. 질문 입력

### 방법 2: 코드 선택 후 질문

1. 코드 범위 선택
2. 마우스 우클릭 → "Copilot에 물어보기"
3. 질문 입력

### 방법 3: 인라인 개선 요청

1. 코드 작성
2. 주석 입력: `// TODO: improve with hossam`
3. Copilot 답변 대기

---

## 📝 자주 사용하는 질문

### 기본 데이터 분석

```
csv 파일을 로드해서 charges 컬럼의 분포를 분석해줄래?
```

**Copilot 응답 (코드 예시):**
```python
from hossam import *

df = hs_util.load_data("data.csv", categories=[...])
stats = hs_stats.describe(df, "charges")
hs_plot.distribution_plot(df, "charges")
```

### 결측치 분석

```
이 DataFrame의 결측치를 분석해서 시각화까지 해줄래?
```

**자동 응답:**
```python
from hossam import *

missing = hs_stats.missing_values(df)
# 결측치가 있으면 처리
df_clean = hs_prep.fillna_method(df)
```

### 전처리

```
이 데이터의 수치형 변수를 표준화하고, 범주형 변수를 인코딩해줄래?
```

**자동 응답:**
```python
from hossam import *

# 표준화
scaled = hs_prep.standard_scaler(df, features=["age", "bmi"])

# One-Hot 인코딩
encoded = hs_prep.get_dummies(df, fields=["sex", "smoker"])
```

### 시각화

```
bmi와 charges의 관계를 smoker 별로 산점도로 보여줄래?
```

**자동 응답:**
```python
from hossam import *

hs_plot.scatterplot(df, xname="bmi", yname="charges", hue="smoker")
```

### 통계 검정

```
charges가 정규분포를 따르는지 검정해줄래?
```

**자동 응답:**
```python
from hossam import *

result = hs_stats.normal_test(df, "charges")
print(result)
```

---

## 🎯 Jupyter Notebook + Copilot

### 셀별 작업 흐름

**셀 1: 라이브러리 및 데이터 로드**
```python
from hossam import *

df = hs_util.load_data("insurance", categories=["sex", "smoker", "region"])
df.head()
```

**셀 2-5: Copilot이 생성한 코드 실행**
```
각 셀의 분석 작업을 Copilot에 요청하면
Copilot이 정확한 hossam 함수로 코드 생성
```

**셀 N: 분석 결과 정리 (마크다운)**
```markdown
## 📊 분석 결과

### 주요 통계
- 평균: $13,270
- 중앙값: $9,382
- 표준편차: $12,110

### 분포 특징
- 우측 꼬리 분포 (왜도: 1.52)
- 이상치: 139개 (10.4%)

### 권장사항
- 로그 변환 고려
- 범주형 변수별 분석 필요
```

---

## 🔧 고급 설정

### Copilot Chat 언어 설정

한국어로 설정:
```json
{
  "github.copilot.chat.localeOverride": "ko"
}
```

영어로 설정:
```json
{
  "github.copilot.chat.localeOverride": "en"
}
```

### 코드 생성 옵션

필요한 코드 유형을 명시:

```
코드만 보여줄래? (실행 안 함)
```

또는

```
실행해서 결과도 보여줄래?
```

### MCP 서버 포트 지정 (고급)

```json
{
  "mcp.servers": {
    "hossam": {
      "command": "hossam-mcp",
      "env": {
        "MCP_PORT": "9000"
      }
    }
  }
}
```

---

## ✅ 검증 체크리스트

### 설정 확인

- [ ] `pip install hossam` 완료
- [ ] GitHub Copilot Chat 설치됨
- [ ] `.vscode/settings.json` 파일 생성됨
- [ ] VSCode 재시작됨

### 기능 확인

- [ ] Copilot Chat 열림 (`Cmd + L`)
- [ ] "CSV 파일을 로드해줄래?" 질문 시 정확한 코드 생성
- [ ] 생성된 코드에 `categories` 파라미터 포함
- [ ] 모듈명 명시 (hs_util., hs_stats. 등)

### 문제 해결

```
Q: Copilot이 categories 파라미터를 놓침
A: settings.json의 instructions에 "categories parameter"를 강조로 추가
```

```
Q: "hossam-mcp" 명령을 찾을 수 없음
A: pip install이 올바른지 확인, python -m hossam.mcp.server 사용
```

```
Q: VSCode가 MCP 서버를 인식하지 못함
A: .vscode/settings.json 경로 확인, VSCode 재시작
```

---

## 🐛 알려진 문제 및 해결책

### 1. "categories 파라미터" 자동 생성 안 됨

**증상**: Copilot이 `load_data("insurance")` 만 생성

**해결**:
- settings.json의 instructions에 명시적으로 추가
- "Always use categories parameter" 문구 강조
- 처음 작업에서 예시 코드 제공

```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    "IMPORTANT: Always include categories parameter in load_data()",
    "Example: hs_util.load_data('insurance', categories=['sex', 'smoker'])"
  ]
}
```

### 2. Copilot이 개별 import 사용

**증상**: `from hossam.hs_stats import describe` 생성

**해결**:
```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    "Always use: from hossam import *",
    "Then call functions with module prefix: hs_stats.describe(), hs_plot.boxplot()"
  ]
}
```

### 3. 시각화 코드가 실행되지 않음

**증상**: Jupyter에서 그래프가 안 나타남

**해결**:
```python
# Jupyter 셀 앞에 추가
%matplotlib inline

# 또는
%matplotlib widget

# 그 후 Copilot 사용
```

---

## 📚 자료

- **API 가이드**: [QUICKSTART.md](./QUICKSTART.md)
- **MCP 상세**: [MCP_SERVER.md](./MCP_SERVER.md)
- **GitHub**: https://github.com/leekh4232/hossam-py

---

## 💬 팁 & 트릭

### 1. 반복 작업 자동화

첫 질문에 충분한 컨텍스트 제공:

```
나는 insurance.csv로 다음을 하고 싶어:
1. 데이터 로드 (sex, smoker, region은 범주형)
2. charges의 분포 분석
3. smoker별로 charges 비교
4. 이상치 제거 후 전처리

전체 코드를 보여줄래?
```

### 2. 코드 개선 요청

생성된 코드 선택 후:

```
이 코드를 더 효율적으로 만들어줄래?
```

### 3. 오류 해결

오류 메시지 + 코드 함께 제공:

```
이 코드를 실행하면 "KeyError" 오류가 나요. 고쳐줄래?

[코드 붙여넣기]
[오류 메시지 붙여넣기]
```

---

## 🎓 학습 경로

### Day 1: 기본 설정
- Hossam 설치
- VSCode 확장 설정
- `.vscode/settings.json` 작성

### Day 2: Copilot 기본 사용
- Copilot Chat 열기
- 간단한 데이터 분석 요청
- 생성된 코드 이해

### Day 3: 심화 활용
- 복잡한 분석 요청
- 시각화 함께 요청
- 통계 검정 활용

### Day 4: Jupyter 통합
- Jupyter Notebook에서 Copilot 사용
- 셀별 분석 자동화
- 결과 정리

---

## 완료! 🎉

이제 Hossam + Copilot으로 효율적인 데이터 분석을 시작할 수 있습니다!
