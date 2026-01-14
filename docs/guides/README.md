# Hossam 가이드 - 문서 인덱스

> PyPI에서 설치한 Hossam 패키지를 사용하기 위한 공식 가이드입니다.

---

## 🎯 상황별 가이드 선택

### 1️⃣ 빠르게 시작하고 싶다면
👉 **[QUICKSTART.md](./QUICKSTART.md)** (5분)

- `pip install hossam` 후 바로 사용
- 기본 사용 예제
- 자주 하는 질문
- 코딩 스타일 가이드

**대상**: 개별 개발자, Jupyter 노트북 사용자

---

### 2️⃣ VSCode + GitHub Copilot을 사용한다면
👉 **[VSCode_COPILOT.md](./VSCode_COPILOT.md)** (15분)

- VSCode 설정 (`.vscode/settings.json`)
- Copilot Chat 사용 방법
- 자주 사용하는 프롬프트
- Jupyter Notebook 통합
- 문제 해결

**대상**: VSCode 사용자, AI 코드 생성 활용자

---

### 3️⃣ MCP 프로토콜 통신을 직접 구현한다면
👉 **[MCP_SERVER.md](./MCP_SERVER.md)** (30분)

- MCP 서버 시작 및 테스트
- JSON 요청/응답 형식
- 모든 도구의 호출 방식
- 데이터 입력 형식
- 작동 모드 (코드 생성 vs 실행)

**대상**: MCP 클라이언트 개발자, API 통합 담당자

---

## 📚 전체 가이드 목록

| 문서 | 내용 | 대상 |
|------|------|------|
| **QUICKSTART.md** | 설치 후 바로 사용하는 법 | 모두 |
| **VSCode_COPILOT.md** | VSCode + Copilot 설정 및 사용 | VSCode 사용자 |
| **MCP_SERVER.md** | MCP 서버 상세 설명 및 API | 개발자 |

---

## ✨ 주요 특징

### 1. Python 라이브러리
```python
from hossam import *

df = hs_util.load_data("insurance", categories=["sex", "smoker"])
stats = hs_stats.describe(df, "charges")
hs_plot.distribution_plot(df, "charges")
```

### 2. VSCode + GitHub Copilot
```
사용자: "CSV를 로드해서 charges의 분포를 보여줄래?"

Copilot이 자동으로:
✅ 정확한 hossam 함수 사용
✅ categories 파라미터 포함
✅ 모듈명 명시
```

### 3. MCP 서버
```bash
hossam-mcp
```
```json
{
  "tool": "hs_stats_describe",
  "args": {"df": "./data.csv", "fields": ["age"]}
}
```

---

## 🚀 3단계 시작 가이드

### Step 1: 설치
```bash
pip install hossam
```

### Step 2: 가이드 선택
| 선택 | 다음 단계 |
|------|---------|
| **개인 프로젝트** | [QUICKSTART.md](./QUICKSTART.md) 읽기 |
| **VSCode + Copilot 사용** | [VSCode_COPILOT.md](./VSCode_COPILOT.md) 읽기 |
| **MCP 서버 개발** | [MCP_SERVER.md](./MCP_SERVER.md) 읽기 |

### Step 3: 작업 시작
각 가이드의 예제 코드를 복사해서 바로 사용

---

## 📖 문서 구조

```
docs/guides/
├── README.md (이 파일)          ← 진입점
├── QUICKSTART.md                ← 기본 사용
├── VSCode_COPILOT.md            ← IDE 통합
└── MCP_SERVER.md                ← 고급 개발
```

---

## 💡 자주 하는 질문

### Q: 어떤 문서부터 읽어야 할까요?
**A**: 모두 **QUICKSTART.md**부터 시작하세요. 5분 정도면 전체 흐름을 이해할 수 있습니다.

### Q: Jupyter Notebook에서 사용하고 싶어요
**A**: QUICKSTART.md의 "Jupyter Notebook 사용" 섹션 참고

### Q: VSCode에서 GitHub Copilot을 쓰고 싶어요
**A**: VSCode_COPILOT.md의 "⚙️ VSCode 설정" 섹션부터 시작

### Q: MCP 서버를 직접 구현하고 싶어요
**A**: MCP_SERVER.md의 "통신 프로토콜" 섹션부터 시작

---

## 🔗 관련 자료

- **GitHub**: https://github.com/leekh4232/hossam-py
- **PyPI**: https://pypi.org/project/hossam
- **이슈**: https://github.com/leekh4232/hossam-py/issues

---

## 📝 문서 버전

- **마지막 업데이트**: 2026년 1월
- **Hossam 버전**: 1.0+
- **Python**: 3.8+

---

## 🎯 학습 경로

### 초급자 (1일)
1. [QUICKSTART.md](./QUICKSTART.md) 읽기 (5분)
2. 예제 코드 실행해보기 (10분)
3. 자신의 데이터로 분석해보기 (45분)

### 중급자 (2-3일)
1. [VSCode_COPILOT.md](./VSCode_COPILOT.md) 읽기 (15분)
2. VSCode 설정 완료 (5분)
3. Copilot으로 데이터 분석 자동화 (연습)

### 고급자 (1주)
1. [MCP_SERVER.md](./MCP_SERVER.md) 읽기 (30분)
2. MCP 프로토콜 이해 (1시간)
3. 커스텀 클라이언트 구현 (연습)

---

## ✅ 체크리스트

### 설치 완료
- [ ] `pip install hossam` 실행됨
- [ ] 설치 확인: `python -c "import hossam; print(hossam.__version__)"`

### 기본 사용 테스트
- [ ] QUICKSTART.md의 "기본 사용" 예제 실행
- [ ] 자신의 CSV 파일로 데이터 로드 성공
- [ ] 기술통계 함수 호출 성공

### (선택) VSCode + Copilot
- [ ] VSCode 확장 설치
- [ ] `.vscode/settings.json` 생성
- [ ] Copilot Chat에서 데이터 분석 요청 성공

### (선택) MCP 서버
- [ ] MCP 서버 시작: `hossam-mcp`
- [ ] 도구 목록 확인: `hs_mcp_list_tools()`

---

## 🆘 지원

### 문서 오류 또는 개선사항
- GitHub Issues에 보고: https://github.com/leekh4232/hossam-py/issues
- 주제: "[Docs]" 접두어로 시작

### 설치/실행 문제
- GitHub Issues에 보고
- 포함 정보: Python 버전, 설치 방식, 오류 메시지

---

## 다음 단계

**준비가 되었나요?**

👉 지금 바로 **[QUICKSTART.md](./QUICKSTART.md)**를 열어보세요!

```python
# 2분 후 당신은 이렇게 하고 있을 거에요:

from hossam import *

df = hs_util.load_data("insurance", categories=["sex", "smoker", "region"])
stats = hs_stats.describe(df, "charges")
hs_plot.distribution_plot(df, "charges")
```

🚀 **행운을 빕니다!**
