# MCP (Model Context Protocol) 사용 가이드

> 이 문서는 **hossam** 라이브러리의 MCP 기반 서버 인터페이스 사용 방법을 설명합니다.

## 개요

hossam은 기존의 Python 라이브러리로서의 기능을 유지하면서, 동시에 **Model Context Protocol(MCP)** 기반 서버로 작동할 수 있도록 확장되었습니다.

- **라이브러리 모드**: 기존처럼 `from hossam import ...` 로 import 후 사용
- **서버 모드**: `hossam-mcp` 명령으로 JSON 라인 기반 MCP 서버 실행

## 서버 설치 및 실행

### 1. 개발 설치

```bash
python -m pip install -e .
```

`pyproject.toml`의 CLI 스크립트 정의:
```toml
[project.scripts]
hossam-mcp = "hossam.mcp.server:run"
```

### 2. 서버 시작

```bash
hossam-mcp
```

서버는 **표준입출력(stdin/stdout)**을 통해 JSON 라인 프로토콜로 통신합니다.

## 통신 프로토콜

### 요청 형식

```json
{"tool": "도구이름", "args": {"파라미터1": 값1, "파라미터2": 값2}}
```

### 응답 형식

**성공:**
```json
{"ok": true, "result": <결과값>}
```

**실패:**
```json
{"ok": false, "error": "에러 메시지"}
```

### 서버 시작 신호

서버 시작 시 등록된 모든 tool을 나열합니다:
```json
{"ok": true, "server": "hossam", "tools": ["hs_boxplot", "hs_category_summary", ...]}
```

## 코드 스니펫 모드(기본)

기본 동작은 "코드 스니펫 반환"입니다. 실행 결과가 필요하면 `mode: "run"` 또는 아래 실행 플래그를 사용하세요.

실행 강제 플래그:

- `mode: "run"`
- `return: "result"`
- `run: true` / `execute: true` / `result: true`

예시:

```bash
# 딕셔너리 데이터로 missing_values 호출 코드 생성
echo '{"tool":"hs_missing_values","args":{"code":true,"df":[{"a":1,"b":null},{"a":null,"b":2}]}}' | hossam-mcp

# CSV에서 로드하는 예제 코드 생성
echo '{"tool":"hs_outlier_table","args":{"mode":"code","df":"./data.csv"}}' | hossam-mcp
```

코드 모드에서는 DataFrame 자동 변환을 수행하지 않으며, 다음 규칙으로 예제 코드를 생성합니다:

- `df`가 경로(`.csv`/`.xlsx`)면 `pd.read_csv`/`pd.read_excel` 사용
- 그 외는 `pd.DataFrame(<직렬화된 데이터>)`
- 도구에 해당하는 hossam 모듈을 자동 추정하여 `from hossam.<module> import <function>` 형태로 import 라인을 생성

실행 결과가 필요한 경우에는 위의 실행 플래그를 추가하세요(기본은 코드 반환).

## 사용 예시

### 1. 정규분포 난수 생성

```bash
echo '{"tool":"hs_make_normalize_values","args":{"mean":0,"std":1,"size":5}}' | hossam-mcp
```

**응답:**
```json
{"ok": true, "result": [-0.19, 1.12, 0.81, -1.53, 0.45]}
```

### 2. 데이터프레임 결측치 분석

딕셔너리 또는 리스트 형태의 데이터를 전달:

```bash
echo '{"tool":"hs_missing_values","args":{"df":[{"a":1,"b":null},{"a":null,"b":2}],"fields":["a","b"]}}' | hossam-mcp
```

### 3. 표준화(Standardization) 전처리

```bash
echo '{"tool":"hs_standard_scaler","args":{"data":[{"x":1,"y":2},{"x":3,"y":4}]}}' | hossam-mcp
```

## 등록된 Tool 목록

### 📊 통계 분석 (hs_stats)

| Tool | 설명 |
|------|------|
| `hs_missing_values` | 결측치 정보 반환 |
| `hs_outlier_table` | 이상치 경계값 및 사분위수 |
| `hs_category_table` | 범주형 변수의 빈도/비율 |
| `hs_category_summary` | 범주형 변수 분포 요약 |
| `hs_normal_test` | 정규성 검정 (Shapiro/D'Agostino) |

### 🎨 시각화 (hs_plot)

| Tool | 설명 |
|------|------|
| `hs_lineplot` | 선 그래프 |
| `hs_boxplot` | 상자그림 |
| `hs_kdeplot` | KDE(커널 밀도) 그래프 |

**참고:** 시각화 함수는 원격 환경에서 `save_path` 파라미터로 파일 저장 권장:
```json
{"tool":"hs_lineplot","args":{"df":[...],"xname":"x","yname":"y","save_path":"/tmp/plot.png"}}
```

### 🔧 데이터 전처리 (hs_prep)

| Tool | 설명 |
|------|------|
| `hs_standard_scaler` | Z-Score 스케일링 |
| `hs_minmax_scaler` | MinMax 정규화(0~1) |
| `hs_set_category` | 컬럼을 카테고리 타입으로 변환 |
| `hs_get_dummies` | One-Hot 인코딩 |
| `hs_replace_outliner` | 이상치 대체/제거 |

### 🌍 지리정보 (hs_gis)

| Tool | 설명 |
|------|------|
| `hs_geocode` | 주소→위경도 지오코딩(VWorld API) |
| `hs_load_shape` | Shapefile 로드 |
| `hs_save_shape` | GeoDataFrame을 Shapefile/GeoPackage로 저장 |

### ⏰ 시계열 분석 (hs_timeserise)

| Tool | 설명 |
|------|------|
| `hs_diff` | ADF 검정 기반 차분 |
| `hs_rolling` | 단순 이동평균(SMA) |

### 🎓 수업 및 편성 (hs_classroom)

| Tool | 설명 |
|------|------|
| `hs_cluster_students` | 관심사/성적 기반 균형잡힌 조편성 |

### 📚 유틸리티 (hs_util)

| Tool | 설명 |
|------|------|
| `hs_make_normalize_values` | 정규분포 난수 배열 생성 |
| `hs_make_normalize_data` | 정규분포 컬럼 DataFrame 생성 |
| `hs_load_data` | 원격/로컬 데이터 로드 |
| `hs_pretty_table` | DataFrame을 표 문자열로 변환 |

## 데이터 입출력

### DataFrame 입력 형식

**1. 딕셔너리 리스트 (권장)**
```json
{"df": [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]}
```

**2. CSV/Excel 파일 경로**
```json
{"df": "/path/to/data.csv"}
```

**3. 이미 DataFrame인 경우** (Python 내부 호출)
```python
df = pd.DataFrame({"x": [1, 2, 3]})
result = mcp.call("hs_missing_values", df=df)
```

### 응답 DataFrame 형식

```json
{
  "ok": true,
  "result": {
    "index": [0, 1, 2],
    "columns": ["col1", "col2"],
    "data": [[1, "a"], [2, "b"], [3, "c"]]
  }
}
```

## 아키텍처

```
hossam/
├── (기존 공개 API 유지)
├── hs_stats.py
├── hs_plot.py
├── hs_prep.py
├── ...
│
└── mcp/
    ├── __init__.py
    ├── server.py          # MCP 서버 진입점
    ├── hs_stats.py        # 통계 모듈 wrapper
    ├── hs_plot.py         # 시각화 모듈 wrapper
    ├── hs_prep.py         # 전처리 모듈 wrapper
    ├── hs_gis.py          # 지리 모듈 wrapper
    ├── hs_timeserise.py   # 시계열 모듈 wrapper
    ├── hs_classroom.py    # 교실/편성 모듈 wrapper
    └── hs_util.py         # 유틸 모듈 wrapper
```

**특징:**
- 기존 라이브러리 import 시 MCP 서버 자동 실행 **안 함** (명시적 엔트리포인트 필수)
- MCP wrapper는 공개 API만 사용 (얇고 명확한 설계)
- `hs_` prefix 유지로 tool 이름 일관성

## 호환성

### 기존 코드는 그대로 작동

```python
# 기존 방식 - 변경 없음
from hossam.hs_stats import missing_values
from hossam.hs_plot import lineplot

df = pd.DataFrame({"x": [1, 2, None]})
result = missing_values(df)  # 그대로 작동
```

### MCP 서버로도 같은 기능 호출

```bash
echo '{"tool":"hs_missing_values","args":{"df":[{"x":1},{"x":2},{"x":null}]}}' | hossam-mcp
```

## 트러블슈팅

### 서버 시작 실패

```bash
# 설치 확인
python -m pip list | grep hossam

# 재설치
python -m pip install -e . --force-reinstall
```

### 특정 의존성 누락 (예: geopandas)

```bash
# 필요한 패키지 설치
python -m pip install geopandas  # for hs_gis
```

### 데이터 직렬화 실패

- NumPy/Pandas 객체는 자동 변환됨
- 복잡한 타입(클래스 인스턴스 등)은 문자열로 변환

## 다음 단계

1. **CI/CD 통합**: GitHub Actions에서 MCP 서버 테스트 추가
2. **문서 개선**: API 문서에 각 tool의 입출력 예시 추가
3. **성능 최적화**: 대용량 DataFrame 핸들링 개선
4. **확장**: 새로운 모듈 추가 시 `hossam/mcp/<모듈>.py` 추가 후 `server.py`에서 import

---

**Last Updated:** 2026년 1월 14일
**MCP Server Version:** 1.0
