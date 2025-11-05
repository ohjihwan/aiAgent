# 🤖 AI Agent 학습 노트북

> AI Agent, RAG, LangGraph, 벡터 데이터베이스 등 AI 에이전트 구축을 위한 핵심 기술을 학습하는 Jupyter Notebook입니다.

> ⚠️ **참고**: 이 프로젝트는 현재 작성 중입니다. 일부 내용이 추가되거나 수정될 수 있습니다.

---

## 📚 프로젝트 개요

이 저장소는 AI Agent(인공지능 에이전트) 구축에 필요한 핵심 기술들을 단계별로 학습할 수 있는 자료를 포함하고 있습니다. AI Agent의 기본 개념부터 RAG(Retrieval-Augmented Generation), LangGraph를 활용한 워크플로우 구성, 벡터 데이터베이스를 활용한 검색 시스템, SQL 쿼리 자동 생성까지 다양한 실전 기술을 구현하고 실습합니다.

---

## 📘 주요 학습 내용

### 1. AI Agent란?

AI Agent는 환경으로부터 정보를 지각(Perception)하고, 주어진 목표를 달성하기 위해 의사결정(Decision Making)을 거쳐 적절한 행동(Action)을 수행하는 지능형 주체입니다. 단순히 입력에 반응하는 프로그램과 달리, AI Agent는 데이터와 경험을 바탕으로 학습하며 상황에 맞게 적응할 수 있습니다. 이를 위해 텍스트·이미지·음성 등을 동시에 이해하는 멀티모달 모델, 외부 지식을 검색해 활용하는 RAG 구조, 다양한 도구와 시스템과 연결해 실행 능력을 확장하는 MCP 같은 기술이 결합되어 더욱 강력한 에이전트로 발전합니다.

#### AI Agent의 주요 구성 요소

- **지각(Perception)**: 환경으로부터 정보를 수집하고 이해
- **의사결정(Decision Making)**: 목표 달성을 위한 최적의 행동 결정
- **행동(Action)**: 결정된 행동을 실제로 실행

#### AI Agent의 대표적 사례

- **로봇청소기**: 집 안 구조를 센서와 카메라로 인식 → 이동 경로 계획 → 청소 수행
- **자율주행 자동차**: 카메라·라이다 등 센서 데이터로 환경 인식 → 교통 상황 판단 → 가속, 감속, 조향 실행
- **스마트 스피커**: 음성 입력 인식 → 의도 분석 → 음악 재생, 날씨 안내, IoT 제어 실행
- **금융 트레이딩 에이전트**: 시장 데이터 분석 → 투자 전략 수립 → 매수·매도 주문 실행
- **게임 AI**: 게임 상태 인식 → 전략 수립 → 액션 실행

### 2. RAG (Retrieval-Augmented Generation)

RAG(Retrieval-Augmented Generation)는 생성형 AI가 외부 지식을 검색해 활용하는 방식으로, 단순히 모델 파라미터에 저장된 정보만으로 답변하지 않고, 관련 문서를 검색(Retrieval)한 뒤 이를 입력 맥락에 포함시켜 답변을 생성(Generation)합니다. 이 방식은 모델이 최신 정보나 도메인 특화 지식을 활용할 수 있게 해주며, 환각(hallucination)을 줄이고 신뢰도를 높입니다.

#### RAG의 주요 워크플로우

- **기본 RAG 워크플로우**: 질문 입력 → 벡터DB에서 관련 문서 검색 → LLM으로 답변 생성 (직선형 파이프라인)
- **에이전틱 RAG 워크플로우**: 질문 분석 → 필요 시 검색 → 쿼리 재작성 → 반복 검색 → 자기평가 → 재시도 (다단계·반복형 파이프라인)
- **멀티홉 질의**: 단일 질문에 답하기 위해 여러 정보 조각을 순차적으로 연결해 추론

### 3. MCP (Model Context Protocol)

MCP(Model Context Protocol)는 AI 에이전트가 외부 도구, 서비스, 데이터베이스와 표준화된 방식으로 연결되도록 설계된 프로토콜입니다. 기존에는 각 도구와 개별적으로 API를 맞춰야 했다면, MCP는 공통된 인터페이스를 제공해 에이전트가 다양한 리소스를 쉽게 호출하고 응답을 이해할 수 있게 합니다.

### 4. LangGraph

LangGraph는 LangChain 생태계에서 에이전트나 RAG 시스템을 단계별로 구성하고 실행할 수 있게 해주는 그래프 기반 오케스트레이션 프레임워크입니다. 기존 RAG가 직선형 파이프라인이었다면, LangGraph는 노드(작업 단위)와 엣지(흐름)를 그래프 형태로 정의해 분기, 반복, 조건 처리, 에이전트 루프 같은 복잡한 흐름을 명확하게 표현하고 실행할 수 있습니다.

#### LangGraph의 필수 구성요소

- **노드(Node)**: 워크플로우 안에서 실행되는 개별 작업 단위
- **엣지(Edge)**: 노드와 노드를 연결하는 흐름
- **상태(State)**: 워크플로우 실행 중 유지되고 공유되는 데이터 저장소
- **조건 분기와 루프**: 특정 상황에 따라 워크플로우의 진행 경로를 바꾸는 기능

#### 그래프 구조

- **무방향 그래프**: 방향이 없는 그래프, 간선을 통해 노드는 양방향으로 이동 가능
- **방향 그래프**: 간선에 방향이 있는 그래프
- **가중치 그래프**: 간선에 비용 또는 가중치가 할당된 그래프
- **너비 우선 탐색(BFS)**: 같은 레벨의 노드들을 먼저 탐색하는 알고리즘

### 5. 벡터 데이터베이스 (Vector Database)

벡터 데이터베이스는 텍스트, 이미지, 오디오와 같은 데이터를 고차원 벡터 형태로 변환해 저장하고, 이 벡터 간의 유사도를 빠르게 검색할 수 있도록 최적화된 데이터베이스입니다. 일반적인 관계형 데이터베이스가 정확한 값 기반 검색에 적합하다면, 벡터 데이터베이스는 의미적 유사성(semantic similarity)에 기반한 검색을 지원하여 "강아지"와 "개"처럼 다른 표현이라도 비슷한 의미의 데이터를 찾아낼 수 있습니다.

#### 벡터 데이터베이스의 주요 기술

- **ChromaDB**: 대표적인 오픈소스 벡터 데이터베이스
- **청크(Chunk)**: 긴 텍스트나 문서를 작은 단위로 나눈 조각
- **SemanticChunker**: 의미적 맥락을 고려해 텍스트를 분할하는 청크 생성 기법
- **벡터 리트리버**: 사용자 질의를 임베딩 벡터로 변환해 관련 문서를 검색
- **앙상블 리트리버**: BM25 리트리버와 벡터 리트리버를 조합해 더 정확한 검색 결과 제공
- **BM25 리트리버**: 키워드 일치 정도를 계산해 관련 문서를 찾는 전통적인 검색 기법

### 6. RAG 평가 시스템

RAG 시스템의 성능을 평가하기 위한 다양한 평가 방법:

- **문서 관련성 평가**: 검색된 문서가 질문과 관련이 있는지 평가
- **답변 해결성 평가**: 생성된 답변이 질문을 해결하는지 평가
- **환각 평가**: 생성된 답변이 문서에 기반한 사실인지 평가 (팩트 체크)
- **질문 재작성**: 부실한 사용자 쿼리를 더 나은 형태로 재작성

### 7. SQL Agent

SQL Agent는 사용자의 자연어 질문을 SQL 쿼리로 변환하고, 데이터베이스에서 결과를 조회한 후 자연어로 답변을 생성하는 AI 에이전트입니다.

#### SQL Agent의 주요 구성 요소

- **SQLDatabaseToolkit**: SQL 쿼리 실행, 스키마 조회, 테이블 목록 조회 등의 도구 제공
- **쿼리 생성**: 사용자 질문과 데이터베이스 스키마를 기반으로 SQL 쿼리 자동 생성
- **쿼리 검증 및 실행**: 생성된 쿼리를 검증하고 실행하여 결과 조회
- **에러 처리**: 쿼리 실행 오류 시 쿼리 재생성 및 재시도
- **답변 생성**: 조회된 데이터를 기반으로 자연어 답변 생성

### 8. 워크플로우 자동화

- **n8n**: 오픈소스 워크플로우 자동화 도구로, 블록(노드)을 이어 붙여 다양한 서비스와 AI 모델을 연결
- **LangChain Hub**: 프롬프트, 체인, 에이전트 같은 LLM 관련 리소스를 공유하고 재사용할 수 있는 오픈 플랫폼

---

## 📁 프로젝트 구조

```
04. aiAgent/
├── 01. AI Agent.ipynb                      # AI Agent 기초 개념 및 사례
├── 02. 랭그래프.ipynb                       # LangGraph를 활용한 워크플로우 구성
├── 06. 백터 데이터베이스.ipynb              # 벡터 데이터베이스 및 RAG 구현
├── 07. 쿼리문을 작성하는 RAG.ipynb          # SQL Agent 구축
└── README.md                                # 프로젝트 설명서
```

---

## ⚙️ 실행 환경 설정

### 1️⃣ 가상환경 생성 (선택)

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2️⃣ 필수 패키지 설치

```bash
# LangChain 및 LangGraph
pip install langchain langchain-core langchain-community
pip install langgraph langgraph-checkpoint

# OpenAI 통합
pip install langchain-openai

# 벡터 데이터베이스
pip install chromadb langchain-chroma

# 문서 처리
pip install pypdf langchain-experimental

# 검색 도구
pip install langchain-tavily rank-bm25

# 데이터 처리
pip install numpy pandas

# 기타 유틸리티
pip install jupyter notebook
pip install pydantic typing-extensions
```

### 3️⃣ 환경 변수 설정

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# macOS/Linux
export OPENAI_API_KEY="your-api-key-here"
```

### 4️⃣ Jupyter Notebook 실행

```bash
jupyter notebook
```

---

## 📦 주요 패키지

### AI Agent 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **LangGraph**: 그래프 기반 워크플로우 오케스트레이션
- **LangChain Hub**: 프롬프트 및 체인 공유 플랫폼

### 벡터 데이터베이스
- **ChromaDB**: 오픈소스 벡터 데이터베이스
- **LangChain Chroma**: LangChain과 ChromaDB 통합

### 문서 처리
- **PyPDF**: PDF 문서 로딩
- **LangChain Experimental**: 실험적 기능 (SemanticChunker 등)

### 검색 및 리트리버
- **rank-bm25**: BM25 리트리버 구현
- **LangChain Tavily**: 웹 검색 통합

### LLM 통합
- **LangChain OpenAI**: OpenAI 모델 통합

### 데이터베이스
- **SQLite**: 경량 관계형 데이터베이스
- **SQLAlchemy**: SQL 툴킷 및 ORM

---

## 💡 핵심 개념

### AI Agent (AI 에이전트)
환경으로부터 정보를 지각하고, 목표를 달성하기 위해 의사결정을 거쳐 적절한 행동을 수행하는 지능형 주체입니다. 단순한 반응형 프로그램과 달리 학습과 적응 능력을 갖추고 있습니다.

### RAG (Retrieval-Augmented Generation)
외부 지식을 검색해 활용하는 생성형 AI 방식입니다. 관련 문서를 검색한 뒤 이를 입력 맥락에 포함시켜 답변을 생성하여 최신 정보 활용과 환각 감소를 가능하게 합니다.

### LangGraph
그래프 기반 워크플로우 오케스트레이션 프레임워크로, 노드와 엣지를 통해 복잡한 에이전트 흐름을 명확하게 표현하고 실행할 수 있습니다.

### 벡터 데이터베이스 (Vector Database)
데이터를 고차원 벡터로 변환해 저장하고, 벡터 간 유사도를 기반으로 검색할 수 있도록 최적화된 데이터베이스입니다. 의미적 유사성 검색을 지원합니다.

### 청크 (Chunk)
긴 텍스트나 문서를 작은 단위로 나눈 조각으로, 대형 언어모델이 처리할 수 있는 크기로 나누어 임베딩 벡터로 변환합니다.

### 벡터 리트리버 (Vector Retriever)
사용자의 질의를 임베딩 벡터로 변환한 뒤, 벡터 데이터베이스에 저장된 청크 벡터들과의 유사도를 계산하여 가장 관련성 높은 결과를 찾아주는 구성 요소입니다.

### 앙상블 리트리버 (Ensemble Retriever)
여러 종류의 리트리버(BM25, 벡터 리트리버 등)를 조합해 더 정확하고 풍부한 검색 결과를 제공하는 방법입니다.

### 멀티홉 질의 (Multi-hop Query)
단일 질문에 답하기 위해 여러 개의 정보 조각을 순차적으로 연결해 추론해야 하는 질문입니다.

### 상태 (State)
워크플로우 실행 중 유지되고 공유되는 데이터 저장소로, 각 노드가 읽고 쓸 수 있는 맥락 정보를 담고 있습니다.

### 노드 (Node)
LangGraph 워크플로우 안에서 실행되는 개별 작업 단위로, 상태를 입력받아 새로운 값이나 업데이트를 반환하는 함수입니다.

### 엣지 (Edge)
그래프에서 노드 간의 실행 흐름을 연결하는 경로로, 기본 엣지와 조건부 엣지로 구분됩니다.

### 환각 (Hallucination)
LLM이 학습 데이터나 제공된 컨텍스트에 없는 잘못된 정보를 생성하는 현상입니다. RAG는 외부 지식을 활용하여 환각을 줄이는 데 도움을 줍니다.

### SQL Agent
사용자의 자연어 질문을 SQL 쿼리로 변환하고, 데이터베이스에서 결과를 조회한 후 자연어로 답변을 생성하는 AI 에이전트입니다.

---

## 🔗 참고 자료

### AI Agent
- [LangChain 공식 문서](https://python.langchain.com/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [OpenAI Agent SDK](https://platform.openai.com/docs/guides/agents)

### RAG
- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAG 논문 (Lewis et al.)](https://arxiv.org/abs/2005.11401)

### 벡터 데이터베이스
- [ChromaDB 공식 문서](https://www.trychroma.com/)
- [Pinecone 벡터 데이터베이스 가이드](https://www.pinecone.io/learn/vector-database/)

### LangGraph
- [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph 예제](https://github.com/langchain-ai/langgraph)

### SQL Agent
- [LangChain SQL Agent 가이드](https://python.langchain.com/docs/integrations/toolkits/sql_database/)

### 워크플로우 자동화
- [n8n 공식 사이트](https://n8n.io/)
- [LangChain Hub](https://smith.langchain.com/hub)

---

## 📝 라이선스

이 프로젝트는 학습 목적으로 작성되었습니다.

---

## ⚠️ 작성 중 안내

이 프로젝트는 현재 작성 중입니다. 다음 내용들이 추가될 예정입니다:

- 추가 실습 예제 및 프로젝트
- 고급 RAG 기법 (멀티모달 RAG, 하이브리드 검색 등)
- 에이전트 평가 및 최적화 기법
- 프로덕션 배포 가이드

최신 내용을 확인하려면 저장소를 주기적으로 확인해 주세요.
