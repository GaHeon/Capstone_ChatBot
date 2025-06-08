import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import os
from dotenv import load_dotenv

# .env 파일로부터 환경 변수 로드
load_dotenv()

# GCP 프로젝트/리전 설정
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 로컬 경로에서 벡터 DB 로드
@st.cache_resource
def load_vector_db():
    # Docker 이미지에 포함된 경로
    local_path = "embed_faiss"
    
    # FAISS 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
    db = FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)
    return db

# 벡터 DB 로드
db = load_vector_db()

st.title("Radiation QA 챗봇")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
user_query = st.chat_input("질문을 입력하세요:")

if user_query:
    # 사용자 질문 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # FAISS 벡터 검색
    docs = db.similarity_search_with_score(user_query, k=5)
    context = "\n\n".join(doc.page_content for doc, _ in docs)
    
    # 이전 대화 기록을 프롬프트에 통합
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    # 프롬프트 구성 (대화 기록 포함)
    prompt = f"""당신은 방사선 장비의 품질관리(QA), 유지보수, 안전 점검에 대한 전문 지식을 갖춘 챗봇입니다.

당신의 역할은:
사용자가 제공한 문서 내용과 이전 대화 흐름을 참고하여,  
방사선 장비의 작동, 오류 대응, 유지보수, 안전관리 등에 대해  
실무적이고 정확하며 자연스러운 방식으로 답변하는 것입니다.

다음은 이전 대화 내용입니다:
{history}

다음은 참고 가능한 문서 정보입니다:
{context}

사용자 질문:
{user_query}

응답 지침:
- 문서에 "나와 있지 않다", "직접적으로 나타나진 않는다", "제공된 문서에는" 등의 표현은 절대 사용하지 마세요.
- 문서에 명확한 내용이 없더라도, 전문가의 입장에서 자연스럽게 지식을 바탕으로 설명하세요.
- 설명은 너무 딱딱하지 않게, 하지만 명확하고 실무적으로 서술형으로 작성하세요.
- 기술 용어는 필요한 경우 명확히 설명하고, 문맥에 맞는 예시를 덧붙이세요.
- 답변 길이는 질문의 난이도에 따라 유연하게 조절하세요.

답변:
"""

    # Vertex AI Gemini API 호출 (스트리밍)
    with st.chat_message("assistant"):
        # Vertex AI 모델 인스턴스 직접 생성
        model = GenerativeModel("gemini-2.0-flash")

        # 스트리밍으로 콘텐츠 생성 요청
        responses = model.generate_content(
            contents=[prompt],
            stream=True
        )

        # st.write_stream을 사용하여 응답을 실시간으로 표시하고, 전체 답변을 반환받음
        answer = st.write_stream(response.text for response in responses)

    # 전체 응답을 대화 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})