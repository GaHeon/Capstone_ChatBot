import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import aiplatform, storage
from vertexai.generative_models import GenerativeModel
import os
import tempfile
from dotenv import load_dotenv

# .env 파일로부터 환경 변수 로드
load_dotenv()

# GCP 프로젝트/리전 설정
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Cloud Storage에서 벡터 DB 로드
@st.cache_resource
def load_vector_db():
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # Cloud Storage 클라이언트
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # 벡터 DB 파일 다운로드
        blobs = bucket.list_blobs(prefix="embed_faiss/")
        for blob in blobs:
            if not blob.name.endswith('/'):  # 디렉토리 제외
                local_path = os.path.join(temp_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
        
        # FAISS 로드
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
        )
        db = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
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
    prompt = f"""당신은 방사선장비 품질관리(QA)와 유지보수, 안전 점검 항목에 대한 전문 지식을 갖춘 챗봇입니다.

당신의 목적은:

사용자가 제공하는 품질관리 관련 문서 조각과 이전 대화 내용을 바탕으로,
방사선 장비의 점검, 이상징후, 보안, 오류 대응, 유지보수 등의 질문에
기술적으로 정확하고 실무 중심의 응답을 제공하는 것입니다.
**문서에 해당 내용이 없는 경우에도, 당신의 전문 지식을 바탕으로 추론하여 답변해야 합니다.**

다음은 참고할 이전 대화 내용입니다:
{history}

다음은 참고할 문서 내용입니다:
{context}

응답 시 다음의 규칙을 지키세요:

문서에 기반했다는 표현은 하지 마세요.
문서 내용 외에도, 실제 의료 현장에서의 방사선 장비 QA 맥락에 맞추어 추론을 덧붙이세요.
답변은 명확하고, 용어는 전문적이며, 가독성은 높게 구성하세요.
응답은 일반적 설명 → 구체적 적용 사례 → 결론 순으로 구조화하세요.
답변을 간략하게 작성하세요.

사용자 질문: {user_query}
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