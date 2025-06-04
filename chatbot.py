import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import torch
import gc
import re
import traceback

# ✅ Streamlit 설정
st.set_page_config(page_title="LoRA QA Chatbot", page_icon="🤖", layout="wide")
st.title("🧠 LoRA 기반 Radiation QA Chatbot")

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "db" not in st.session_state:
    st.session_state.db = None
if "device" not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

# 📦 벡터 DB 로딩
try:
    if st.session_state.db is None:
        with st.spinner("🔄 벡터 데이터베이스 로딩 중..."):
            if not os.path.exists("embed_faiss"):
                st.error("❌ 임베딩 파일이 없습니다. 먼저 create_embeddings.py를 실행해주세요.")
                st.stop()
                
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
                model_kwargs={'device': st.session_state.device}
            )
            db = FAISS.load_local("embed_faiss", embeddings, allow_dangerous_deserialization=True)
            st.session_state.db = db
except Exception as e:
    st.error(f"❌ 벡터 DB 로딩 실패: {str(e)}")
    st.error("상세 오류 정보:")
    st.error(traceback.format_exc())
    st.stop()

# ✅ 사용자 질문 입력
user_query = st.chat_input("질문을 입력하세요:")

if user_query:
    try:
        # 🔍 문서 검색
        docs = st.session_state.db.similarity_search_with_score(user_query, k=5)

        # 거리 → 유사도 변환 함수
        def distance_to_similarity(distance):
            return 1 / (1 + distance)

        # 정렬 및 필터링 (유사도 0.3 이상만 사용)
        filtered_docs = []
        for doc, score in docs:
            sim = distance_to_similarity(score)
            if sim > 0.3:
                filtered_docs.append((doc, sim))

        if not filtered_docs:
            filtered_docs = [(doc, distance_to_similarity(score)) for doc, score in docs[:3]]

        # 유사도 기준 내림차순 정렬
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        # 👀 검색된 청크 시각화
        with st.expander("🔍 검색된 문서 청크 보기", expanded=True):
            st.markdown("### 📚 검색 결과")
            for i, (doc, sim) in enumerate(filtered_docs, 1):
                source = doc.metadata.get("source", "알 수 없음")
                chunk_num = doc.metadata.get("chunk", "알 수 없음")
                st.markdown(f"""
                #### 청크 {i} (유사도: {sim:.2f})
                **출처:** {source} / 청크 번호: {chunk_num}
                ```text
                {doc.page_content}
                ```
                ---
                """)

        # 🧠 프롬프트 구성
        context = "\n\n".join(doc.page_content for doc, _ in filtered_docs)
        prompt = f"""
다음 문서를 참고하여 질문에 답변해 주세요.

문서:
{context}

질문: {user_query}
답변:
"""

        # 🤖 모델 로딩
        if st.session_state.model is None:
            with st.spinner("🧠 LoRA 모델 로딩 중..."):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/pythia-70m",
                    device_map="auto",
                    torch_dtype=torch.float16 if st.session_state.device == "cuda" else torch.float32
                )
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    base_model.config.pad_token_id = tokenizer.eos_token_id

                model = PeftModel.from_pretrained(base_model, "slm_lora")
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer

        # 🗣️ 응답 생성
        inputs = st.session_state.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(st.session_state.device)
        
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.pad_token_id
            )
        answer = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 답변 포맷팅
        def format_answer(text):
            # 불필요한 공백 제거
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text

        # 💬 대화 저장
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", format_answer(answer)))

    except Exception as e:
        st.error(f"❌ 처리 중 오류: {str(e)}")
        st.error("상세 오류 정보:")
        st.error(traceback.format_exc())
        st.stop()

# ✅ 대화 기록 출력
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
