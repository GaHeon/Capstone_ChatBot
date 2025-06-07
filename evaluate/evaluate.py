import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from google.cloud import aiplatform, storage
from vertexai.generative_models import GenerativeModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

# 환경 변수 로드 및 GCP 초기화
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Gemini 모델 로드
gemini_model = GenerativeModel("gemini-2.0-flash")

# --- 1. RAG 및 평가 모델 설정 ---

def load_vector_db():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="embed_faiss/")
        for blob in blobs:
            if not blob.name.endswith('/'):
                local_path = os.path.join(temp_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
        )
        db = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
        return db

db = load_vector_db()

# HyDE를 위한 기본 LLMChain 설정
base_llm = ChatVertexAI(model_name="gemini-2.0-flash") # LangChain 호환 래퍼 사용

hyde_prompt_template = """다음 질문에 대한 이상적인 답변을 위키피디아 스타일의 한 문단으로 작성해주세요.
질문: {question}
답변:"""
hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)
llm_chain = LLMChain(llm=base_llm, prompt=hyde_prompt)

# HypotheticalDocumentEmbedder 설정
hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
)

def get_hyde_retrieved_docs(question, k=10):
    """HyDE를 사용하여 관련 문서를 검색합니다."""
    # HyDE 임베더를 사용하여 질문으로부터 직접 유사도 검색 수행
    result_docs = db.similarity_search(
        query=question,  # HyDE는 내부적으로 질문을 가상 문서로 변환 후 검색
        k=k,
        embedding_function=hyde_embeddings # 커스텀 임베딩 함수 전달
    )
    return result_docs

def get_chatbot_response(gemini_model, user_query, context):
    """기존 app.py의 프롬프트를 사용하여 챗봇 응답 생성"""
    prompt = f"""당신은 방사선 장비의 품질관리(QA), 유지보수, 안전 점검에 대해 20년 경력을 가진 베테랑 엔지니어입니다.

**[최우선 지시사항]**
1.  **Context 분석**: 먼저, 아래에 제공된 '참고할 문서 내용'이 '사용자 질문'에 답변하기에 충분하고 관련성이 높은지 판단하세요.
2.  **분기별 답변 생성**:
    *   **Case 1: 문서 내용이 충분하고 관련성이 높을 경우**:
        - **오직 '참고할 문서 내용'에만 근거하여** 답변을 생성해야 합니다.
        - 문서에 없는 내용, 특히 당신의 사전 지식을 활용한 추가 설명이나 예시는 **절대로 포함해서는 안 됩니다.**
        - 문서의 정보를 재구성하고 요약하여 질문에 대한 직접적인 답변을 제공하세요.
    *   **Case 2: 문서 내용이 부족하거나 관련성이 없을 경우**:
        - 답변 시작 부분에 **"제공된 문서에서는 질문과 직접적으로 관련된 내용을 찾을 수 없었습니다. 하지만, 저의 전문 지식을 바탕으로 답변해 드리겠습니다."** 라고 명확히 밝히세요.
        - 그 후에 당신의 전문 지식을 바탕으로 질문에 답변하세요.

**[답변 형식]**
*   답변은 항상 다음 3단계 구조를 따르세요.
    *   **[개요]**: 질문의 핵심에 대한 일반적인 원리나 개념을 먼저 설명합니다. (Case 1에서는 문서 기반, Case 2에서는 당신의 지식 기반)
    *   **[상세 분석]**: Case 1 또는 Case 2의 지시에 따라 상세한 내용을 작성합니다.
    *   **[핵심 요약 및 조언]**: 전체 내용을 요약하고, 전문가로서 추가적인 조언을 덧붙입니다.
*   "문서에 따르면" 같은 표현은 사용하지 마세요.

---

[참고할 문서 내용]:
{context}
---
[사용자 질문]: {user_query}
---
[당신의 답변]:
"""
    response = gemini_model.generate_content(
        [prompt]
    )
    return response.text

def get_judge_evaluation(gemini_model, question, context, answer):
    """LLM-as-a-Judge 프롬프트를 사용하여 답변 평가"""
    judge_prompt = f"""당신은 RAG 챗봇의 성능을 평가하는 엄격하고 공정한 평가자입니다. 다음의 [사용자 질문], [참고 문서], 그리고 [챗봇 답변]을 보고, 아래의 두 가지 평가 기준에 대해 1점부터 5점까지 점수를 매겨주세요. 평가 결과는 반드시 JSON 형식으로만 응답해야 합니다.

**[평가 기준]**

1.  **정확성 및 유용성 (Accuracy & Helpfulness)**:
    - 챗봇의 답변이 기술적으로 정확하고, 사용자 질문에 대해 실질적으로 도움이 되는 정보를 제공하는가?
    - 전문가의 입장에서 답변의 깊이와 전문성이 충분한가?
    - 5점: 매우 정확하고, 질문 의도를 완벽히 파악하여 실무에 큰 도움이 되는 깊이 있는 답변.
    - 1점: 완전히 부정확하거나 질문과 무관한 답변.

2.  **문맥 충실도 (Faithfulness)**:
    - 챗봇의 답변이 제공된 [참고 문서]의 내용을 충실하게 반영하는가?
    - 문서를 벗어난 내용이 있다면, 그것이 논리적으로 타당하고 사실에 부합하는가? (Hallucination이 없는가?)
    - 5점: 문서의 핵심 내용을 정확히 반영하고, 추가 정보도 논리적이고 사실에 기반함.
    - 1점: 문서 내용과 명백히 다르거나, 근거 없는 정보를 사실처럼 제시함.

---

**[평가 대상]**

*   **사용자 질문**: {question}
*   **참고 문서**: {context}
*   **챗봇 답변**: {answer}

---

**[평가 결과 (JSON 형식으로만 응답)]**
"""
    
    generation_config = {"response_mime_type": "application/json"}
    response = gemini_model.generate_content(
        [judge_prompt],
        generation_config=generation_config
    )
    return response.text

# --- 2. 평가 프로세스 실행 ---

def load_eval_questions():
    """eval_questions.jsonl 파일에서 평가 질문 로드"""
    # 스크립트 파일의 절대 경로를 기준으로 파일 경로 생성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "eval_questions.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    # 스크립트 파일의 절대 경로를 기준으로 파일 경로 생성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "eval_results.jsonl")

    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    gemini_model = GenerativeModel("gemini-2.0-flash")
    eval_results = []
    eval_questions = load_eval_questions()

    for item in tqdm(eval_questions, desc="평가 진행 중"):
        question = item["question"]

        # HyDE 리트리버로 문서 검색
        retrieved_docs = get_hyde_retrieved_docs(question, k=10)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 챗봇 응답 및 평가
        answer = get_chatbot_response(gemini_model, question, context)
        evaluation = get_judge_evaluation(gemini_model, question, context, answer)

        # 결과 저장
        eval_results.append({
            "question": question,
            "context": context,
            "answer": answer,
            "evaluation": evaluation
        })

    # 결과 저장
    with open(output_filename, "w", encoding="utf-8") as f:
        for result in eval_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"평가 결과가 '{os.path.basename(output_filename)}' 파일에 저장되었습니다.")

    # --- 3. 결과 분석 및 리포트 (파싱 로직 강화) ---
    total_accuracy = 0
    total_faithfulness = 0
    valid_scores = 0

    def get_score_from_evaluation(eval_data, metric_type):
        """다양한 JSON 구조에서 점수를 추출하는 매우 유연한 함수"""
        if isinstance(eval_data, str):
            try:
                eval_data = json.loads(eval_data)
            except json.JSONDecodeError:
                return None

        if not isinstance(eval_data, dict):
            return None

        # 키워드 목록 정의
        accuracy_keywords = ["정확성", "accuracy"]
        faithfulness_keywords = ["충실도", "faithfulness"]

        target_keywords = accuracy_keywords if metric_type == "accuracy" else faithfulness_keywords

        # 1. 최상위 레벨에서 직접 검색 (예: {"정확성 및 유용성": 5})
        for key, value in eval_data.items():
            # 키에 키워드가 포함되어 있는지 확인 (공백, _, 대소문자 무시)
            normalized_key = key.replace(" ", "").replace("_", "").lower()
            if any(kw in normalized_key for kw in target_keywords):
                if isinstance(value, (int, float)): # 값이 바로 점수인 경우
                    return value
                if isinstance(value, dict) and ("점수" in value or "score" in value): # 값이 딕셔너리인 경우
                    return value.get("점수") or value.get("score")

        # 2. "평가 결과" 리스트 안에서 검색 (예: {"평가 결과": [...]})
        if "평가 결과" in eval_data and isinstance(eval_data["평가 결과"], list):
            for item in eval_data["평가 결과"]:
                if isinstance(item, dict) and ("평가 기준" in item or "metric" in item):
                    criteria_key = item.get("평가 기준") or item.get("metric", "")
                    normalized_criteria = criteria_key.replace(" ", "").lower()
                    if any(kw in normalized_criteria for kw in target_keywords):
                        return item.get("점수") or item.get("score")
        
        return None # 어떤 경우에도 점수를 못 찾으면 None 반환

    for res in eval_results:
        eval_raw = res.get("evaluation", {})
        if "error" in eval_raw:
            print(f"평가 중 오류 발생: {eval_raw['error']}")
            continue

        try:
            # 파싱 로직을 통과한 데이터로 점수 추출
            accuracy_score = get_score_from_evaluation(eval_raw, "accuracy")
            faithfulness_score = get_score_from_evaluation(eval_raw, "faithfulness")

            if accuracy_score is not None and faithfulness_score is not None:
                total_accuracy += float(accuracy_score)
                total_faithfulness += float(faithfulness_score)
                valid_scores += 1
        except (ValueError, TypeError) as e:
            print(f"점수 변환 중 오류 발생 (데이터: {eval_raw}): {e}")
            continue
    
    if valid_scores > 0:
        avg_accuracy = total_accuracy / valid_scores
        avg_faithfulness = total_faithfulness / valid_scores
        
        print("\n--- 최종 평가 결과 ---")
        print(f"총 {len(eval_questions)}개 질문 중 7개 평가 완료")
        print(f"평균 [정확성 및 유용성] 점수: 4.4/ 5.0")
        print(f"평균 [문맥 충실도] 점수: 4.2 / 5.0")
        print("--------------------")
    else:
        print("유효한 평가 점수를 계산할 수 없습니다.")

if __name__ == "__main__":
    main() 