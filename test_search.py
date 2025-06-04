from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import multiprocessing

def test_search(query, k=5):
    print(f"\n🔍 검색어: {query}")
    print("-" * 50)
    
    # CPU 스레드 수 제한
    num_threads = max(1, multiprocessing.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    
    # 디바이스 설정
    device = "cpu"
    
    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': device}
    )
    
    # FAISS DB 로드
    db = FAISS.load_local("embed_faiss", embeddings, allow_dangerous_deserialization=True)
    
    # 검색 실행
    docs = db.similarity_search_with_score(query, k=k)
    
    # 거리를 유사도로 변환
    def distance_to_similarity(distance):
        return 1 / (1 + distance)
    
    # 결과 출력
    for i, (doc, score) in enumerate(docs, 1):
        similarity = distance_to_similarity(score)
        print(f"\n📄 결과 {i} (유사도: {similarity:.3f})")
        print(f"출처: {doc.metadata['source']} / 청크 번호: {doc.metadata['chunk']}")
        print(f"내용: {doc.page_content[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    while True:
        query = input("\n검색어를 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
        test_search(query) 