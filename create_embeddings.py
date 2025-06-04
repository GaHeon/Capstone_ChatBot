from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os
import re
import gc
from tqdm import tqdm
import multiprocessing

def split_to_chunks(text, min_length=30, max_length=300):
    # 문단 단위로 먼저 분리
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > min_length]
    chunks = []
    
    for para in paragraphs:
        # 문장 단위로 분리 (한글 문장 구분자 추가)
        sentences = re.split(r'(?<=[.!?。]) +', para)
        current_chunk = ""
        
        for sent in sentences:
            # 현재 청크에 문장을 추가했을 때의 길이
            potential_length = len(current_chunk) + len(sent) + 1
            
            if potential_length <= max_length:
                current_chunk += (sent + " ").strip()
            else:
                if len(current_chunk.strip()) >= min_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sent + " "
        
        # 마지막 청크 처리
        if len(current_chunk.strip()) >= min_length:
            chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings():
    print("🔄 임베딩 생성 시작...")
    
    # CPU 스레드 수 제한 (전체 코어의 50%만 사용)
    num_threads = max(1, multiprocessing.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    print(f"CPU 스레드 수: {num_threads}")
    
    # 디바이스 설정
    device = "cpu"  # CPU만 사용
    print(f"사용 중인 디바이스: {device}")
    
    # 메모리 정리
    gc.collect()
    
    # 더 가벼운 임베딩 모델 사용
    print("📥 임베딩 모델 로딩 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",  # 더 가벼운 모델
        model_kwargs={'device': device}
    )
    
    texts = []
    metadatas = []
    
    # 파일 목록 가져오기
    files = [f for f in os.listdir("chunks") if f.endswith(".txt")]
    print(f"📚 총 {len(files)}개의 파일 처리 예정")
    
    # tqdm으로 진행 상황 표시
    for filename in tqdm(files, desc="파일 처리 중"):
        try:
            print(f"\n📄 {filename} 처리 중...")
            with open(os.path.join("chunks", filename), "r", encoding="utf-8") as f:
                content = f.read()
                # 문장 단위로 청크 분할
                chunks = split_to_chunks(content, min_length=30, max_length=300)
                print(f"  - {len(chunks)}개의 청크 생성됨")
                
                for idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({"source": filename, "chunk": idx})
                
                # 메모리 관리
                if len(texts) % 100 == 0:  # 더 자주 메모리 정리
                    gc.collect()
        except Exception as e:
            print(f"❌ {filename} 처리 중 오류 발생: {str(e)}")
            continue
    
    print(f"\n📊 총 {len(texts)}개의 청크 생성됨")
    
    # 중복 제거 (텍스트 기준)
    print("🔄 중복 제거 중...")
    unique = list({text: meta for text, meta in zip(texts, metadatas)}.items())
    texts = [u[0] for u in unique]
    metadatas = [u[1] for u in unique]
    print(f"📊 중복 제거 후 {len(texts)}개의 고유한 청크 남음")
    
    # 벡터 DB 생성
    print("\n🔄 FAISS 벡터 DB 생성 중...")
    try:
        # 더 작은 배치 크기로 처리
        batch_size = 100  # 배치 크기 축소
        for i in tqdm(range(0, len(texts), batch_size), desc="벡터 DB 생성 중"):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            if i == 0:
                db = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
            else:
                db.add_texts(batch_texts, metadatas=batch_metadatas)
            
            # 메모리 관리
            gc.collect()
        
        # 저장
        print("\n💾 벡터 DB 저장 중...")
        db.save_local("embed_faiss")
        print("✅ 임베딩 생성 완료!")
        
    except Exception as e:
        print(f"❌ 벡터 DB 생성 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    create_embeddings() 