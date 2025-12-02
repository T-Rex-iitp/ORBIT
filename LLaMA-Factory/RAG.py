# RAG.py
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================================
# 0. 경로 설정 (네 환경에 맞게 조절)
# ================================
DOC_DIR = "./AI-Enabled-IFTA/LLaMA-Factory/RAG_docs"          # RAG에 쓸 문서 폴더
DB_DIR = "./AI-Enabled-IFTA/LLaMA-Factory/RAG_DB"      # 벡터DB 저장 폴더

BASE_URL = "http://localhost:8000/v1"       # LLaMA-Factory API 서버 주소
API_KEY = "EMPTY"                           # 아무 문자열이나 OK

# ================================
# 1. LLaMA-Factory OpenAI 서버 설정
# ================================
# └── 이 전에 다른 터미널에서 반드시:
# API_PORT=8000 llamafactory-cli api \
#   --model_name_or_path ./weights/gpt-oss \
#   --template gpt \
#   --infer_backend huggingface \
#   --trust_remote_code true
# 를 실행해두어야 함.

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

llm = ChatOpenAI(
    model="gpt-oss",    # 서버쪽에서 인식용 이름 (그냥 문자열이면 됨)
    base_url=BASE_URL,
    api_key=API_KEY,
)

# ================================
# 2. 문서 로딩 & 쪼개기 & 벡터DB 구축
# ================================
if not os.path.isdir(DOC_DIR):
    raise ValueError(f"문서 폴더가 없음: {DOC_DIR}")

print(f"[INFO] 문서 폴더에서 로딩 중: {DOC_DIR}")
# txt 파일 기준. 필요하면 glob="**/*.*" 등으로 바꿔도 됨.
loader = DirectoryLoader(DOC_DIR, glob="**/*.txt", show_progress=True)
docs = loader.load()

if not docs:
    print("[WARN] 로드된 문서가 없습니다. DOC_DIR 안에 .txt 파일 넣어주세요.")
else:
    print(f"[INFO] 로드된 문서 개수: {len(docs)}")

print("[INFO] 문서를 청크로 쪼개는 중...")
# 여기서 "쪼개는" 부분:
# - chunk_size=800: 800자 정도마다 잘라서
# - chunk_overlap=200: 앞뒤로 200자씩 겹치게
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
splits = splitter.split_documents(docs)
print(f"[INFO] 만들어진 청크 개수: {len(splits)}")

print("[INFO] 임베딩 & 벡터 스토어(Chroma) 생성 중...")
os.makedirs(DB_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=DB_DIR,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ================================
# 3. 인터랙티브 RAG 채팅 루프
# ================================
print("\n=== RAG Chat 시작! (종료: /exit 또는 /quit 입력) ===\n")

while True:
    try:
        user_query = input("질문 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n종료합니다.")
        break

    if user_query in ("/exit", "/quit"):
        print("종료합니다.")
        break
    if not user_query:
        continue

    # 🔥 LangChain 최신 버전: retriever.invoke() 사용
    rel_docs = retriever.invoke(user_query)
    if not rel_docs:
        context = ""
    else:
        context = "\n\n---\n\n".join(d.page_content for d in rel_docs)

    system_prompt = (
        "너는 RAG 어시스턴트야. 아래 '컨텍스트' 내용을 최대한 활용해서 "
        "사용자의 질문에 답변해줘. 컨텍스트에 없는 내용은 모른다고 솔직히 말해."
    )

    user_content = (
        f"[컨텍스트]\n{context}\n\n"
        f"[질문]\n{user_query}"
    )

    # LLaMA-Factory의 OpenAI 호환 /v1/chat/completions 엔드포인트 호출
    response = client.chat.completions.create(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = response.choices[0].message.content
    print("\n[전체답변]")
    print(answer)
    answer_final = (lambda x: x.split("assistantfinal",1)[1].strip() if "assistantfinal" in x else x.strip())(response.choices[0].message.content)
    print("\n[최종답변]")
    print(answer_final)
    print("\n" + "=" * 60 + "\n")
