이 문서는 LLaMA-Factory로 GPT-OSS 모델을 inference 하면서,
외부 문서를 활용한 RAG(검색 기반 강화) 챗봇을 만드는 전체 과정을 정리한 것입니다.

라마 팩토리 리드미를 참고하여 환경 구축을 완료한 후 새로운 패키지를 추가로 설치해주세요 

### 1-1. 추가 패키지 설치
```bash
pip install "openai>=1.0.0"             langchain             langchain-openai             langchain-community             chromadb             sentence-transformers
```

### 1-2. [중요!!] GPT-OSS 모델로 API 서버 실행

> ⚠️ 이 명령은 **별도 터미널에서 계속 켜둔 상태**로 사용합니다.

```bash
cd ./LLaMA-Factory
conda activate your envs

API_PORT=8000 llamafactory-cli api   --model_name_or_path ./weights/gpt-oss-20b   --template gpt   --infer_backend huggingface   --trust_remote_code true
```

이제 LLaMA-Factory가 **OpenAI 호환 /v1 API 서버**로 동작하며,  
엔드포인트는 `http://localhost:8000/v1` 입니다.

## 📌 2. RAG용 문서 준비

RAG에 사용할 문서들을 아래 폴더에 저장합니다.

```bash
mkdir -p ./Rag_docs
```

예시 구조:

```text
./Rag_docs
 ├─ Ewart.txt
 ├─ 연구노트1.txt
 └─ 안내문.txt
```

현재 예시는 `.txt` 파일을 기준으로 되어 있습니다.

### 3. RAG.py 실행 (터미널 2) <- 반드시 별도 터미널에서 

```bash
conda activate hyndai
cd ./AI-Enabled-IFTA/LLaMA-Factory
python RAG.py
```

### 실행결과

유림이는 **왕바보**입니다
예시파일을 RAG_docs에 만들어서 시도해보세요 !
<img width="925" height="283" alt="result" src="https://github.com/user-attachments/assets/1abc81db-f055-46c7-aa51-e68f573d4299" />


