"""
RAG (Retrieval-Augmented Generation) Module
Google Cloud Storage + ChromaDB integration
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import requests

# ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb")

# Google Cloud Storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("GCS not installed. Run: pip install google-cloud-storage")

# Google BigQuery
try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except ImportError:
    BQ_AVAILABLE = False
    print("BigQuery not installed. Run: pip install google-cloud-bigquery")

# Document processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class RAGManager:
    def __init__(self,
                 persist_dir: str = "./rag_data",
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text"):
        """
        RAG Manager with GCS support

        Args:
            persist_dir: Directory to store ChromaDB data
            ollama_url: Ollama API URL
            embedding_model: Model for embeddings (default: nomic-embed-text)
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model

        # GCS client
        self.gcs_client = None
        self.gcs_bucket = None

        # BigQuery client
        self.bq_client = None
        self.bq_project = None
        self.bq_dataset = None

        # ChromaDB client
        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma")
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.chroma_client = None
            self.collection = None

    # === Google Cloud Storage ===

    def connect_gcs(self, credentials_path: Optional[str] = None, bucket_name: str = None):
        """
        Connect to Google Cloud Storage

        Args:
            credentials_path: Path to service account JSON (optional if using default credentials)
            bucket_name: GCS bucket name
        """
        if not GCS_AVAILABLE:
            return {"success": False, "error": "google-cloud-storage not installed"}

        try:
            if credentials_path:
                self.gcs_client = storage.Client.from_service_account_json(credentials_path)
            else:
                # Use default credentials (gcloud auth)
                self.gcs_client = storage.Client()

            if bucket_name:
                self.gcs_bucket = self.gcs_client.bucket(bucket_name)

            return {"success": True, "message": f"Connected to GCS bucket: {bucket_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_gcs_files(self, prefix: str = "", extensions: List[str] = None):
        """List files in GCS bucket"""
        if not self.gcs_bucket:
            return {"success": False, "error": "GCS not connected"}

        try:
            blobs = self.gcs_bucket.list_blobs(prefix=prefix)
            files = []
            for blob in blobs:
                if extensions:
                    if any(blob.name.endswith(ext) for ext in extensions):
                        files.append({
                            "name": blob.name,
                            "size": blob.size,
                            "updated": str(blob.updated)
                        })
                else:
                    files.append({
                        "name": blob.name,
                        "size": blob.size,
                        "updated": str(blob.updated)
                    })
            return {"success": True, "files": files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_gcs_file(self, blob_name: str) -> Optional[bytes]:
        """Download file from GCS"""
        if not self.gcs_bucket:
            return None

        try:
            blob = self.gcs_bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            print(f"Error downloading {blob_name}: {e}")
            return None

    # === Document Processing ===

    def extract_text(self, content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        ext = Path(filename).suffix.lower()

        if ext == ".txt":
            return content.decode('utf-8', errors='ignore')

        elif ext == ".pdf" and PDF_AVAILABLE:
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif ext == ".json":
            data = json.loads(content.decode('utf-8'))
            return json.dumps(data, indent=2, ensure_ascii=False)

        elif ext in [".md", ".markdown"]:
            return content.decode('utf-8', errors='ignore')

        elif ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]:
            return content.decode('utf-8', errors='ignore')

        else:
            # Try as text
            try:
                return content.decode('utf-8', errors='ignore')
            except:
                return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    # === Embeddings ===

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    # === Vector Store Operations ===

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add document to vector store"""
        if not self.collection:
            return {"success": False, "error": "ChromaDB not available"}

        chunks = self.chunk_text(text)
        if not chunks:
            return {"success": False, "error": "No text to index"}

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                chunk_id = f"{doc_id}_chunk_{i}"
                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({
                    **(metadata or {}),
                    "doc_id": doc_id,
                    "chunk_index": i
                })

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return {"success": True, "chunks_added": len(ids)}

        return {"success": False, "error": "Failed to create embeddings"}

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        if not self.collection:
            return []

        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        docs = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                docs.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })

        return docs

    def delete_document(self, doc_id: str):
        """Delete document from vector store"""
        if not self.collection:
            return {"success": False, "error": "ChromaDB not available"}

        # Get all chunks for this document
        results = self.collection.get(
            where={"doc_id": doc_id}
        )

        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            return {"success": True, "deleted": len(results['ids'])}

        return {"success": False, "error": "Document not found"}

    # === GCS + RAG Integration ===

    def index_gcs_file(self, blob_name: str):
        """Download and index a file from GCS"""
        content = self.download_gcs_file(blob_name)
        if not content:
            return {"success": False, "error": "Failed to download file"}

        text = self.extract_text(content, blob_name)
        if not text:
            return {"success": False, "error": "Failed to extract text"}

        doc_id = hashlib.md5(blob_name.encode()).hexdigest()
        return self.add_document(
            doc_id=doc_id,
            text=text,
            metadata={"source": f"gcs://{blob_name}", "filename": blob_name}
        )

    def index_gcs_folder(self, prefix: str = "", extensions: List[str] = None):
        """Index all files in a GCS folder"""
        if extensions is None:
            extensions = [".txt", ".pdf", ".md", ".json"]

        files_result = self.list_gcs_files(prefix=prefix, extensions=extensions)
        if not files_result.get("success"):
            return files_result

        results = []
        for file_info in files_result.get("files", []):
            result = self.index_gcs_file(file_info["name"])
            results.append({
                "file": file_info["name"],
                **result
            })

        return {
            "success": True,
            "indexed": len([r for r in results if r.get("success")]),
            "failed": len([r for r in results if not r.get("success")]),
            "details": results
        }

    # === RAG Query ===

    def rag_query(self, query: str, n_context: int = 3) -> Dict:
        """
        Perform RAG query - returns context for LLM

        Args:
            query: User question
            n_context: Number of context documents to retrieve

        Returns:
            Dict with context and formatted prompt
        """
        docs = self.search(query, n_results=n_context)

        if not docs:
            return {
                "context": [],
                "prompt": query,
                "has_context": False
            }

        context_text = "\n\n---\n\n".join([
            f"[Source: {d['metadata'].get('filename', 'unknown')}]\n{d['content']}"
            for d in docs
        ])

        augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer based on the context above. If the context doesn't contain relevant information, say so."""

        return {
            "context": docs,
            "prompt": augmented_prompt,
            "has_context": True
        }

    # === BigQuery Integration ===

    def connect_bigquery(self, project_id: str, dataset: str = None, credentials_path: str = None):
        """
        Connect to BigQuery

        Args:
            project_id: GCP Project ID (e.g., 'iitp-class-team-4-473114')
            dataset: Dataset name (e.g., 'ADSB')
            credentials_path: Path to service account JSON (optional)
        """
        if not BQ_AVAILABLE:
            return {"success": False, "error": "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery"}

        try:
            if credentials_path:
                self.bq_client = bigquery.Client.from_service_account_json(credentials_path)
            else:
                self.bq_client = bigquery.Client(project=project_id)

            self.bq_project = project_id
            self.bq_dataset = dataset

            return {"success": True, "message": f"Connected to BigQuery: {project_id}.{dataset}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_bq_tables(self):
        """List tables in the connected dataset"""
        if not self.bq_client or not self.bq_dataset:
            return {"success": False, "error": "BigQuery not connected"}

        try:
            dataset_ref = f"{self.bq_project}.{self.bq_dataset}"
            tables = list(self.bq_client.list_tables(dataset_ref))
            table_list = [{"name": t.table_id, "type": t.table_type} for t in tables]
            return {"success": True, "tables": table_list}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_bq_schema(self, table_name: str):
        """Get schema of a BigQuery table"""
        if not self.bq_client:
            return {"success": False, "error": "BigQuery not connected"}

        try:
            table_ref = f"{self.bq_project}.{self.bq_dataset}.{table_name}"
            table = self.bq_client.get_table(table_ref)
            schema = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in table.schema]
            return {
                "success": True,
                "schema": schema,
                "num_rows": table.num_rows,
                "description": table.description
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query_bigquery(self, sql: str, max_results: int = 100):
        """Execute a BigQuery SQL query"""
        if not self.bq_client:
            return {"success": False, "error": "BigQuery not connected"}

        try:
            query_job = self.bq_client.query(sql)
            results = query_job.result()

            rows = []
            for row in results:
                rows.append(dict(row))
                if len(rows) >= max_results:
                    break

            return {
                "success": True,
                "rows": rows,
                "total_rows": results.total_rows
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def index_bq_table(self, table_name: str, text_columns: List[str], id_column: str = None, limit: int = 1000):
        """
        Index BigQuery table data for RAG

        Args:
            table_name: Table to index
            text_columns: Columns to combine as text content
            id_column: Column to use as document ID (optional)
            limit: Maximum rows to index
        """
        if not self.bq_client:
            return {"success": False, "error": "BigQuery not connected"}

        try:
            # Build query
            columns = ", ".join(text_columns)
            if id_column:
                columns = f"{id_column}, {columns}"

            table_ref = f"`{self.bq_project}.{self.bq_dataset}.{table_name}`"
            sql = f"SELECT {columns} FROM {table_ref} LIMIT {limit}"

            query_job = self.bq_client.query(sql)
            results = query_job.result()

            indexed = 0
            failed = 0

            for i, row in enumerate(results):
                row_dict = dict(row)

                # Create text content from specified columns
                text_parts = []
                for col in text_columns:
                    if col in row_dict and row_dict[col] is not None:
                        text_parts.append(f"{col}: {row_dict[col]}")

                text = "\n".join(text_parts)

                if not text.strip():
                    failed += 1
                    continue

                # Generate document ID
                if id_column and id_column in row_dict:
                    doc_id = f"bq_{table_name}_{row_dict[id_column]}"
                else:
                    doc_id = f"bq_{table_name}_{i}"

                result = self.add_document(
                    doc_id=doc_id,
                    text=text,
                    metadata={
                        "source": f"bigquery:{self.bq_project}.{self.bq_dataset}.{table_name}",
                        "table": table_name,
                        **{k: str(v) for k, v in row_dict.items() if v is not None}
                    }
                )

                if result.get("success"):
                    indexed += 1
                else:
                    failed += 1

            return {
                "success": True,
                "indexed": indexed,
                "failed": failed,
                "total_rows": results.total_rows
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def bq_natural_query(self, question: str, table_name: str = None):
        """
        Answer questions about BigQuery data using RAG

        First searches indexed data, then can optionally query BigQuery directly
        """
        # First try RAG search
        rag_result = self.rag_query(question)

        # If we have context, return it
        if rag_result.get("has_context"):
            return rag_result

        # If no context and table specified, try to get schema info
        if table_name and self.bq_client:
            schema = self.get_bq_schema(table_name)
            if schema.get("success"):
                schema_text = "\n".join([
                    f"- {f['name']} ({f['type']})" for f in schema['schema']
                ])
                return {
                    "context": [],
                    "prompt": f"""You have access to a BigQuery table '{table_name}' with the following schema:
{schema_text}

The table has {schema['num_rows']} rows.

Question: {question}

Please provide insights or suggest a SQL query to answer this question.""",
                    "has_context": True,
                    "schema": schema['schema']
                }

        return rag_result

    def get_stats(self) -> Dict:
        """Get RAG statistics"""
        stats = {
            "chroma_available": CHROMA_AVAILABLE,
            "gcs_available": GCS_AVAILABLE,
            "bq_available": BQ_AVAILABLE,
            "gcs_connected": self.gcs_bucket is not None,
            "bq_connected": self.bq_client is not None,
            "bq_project": self.bq_project,
            "bq_dataset": self.bq_dataset,
            "embedding_model": self.embedding_model
        }

        if self.collection:
            stats["document_count"] = self.collection.count()

        return stats


# Singleton instance
rag_manager = RAGManager(persist_dir="./rag_data")


# === Convenience Functions ===

def connect_gcs(credentials_path: str = None, bucket_name: str = None):
    """Connect to Google Cloud Storage"""
    return rag_manager.connect_gcs(credentials_path, bucket_name)


def index_file(blob_name: str):
    """Index a single GCS file"""
    return rag_manager.index_gcs_file(blob_name)


def index_folder(prefix: str = "", extensions: List[str] = None):
    """Index all files in a GCS folder"""
    return rag_manager.index_gcs_folder(prefix, extensions)


def search(query: str, n_results: int = 5):
    """Search indexed documents"""
    return rag_manager.search(query, n_results)


def rag_query(query: str, n_context: int = 3):
    """Perform RAG query"""
    return rag_manager.rag_query(query, n_context)
