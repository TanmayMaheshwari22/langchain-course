import os
from groq import Groq
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def read_blog(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split(" ")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

file_path = "./blog.txt"
blog_content = read_blog(file_path)
document_chunks = chunk_text(blog_content,chunk_size=600)

# 3. Setup Open-Source Vector DB (Stays the same)
qdrant_client = QdrantClient(":memory:")
collection_name = "local_blog_knowledge"

qdrant_client.add(
    collection_name=collection_name,
    documents=document_chunks,
    metadata=[{"source": file_path} for _ in document_chunks]
)

# 4. Perform Retrieval
user_query = "What is the primary conclusion of the author in this file?"

search_results = qdrant_client.query(
    collection_name=collection_name,
    query_text=user_query,
    limit=2
)


retrieved_context = "\n\n".join([hit.document for hit in search_results])

# 5. Connect to Generation Model (Groq LLM)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_instruction = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the retrieved context provided below. If you don't know, say you don't know."
)

augmented_prompt = f"Context:\n{retrieved_context}\n\nUser Question: {user_query}"

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": augmented_prompt}
    ],
    temperature=0.2
)

print("\n=== ANSWER ===")
print(response.choices[0].message.content)



