# python3
# Create Date: 2025-04-08
# Func: 文本处理 & 相似
# ==========================================================================================

from typing import Dict, List, Tuple, Optional, Union
import os 
import sys 
import json 
from tqdm.auto import tqdm
import numpy as np 
from openai import OpenAI
api_key = os.environ['DEEPSEEK_API_KEY']
df_api_key = os.environ['DEEPINFRA_API_KEY']
ali_api_key = os.environ['ALI_API_KEY']
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
emb_client = OpenAI(api_key=df_api_key, base_url="https://api.deepinfra.com/v1/openai")
ali_client = OpenAI(api_key=ali_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
'ali_client qwen-long'


def load_documents(dir_path: str) -> List[str]:
    docs = []
    for f in os.listdir(dir_path):
        if f.endswith('.txt'):
            with open(os.path.join(dir_path, f), 'r', encoding='utf-8') as f_o:
                docs.append(f_o.read())
    return docs


def preprocess_text(text: str) -> str:
    text = text.lower()
    # Remove special characters, keeping only alphanumeric characters and spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text


def split_into_clean_chunks(docs: List[str], chunk_size: int = 30) -> List[str]:
    chunks = []  
    for doc in docs:  
        words = doc.split()  
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])   
            chunks.append(preprocess_text(chunk))
    return chunks 


def prepare_text_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    documents = load_documents(dir_p)
    preprocessed_chunks = split_into_clean_chunks(documents)
    for i in range(2):
        print(f"Chunk {i+1}: {preprocessed_chunks[i][:50]} ... ")
        print("-" * 50)  


# embedding & store vector
# -------------------------------------------------------------------------------------------------------------------
def generate_embeddings_batch(chunks_batch: List[str], model: str = "BAAI/bge-en-icl", emb_client=emb_client) -> List[List[float]]:
    """
    # 智源发布三款BGE新模型，再次刷新向量检索最佳水平 https://zhuanlan.zhihu.com/p/711891274
    # Making Text Embedders Few-Shot Learners https://arxiv.org/abs/2409.15700
    # 带指令的embedding是否更配RAG? https://zhuanlan.zhihu.com/p/675559315
    # Bge-en-icl: 当in-context learning遇上了text embedding... https://zhuanlan.zhihu.com/p/743630798
    # BAAI/bge-en-icl 	$0.0100/Mtoken  https://deepinfra.com/BAAI/bge-en-icl/api  
    # text-embedding-v3: 0.5/Mtoken https://help.aliyun.com/zh/model-studio/user-guide/embedding
    """
    response = emb_client.embeddings.create(
        model=model,   
        input=chunks_batch,
        encoding_format='float'
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


def generate_embeddings(chunks: List[str], batch_size: int = 10) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        # Extract the current batch of chunks
        batch = chunks[i:i + batch_size]
        embeddings = generate_embeddings_batch(batch)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


def save_embeddings(embeddings: np.ndarray, output_file: str) -> None:
    # Open the specified file in write mode with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as file:
        # Convert the NumPy array to a list and save it as JSON
        json.dump(embeddings.tolist(), file)


def load_embedding(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        # Convert the NumPy array to a list and save it as JSON
        emb = np.array(json.load(f))
    return emb


def embedding_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    emb_f = os.path.join(cur_p, "data", "embeddings.json")
    documents = load_documents(dir_p)
    preprocessed_chunks = split_into_clean_chunks(documents)
    # Generate embeddings for the preprocessed chunks
    embeddings = generate_embeddings(preprocessed_chunks)
    save_embeddings(embeddings, emb_f)    
    print('Finished embedding_test')


vector_store: dict[int, dict[str, object]] = {}

# Function to add embeddings and corresponding text chunks to the vector store
def add_to_vector_store(embeddings: np.ndarray, chunks: List[str]) -> None:
    """
    Add embeddings and their corresponding text chunks to the vector store.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to add.
        chunks (List[str]): A list of text chunks corresponding to the embeddings.

    Returns:
        None
    """
    # Iterate over embeddings and chunks simultaneously
    for embedding, chunk in zip(embeddings, chunks):
        # Add each embedding and its corresponding chunk to the vector store
        # Use the current length of the vector store as the unique key
        vector_store[len(vector_store)] = {"embedding": embedding, "chunk": chunk}


# def vector2pickle(v, v_f):
#     with open('')


# similarity search
# -------------------------------------------------------------------------------------------------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量之间的余弦相似度。

    参数：
        vec1 (np.ndarray): 第一个向量。
        vec2 (np.ndarray): 第二个向量。

    返回：
        float: 两个向量之间的余弦相似度，范围在 -1 到 1 之间。
    """
    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)
    # 返回余弦相似度
    return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
    """
    Perform similarity search in the vector store and return the top_k most similar chunks.

    Args:
        query_embedding (np.ndarray): The embedding vector of the query.
        top_k (int): The number of most similar chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most similar text chunks.
    """
    similarities = []
    for key, value in vector_store.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        similarities.append((key, similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Retrieve the top_k most similar strings based on their keys
    return [vector_store[key]["chunk"] for key, _ in similarities[:top_k]]


def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant document chunks for a given query text.

    Args:
        query_text (str): The query text for which relevant chunks are to be retrieved.
        top_k (int): The number of most relevant chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most relevant text chunks.
    """
    # Generate embedding for the query text using the embedding model
    query_embedding = generate_embeddings([query_text])[0]
    relevant_chunks = similarity_search(query_embedding, top_k=top_k)
    return relevant_chunks


def search_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    documents = load_documents(dir_p)
    preprocessed_chunks = split_into_clean_chunks(documents)
    emb_f = os.path.join(cur_p, "data", "embeddings.json")
    embeddings = load_embedding(emb_f)
    add_to_vector_store(embeddings, preprocessed_chunks)
    
    query_text = "What is Quantum Computing?"
    relevant_chunks = retrieve_relevant_chunks(query_text)
    for idx, chunk in enumerate(relevant_chunks):
        print(f"Chunk {idx + 1}: {chunk[:50]} ... ")
        print("-" * 50)  # Print a separator line


# LLM Response Generation-basic RAG
# -------------------------------------------------------------------------------------------------------------------
def construct_prompt(query: str, context_chunks: List[str]) -> str:
    """
    通过将查询与检索到的上下文片段结合，构建提示。
    参数：
        query (str): 要构建提示的查询文本。
        context_chunks (List[str]): 要包含在提示中的相关上下文片段列表。

    返回：
        str: 用于作为 LLM 输入的构建好的提示。
    """
    # chinese_prompt_template = """
    # System:
    # 你是一个问答机器人。你的任务是根据下述给定的已知信息回答用户问题。
    # 如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

    # 已知信息:
    # {context} # 检索出来的原始文档

    # Question:
    # {query} # 用户的提问

    # Answer:
    # """
    context = "\n".join(context_chunks)
    system_message = (
        "You are a helpful assistant. Only use the provided context to answer the question. "
        "If the context doesn't contain the information needed, say 'I don't have enough information to answer this question.'"
    )
    prompt = f"System: {system_message}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    return prompt


def generate_response(
    prompt: str,
    model: str = 'qwen-long', # "deepseek-chat",
    client_in = None,
    max_tokens: int = 512,
    temperature: float = 1,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    根据构建的prompt从OpenAI-模型生成回答

    Args:
        prompt (str): construct_prompt 生成的提示词
        model (str): LLM default "google/gemma-2-2b-it".
        max_tokens (int): 生成回答的最多tokens数  Default is 512.
        temperature (float): Sampling temperature for response diversity. Default is 0.5.
        top_p (float): Probability mass for nucleus sampling. Default is 0.9.
        top_k (int): Number of highest probability tokens to consider. Default is 50.

    Returns:
        str: The generated response from the chat model.
    """
    # Use the OpenAI client to create a chat completion
    client = client_in if client_in is not None else client
    # print(f"{model=}")
    content = prompt if 'qwen' in model else [ 
        {"type": "text",  "text": prompt}
    ]
    response = client.chat.completions.create(
        model=model,   
        max_tokens=max_tokens,   
        temperature=temperature,   
        top_p=top_p,   
        extra_body={   
            "top_k": top_k  
        },
        messages=[  # List of messages to provide context for the chat model
            {
                "role": "user", 
                "content": content
            }
        ]
    )
    # Return the content of the first choice in the response
    return response.choices[0].message.content


# Function to implement the basic Retrieval-Augmented Generation (RAG) pipeline
def basic_rag_pipeline(query: str, model: str="deepseek-chat", api_client=None) -> str:
    """
    实现基础检索增强生成(RAG) pipeline
    检索相关片段 -> 构建提示 -> 并生成回答
    Args:
        query (str): 输入查询，用于生成回答。
    Returns:
        str: 基于查询和检索到的上下文，由 LLM 生成的回答。
    """
    # Step 1: 检索与给定query最相关的片段
    relevant_chunks: List[str] = retrieve_relevant_chunks(query)
    
    # Step 2: 使用query和检索到的片段构建提示
    prompt: str = construct_prompt(query, relevant_chunks)
    
    # Step 3: 使用构建好的提示从 LLM 生成回答
    response: str = generate_response(prompt, model=model, client_in=api_client)
    return response


def RAG_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    documents = load_documents(dir_p)
    preprocessed_chunks = split_into_clean_chunks(documents)
    emb_f = os.path.join(cur_p, "data", "embeddings.json")
    embeddings = load_embedding(emb_f)
    add_to_vector_store(embeddings, preprocessed_chunks)

    test_f = os.path.join(cur_p, "data", 'val.json')
    with open(test_f, 'r') as file:
        validation_data = json.load(file)

    sample_query = validation_data['basic_factual_questions'][0]['question']  
    expected_answer = validation_data['basic_factual_questions'][0]['answer']  

    print(f"Sample Query: {sample_query}\n")
    print(f"Expected Answer: {expected_answer}\n")

    print("🔍 Running the Retrieval-Augmented Generation (RAG) pipeline...")
    print(f"📥 Query: {sample_query}\n")

    # Run the RAG pipeline and get the response
    #  $0.02/$0.04 in/out Mtoken  google/gemma-3-4b-it
    # Instruction Tuned: The "it" in the model name indicates that this version is instruction-tuned, meaning it has been optimized to follow instructions more effectively.
    # response = basic_rag_pipeline(sample_query, model='google/gemma-3-4b-it', api_client=emb_client)
    # response = basic_rag_pipeline(sample_query, model='deepseek-chat', api_client=client)
    response = basic_rag_pipeline(sample_query, model='qwen-long', api_client=ali_client)
    # Print the response with better formatting
    print("🤖 AI Response:")
    print("-" * 50)
    print(response.strip())
    print("-" * 50)

    # Print the ground truth answer for comparison
    print("✅ Ground Truth Answer:")
    print("-" * 50)
    print(expected_answer)
    print("-" * 50)
    response_embedding = generate_embeddings([response])[0]
    ground_truth_embedding = generate_embeddings([expected_answer])[0]
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    
    print(f"✅ similarity: {similarity:.5f}")


res_ = """ 
    Sample Query: What is the mathematical representation of a qubit in superposition?
    Expected Answer: |ψ⟩ = α|0⟩ + β|1⟩, where α and β are complex numbers satisfying |α|² + |β|² = 1, representing the probability amplitudes for measuring the qubit in state |0⟩ or |1⟩ respectively.

    🔍 Running the Retrieval-Augmented Generation (RAG) pipeline...
    📥 Query: What is the mathematical representation of a qubit in superposition?


    🤖 AI Response:
    --------------------------------------------------
    The mathematical representation of a qubit in superposition is given by:  
    **ψ = α|0⟩ + β|1⟩**,  
    where α and β are complex numbers satisfying |α|² + |β|² = 1. These represent the probability amplitudes for measuring the qubit in state |0⟩ or |1⟩, respectively.  

    (Answer derived directly from the provided context.)
    --------------------------------------------------
    ✅ Ground Truth Answer:
    --------------------------------------------------
    |ψ⟩ = α|0⟩ + β|1⟩, where α and β are complex numbers satisfying |α|² + |β|² = 1, representing the probability amplitudes for measuring the qubit in state |0⟩ or |1⟩ respectively.
    --------------------------------------------------
    ✅ similarity: 0.92927
"""
import chromadb
from chromadb.config import Settings
from functools import partial


class simpleVectorDB:
    def __init__(self, collection_name, embedding_fn=generate_embeddings_batch):
        self.collection_name = collection_name
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn
        self.add_counts = 0

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        # print(f'self.add_counts={self.add_counts}', documents[:2])
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[f"id{self.add_counts}_{i}" for i in range(len(documents))]
        )
        self.add_counts += 1

    def reset(self):
        self.chroma_client.reset()
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self.add_counts = 0

    def search(self, query: str, top_n: int=5):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results['documents'][0]

    def batch_add_documents(self, chunks: List[str], batch_size: int = 10):
        all_embeddings = []
        for i in tqdm(range(0, len(chunks), batch_size)):
            self.add_documents(chunks[i:i + batch_size])


def vdb_search_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    documents = load_documents(dir_p)
    preprocessed_chunks = split_into_clean_chunks(documents)
    v_db = simpleVectorDB(
        'ragVectorDB-tt1', 
        partial(generate_embeddings_batch, model='text-embedding-v3', emb_client=ali_client)
    )
    v_db.reset()
    v_db.batch_add_documents(preprocessed_chunks)
    query_text = "What is Quantum Computing?"
    relevant_chunks = v_db.search(query_text)
    print(relevant_chunks)
    for idx, chunk in enumerate(relevant_chunks):
        print(f"Chunk {idx + 1}: {chunk[:50]} ... ")
        print("-" * 50)  # Print a separator line


def basic_vdb_rag_pipeline(vdb, query: str, model: str="deepseek-chat", api_client=None) -> str:
    """
    实现基础检索增强生成(RAG) pipeline
    向量数据库中检索相关片段 -> 构建提示 -> 并生成回答
    Args:
        query (str): 输入查询，用于生成回答。
    Returns:
        str: 基于查询和检索到的上下文，由 LLM 生成的回答。
    """
    relevant_chunks = vdb.search(query)
    prompt = construct_prompt(query, relevant_chunks)
    response = generate_response(prompt, model=model, client_in=api_client)
    return response


class VDB_RAG_Bot:
    def __init__(
        self, 
        collection_name: str='ragVectorDB', 
        client: OpenAI = ali_client,
        embedding_fn=partial(generate_embeddings_batch, model='text-embedding-v3', emb_client=ali_client)
    ):
        self.v_db = simpleVectorDB(
            'ragVectorDB', 
            partial(generate_embeddings_batch, model='text-embedding-v3', emb_client=ali_client)
        )
        self.client = client
    
    def db_prepare(self, doc_dir: str):
        documents = load_documents(doc_dir)
        preprocessed_chunks = split_into_clean_chunks(documents)
        self.v_db.batch_add_documents(preprocessed_chunks)
    
    def retrieve_relevant_chunks(self, query_text: str, top_k: int = 5):
        return self.v_db.search(query_text, top_k)
    
    def chat(self, 
            query: str,
            model: str = "qwen-long",
            max_tokens: int = 512,
            temperature: float = 1,
            top_p: float = 0.9,
            top_k: int = 50
    ):
        """
        Args:
            query (str): 提问
            model (str): LLM default "qwen-long".
            max_tokens (int): 生成回答的最多tokens数  Default is 512.
            temperature (float): Sampling temperature for response diversity. Default is 0.5.
            top_p (float): Probability mass for nucleus sampling. Default is 0.9.
            top_k (int): Number of highest probability tokens to consider. Default is 50.
        """
        relevant_chunks = self.v_db.search(query)
        prompt = construct_prompt(query, relevant_chunks)
        response = generate_response(
            prompt, 
            model=model, 
            client_in=self.client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        return response
    
    def generate_response(self, 
        prompt: str,
        model: str = "qwen-long",
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        return generate_response(
            prompt, 
            model=model, 
            client_in=self.client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )


def VDB_RAG_test():
    cur_p = os.path.dirname(__file__)
    dir_p = os.path.join(cur_p, "data")
    ali_client = OpenAI(api_key=ali_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    chat_box = VDB_RAG_Bot(
        collection_name='ragVectorDB', 
        client=ali_client,
        embedding_fn=partial(generate_embeddings_batch, model='text-embedding-v3', emb_client=ali_client)
    )
    chat_box.db_prepare(dir_p)

    test_f = os.path.join(cur_p, "data", 'val.json')
    with open(test_f, 'r') as file:
        validation_data = json.load(file)

    sample_query = validation_data['basic_factual_questions'][0]['question']  
    expected_answer = validation_data['basic_factual_questions'][0]['answer']  

    print(f"Sample Query: {sample_query}\n")
    print(f"Expected Answer: {expected_answer}\n")

    print("🔍 Running the Retrieval-Augmented Generation (RAG) pipeline...")
    print(f"📥 Query: {sample_query}\n")
    
    response = chat_box.chat(sample_query)
    print("🤖 AI Response:")
    print("-" * 50)
    print(response.strip())
    print("-" * 50)

    # Print the ground truth answer for comparison
    print("✅ Ground Truth Answer:")
    print("-" * 50)
    print(expected_answer)
    print("-" * 50)
    response_embedding = generate_embeddings_batch([response])[0]
    ground_truth_embedding = generate_embeddings_batch([expected_answer])[0]
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    
    print(f"✅ similarity: {similarity:.5f}")



if __name__ == '__main__':
    # print(f"{api_key=}")
    # prepare_text_test()
    # 生成embedding
    # embedding_test()
    # 检索相关片段
    # search_test()
    # 检索相关片段 -> 构建提示 -> 并生成回答
    # RAG_test()
    # vdb_search_test()
    VDB_RAG_test()





