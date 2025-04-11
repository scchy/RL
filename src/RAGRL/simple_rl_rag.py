# python3
# Create Date: 2025-04-08
# Func: rl-rag
# reference: https://github.com/FareedKhan-dev/rag-with-rl/tree/main
# similar paper: Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning
# ==========================================================================================
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import numpy as np
from tqdm.auto import tqdm
from rag_utils import (
    generate_embeddings_batch, cosine_similarity, construct_prompt,
    VDB_RAG_Bot, partial
)
from openai import OpenAI


ali_api_key = os.environ['ALI_API_KEY']
ali_client = OpenAI(api_key=ali_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
chat_box = VDB_RAG_Bot(
    collection_name='ragVectorDB', 
    client=ali_client,
    embedding_fn=partial(generate_embeddings_batch, model='text-embedding-v3', emb_client=ali_client)
)
cur_p = os.path.dirname(__file__)
dir_p = os.path.join(cur_p, "data")
chat_box.db_prepare(dir_p)


def define_state(
    query: str, 
    context_chunks: List[str], 
    rewritten_query: str = None, 
    previous_responses: List[str] = None, 
    previous_rewards: List[float] = None
) -> dict:
    """
    RL-state.
    
    Args:
        query (str): 提出的问题.
        context_chunks (List[str]): Retrieved context chunks from the knowledge base.
        rewritten_query (str, optional): A reformulated version of the original query.
        previous_responses (List[str], optional): List of previously generated responses.
        previous_rewards (List[float], optional): List of rewards received for previous actions.
    
    Returns:
        dict: A dictionary representing the current state with all relevant information.
    """
    state = {
        "original_query": query,                                    # The initial query from the user
        "current_query": rewritten_query if rewritten_query else query,  # Current version of the query (may be rewritten)
        "context": context_chunks,                                 # Retrieved context chunks from the knowledge base
        "previous_responses": previous_responses if previous_responses else [],  # History of generated responses
        "previous_rewards": previous_rewards if previous_rewards else []         # History of received rewards
    }
    return state


def define_action_space() -> List[str]:
    """
    Define the set of possible actions the reinforcement learning agent can take.
    
    Actions include:
    - rewrite_query: Reformulate the original query to improve retrieval
    - expand_context: Retrieve additional context chunks
    - filter_context: Remove irrelevant context chunks
    - generate_response: Generate a response based on current query and context
    
    Returns:
        List[str]: A list of available actions.
    """

    # Define the set of actions the agent can take
    actions = ["rewrite_query", "expand_context", "filter_context", "generate_response"]
    return actions


def calculate_reward(response: str, ground_truth: str) -> float:
    """
    通过比较生成的回答与真实答案来计算奖励值-cosine_similarity 
    
    Args:
        response (str): basic-RAG pipeline 生成的回答。
        ground_truth (str)：预期的正确答案。
    
    Returns:
        float: 奖励值在 -1 到 1 之间，值越高表示与真实答案的相似度越高。
    """
    response_embedding = generate_embeddings_batch([response])[0]
    ground_truth_embedding = generate_embeddings_batch([ground_truth])[0]
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity


# Action  Functions
# ----------------------------------------------------------------------------------
def rewrite_query(
    query: str, 
    context_chunks: List[str], 
    model: str = "qwen-long", 
    max_tokens: int = 100, 
    temperature: float = 0.3
) -> str:
    # 构建提示，让 LLM 重写查询
    rewrite_prompt = f"""
    You are a query optimization assistant. Your task is to rewrite the given query to make it more effective 
    for retrieving relevant information. The query will be used for document retrieval.
    
    Original query: {query}
    
    Based on the context retrieved so far:
    {' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'}
    
    Rewrite the query to be more specific and targeted to retrieve better information.
    Rewritten query:
    """
    
    # 使用 LLM 生成重写后的查询
    response = ali_client.chat.completions.create(
        model=model,  # 指定用于生成响应的模型
        max_tokens=max_tokens,  # 响应中的最大标记数
        temperature=temperature,  # 响应多样性的采样温度
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )
    
    # 从响应中提取并返回重写后的查询
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query


def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:
    # 检索比当前可用片段更多的片段
    additional_chunks = chat_box.retrieve_relevant_chunks(query, top_k=top_k + len(current_chunks))
    
    # 过滤掉当前上下文中已有的片段
    new_chunks = []
    for chunk in additional_chunks:
        if chunk not in current_chunks:
            new_chunks.append(chunk)
    
    # 将新的唯一片段添加到当前上下文中，限制为 top_k
    expanded_context = current_chunks + new_chunks[:top_k]
    return expanded_context


# Function to filter the context to keep only the most relevant chunks
def filter_context(query: str, context_chunks: List[str]) -> List[str]:
    """
    Filter the context to keep only the most relevant chunks.

    Args:
        query (str): 询问的问题
        context_chunks (List[str]): 相似的文本块 
            最初来自rag_utils.retrieve_relevant_chunks(query_text: str, top_k: int = 5)

    Returns:
        List[str]: A filtered list of the most relevant context chunks.
    """
    if not context_chunks:
        return []
        
    # 为查询和每个片段生成embedding 
    query_embedding = generate_embeddings_batch([query])[0]
    chunk_embeddings = [generate_embeddings_batch([chunk])[0] for chunk in context_chunks]
    
    # 计算每个片段的相关性分数
    relevance_scores = []
    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk_embedding)
        relevance_scores.append(score)
    
    # 按相关性分数降序排序片段
    sorted_chunks = [x for _, x in sorted(zip(relevance_scores, context_chunks), reverse=True)]
    
    # 保留最多 5 个最相关的片段，或更少（如果不足 5 个）
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
    return filtered_chunks


# Agent 
# ----------------------------------------------------------------------------------

def policy_network(
    state: dict, 
    action_space: List[str], 
    epsilon: float = 0.2
) -> str:
    """
    Define a policy network to select an action based on the current state using an epsilon-greedy strategy.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        epsilon (float): The probability of choosing a random action for exploration. Default is 0.2.

    Returns:
        str: The selected action from the action space.
    """
    # Use epsilon-greedy strategy: random exploration vs. exploitation
    # 使用 epsilon-greedy 策略：随机探索与利用
    if np.random.random() < epsilon:
        # 探索：从动作空间中随机选择一个动作
        action = np.random.choice(action_space)
    else:
        # 利用：
        # 如果没有之前的回答，优先重写查询
        if len(state["previous_responses"]) == 0:
            action = "rewrite_query"
        # 如果有之前的回答但奖励较低，尝试扩展上下文
        elif state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
            action = "expand_context"
        # 如果上下文片段过多，尝试过滤上下文
        elif len(state["context"]) > 5:
            action = "filter_context"
        # 否则，生成回答
        else:
            action = "generate_response"
    
    return action




# Function to perform a single RL step
def rl_step(
    state: dict, 
    action_space: List[str], 
    ground_truth: str
) -> tuple[dict, str, float, str]:
    """
    Perform a single RL step: select an action, execute it, and calculate the reward.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        ground_truth (str): The expected correct answer to calculate the reward.

    Returns:
        tuple: A tuple containing:
            - state (dict): The updated state after executing the action.
            - action (str): The action selected by the policy network.
            - reward (float): The reward received for the action.
            - response (str): The response generated (if applicable).
    """
    # 使用策略网络选择一个动作
    action: str = policy_network(state, action_space)
    response: str = None  # 初始化回答为 None
    reward: float = 0  # 初始化奖励为 0

    # 执行选择的动作
    if action == "rewrite_query":
        # 重写查询以提升检索效果
        rewritten_query: str = rewrite_query(state["original_query"], state["context"])
        state["current_query"] = rewritten_query  # 更新状态中的当前查询
        # 根据重写后的查询检索新的上下文
        new_context: List[str] = chat_box.retrieve_relevant_chunks(rewritten_query)
        state["context"] = new_context  # 更新状态中的上下文

    elif action == "expand_context":
        # 通过检索额外的片段扩展上下文
        expanded_context: List[str] = expand_context(state["current_query"], state["context"])
        state["context"] = expanded_context  # 更新状态中的上下文

    elif action == "filter_context":
        # 过滤上下文以保留最相关的片段
        filtered_context: List[str] = filter_context(state["current_query"], state["context"])
        state["context"] = filtered_context  # 更新状态中的上下文

    elif action == "generate_response":
        # 使用当前查询和上下文构建提示
        prompt: str = construct_prompt(state["current_query"], state["context"])
        # 使用 LLM 生成回答
        response: str = chat_box.generate_response(prompt)
        # 根据生成的回答与真实答案之间的相似度计算奖励
        reward: float = calculate_reward(response, ground_truth)
        # 更新状态中的回答和奖励历史
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)

    # 返回更新后的状态、选择的动作、获得的奖励和生成的回答
    return state, action, reward, response


def initialize_training_params() -> Dict[str, Union[float, int]]:
    params = {
        "learning_rate": 0.01,  
        "num_episodes": 100,   
        "gamma": 0.99   
    }
    return params


def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]], 
    state: Dict[str, object], 
    action: str, 
    reward: float, 
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Update the policy based on the reward received.

    Args:
        policy (Dict[str, Dict[str, Union[float, str]]]): The current policy to be updated.
        state (Dict[str, object]): The current state of the environment.
        action (str): The action taken by the agent.
        reward (float): The reward received for the action.
        learning_rate (float): The learning rate for updating the policy.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: The updated policy.
    """
    # Example: Simple policy update (to be replaced with a proper RL algorithm)
    policy[state["query"]] = {
        "action": action,  # 存储采取的动作
        "reward": reward   # 存储获得的奖励
    }
    return policy


# Function to implement the training loop
def training_loop(
    query_text: str, 
    ground_truth: str, 
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    """
    Implement the training loop for RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.
        params (Optional[Dict[str, Union[float, int]]]): Training parameters such as learning rate, 
            number of episodes, and discount factor. If None, default parameters are initialized.

    Returns:
        Tuple: A tuple containing:
            - policy (Dict[str, Dict[str, Union[float, str]]]): The updated policy after training.
            - rewards_history (List[float]): A list of rewards received in each episode.
            - actions_history (List[List[str]]): A list of actions taken in each episode.
            - best_response (Optional[str]): The best response generated during training.
    """
     # 如果未提供训练参数，则初始化默认参数
    if params is None:
        params = initialize_training_params()
    
    # 初始化变量以跟踪进度
    rewards_history: List[float] = [] 
    actions_history: List[List[str]] = [] 
    policy: Dict[str, Dict[str, Union[float, str]]] = {} 
    action_space: List[str] = define_action_space()
    best_response: Optional[str] = None
    best_reward: float = -1
    
    # 获取简单 RAG 流水线的初始性能以供比较
    simple_response: str = chat_box.chat(query_text)
    simple_reward: float = calculate_reward(simple_response, ground_truth)
    print(f"简单 RAG 奖励：{simple_reward:.4f}")

    # 开始训练循环
    for episode in tqdm(range(params["num_episodes"])):
        # 使用相同的查询重置环境
        context_chunks: List[str] = chat_box.retrieve_relevant_chunks(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0  # 初始化当前周期的奖励
        episode_actions: List[str] = []  # 初始化当前周期的动作列表
        
        # 每个周期的最大步骤数，防止无限循环
        for step in tqdm(range(10), leave=False):
            state, action, reward, response = rl_step(state, action_space, ground_truth)
            episode_actions.append(action)  # 记录采取的动作
            # 如果生成了回答，则结束周期
            if response:
                episode_reward = reward  # 更新周期奖励
                # 跟踪最佳回答和奖励
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
                
                break
        
        # 更新奖励和动作历史记录
        rewards_history.append(episode_reward)
        actions_history.append(episode_actions)

        # Print progress every 5 episodes
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.4f}, Actions = {episode_actions}")
    
    # Compare the best RL-enhanced RAG reward with the simple RAG reward
    improvement: float = best_reward - simple_reward
    print(f"\nTraining completed:")
    print(f"Simple RAG reward: {simple_reward:.4f}")
    print(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

    return policy, rewards_history, actions_history, best_response


# Function to compare Simple RAG vs RL-Enhanced RAG
def compare_rag_approaches(query_text: str, ground_truth: str) -> Tuple[str, str, float, float]:
    """
    Compare the outputs of simple RAG versus RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.

    Returns:
        Tuple[str, str, float, float]: A tuple containing:
            - simple_response (str): The response generated by the simple RAG pipeline.
            - best_rl_response (str): The best response generated by the RL-enhanced RAG pipeline.
            - simple_similarity (float): The similarity score of the simple RAG response to the ground truth.
            - rl_similarity (float): The similarity score of the RL-enhanced RAG response to the ground truth.
    """
    print("=" * 80)
    print(f"Query: {query_text}")
    print("=" * 80)
    
    # Step 1: 生成rag答案
    simple_response: str = chat_box.chat(query_text)
    # 计算相似度
    simple_similarity: float = calculate_reward(simple_response, ground_truth)
    
    print("\nSimple RAG Output:")
    print("-" * 40)
    print(simple_response)
    print(f"Similarity to ground truth: {simple_similarity:.4f}")
    
    # Step 2: 训练 RL-enhanced RAG model
    print("\nTraining RL-enhanced RAG model...")
    params: Dict[str, float | int] = initialize_training_params()
    params["num_episodes"] = 5

    _, rewards_history, actions_history, best_rl_response = training_loop(
        query_text, ground_truth, params
    )
    
    # 最终未生成答案: 基于RL-enhanced 的query_text生成答案
    if best_rl_response is None:
        # context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        # prompt: str = construct_prompt(query_text, context_chunks)
        # best_rl_response: str = generate_response(prompt)
        best_rl_response: str = chat_box.chat(query_text)
    
    # Calculate the similarity score between the RL-enhanced RAG response and the ground truth.
    rl_similarity: float = calculate_reward(best_rl_response, ground_truth)
    
    print("\nRL-enhanced RAG Output:")
    print("-" * 40)
    print(best_rl_response)
    print(f"Similarity to ground truth: {rl_similarity:.4f}")
    
    # Step 3: Evaluate and compare the results
    # Calculate the improvement in similarity score achieved by the RL-enhanced RAG model.
    improvement: float = rl_similarity - simple_similarity
    
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Simple RAG similarity to ground truth: {simple_similarity:.4f}")
    print(f"RL-enhanced RAG similarity to ground truth: {rl_similarity:.4f}")
    print(f"Improvement: {improvement * 100:.2f}%")
    
    # Step 4: Plot the reward history (if there are enough episodes and matplotlib is available)
    if len(rewards_history) > 1:
        try:
            import matplotlib.pyplot as plt
            # Create a plot to visualize the reward history during RL training.
            plt.figure(figsize=(10, 6))
            plt.plot(rewards_history)
            plt.title('Reward History During RL Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        except ImportError:
            # If matplotlib is not available, print a message instead of plotting.
            print("Matplotlib not available for plotting rewards")
    
    # Return the results: responses and similarity scores for both approaches.
    return simple_response, best_rl_response, simple_similarity, rl_similarity


def test_cmp_rag():
    cur_p = os.path.dirname(__file__)
    test_f = os.path.join(cur_p, "data", 'val.json')
    with open(test_f, 'r') as file:
        validation_data = json.load(file)

    sample_query = validation_data['basic_factual_questions'][0]['question']  
    expected_answer = validation_data['basic_factual_questions'][0]['answer']  
    simple_response, best_rl_response, simple_similarity, rl_similarity = compare_rag_approaches(sample_query, expected_answer)
    print(f'{simple_response=}, {best_rl_response=}, {simple_similarity=}, {rl_similarity=}')



res_ = """ 
================================================================================
Query: What is the mathematical representation of a qubit in superposition?
================================================================================

Simple RAG Output:
----------------------------------------
A qubit in superposition can be mathematically represented as:

\[ \alpha|0\rangle + \beta|1\rangle \]

where:
- \( \alpha \) and \( \beta \) are complex numbers called probability amplitudes.
- \( |0\rangle \) and \( |1\rangle \) are the basis states of the qubit.
- The probabilities of measuring the qubit in state \( |0\rangle \) or \( |1\rangle \) are given by \( |\alpha|^2 \) and \( |\beta|^2 \), respectively, with the condition \( |\alpha|^2 + |\beta|^2 = 1 \).
Similarity to ground truth: 0.9147

Training RL-enhanced RAG model...
简单 RAG 奖励：0.9079
                                                                                                                                                                                    
Training completed:
Simple RAG reward: 0.9079
Best RL-enhanced RAG reward: 0.9407
Improvement: 0.0328 (3.28%)

RL-enhanced RAG Output:
----------------------------------------
The mathematical formula that represents a qubit in a superposition of states, including its probability amplitudes α and β, is:

ψ = α|0⟩ + β|1⟩ 

where α and β are complex numbers satisfying |α|² + |β|² = 1. Here, |0⟩ and |1⟩ represent the basis states of the qubit.
Similarity to ground truth: 0.9407

Evaluation Results:
----------------------------------------
Simple RAG similarity to ground truth: 0.9147
RL-enhanced RAG similarity to ground truth: 0.9407
Improvement: 2.59%
simple_response='A qubit in superposition can be mathematically represented as:\n\n\\[ \\alpha|0\\rangle + \\beta|1\\rangle \\]\n\nwhere:\n- \\( \\alpha \\) and \\( \\beta \\) are complex numbers called probability amplitudes.\n- \\( |0\\rangle \\) and \\( |1\\rangle \\) are the basis states of the qubit.\n- The probabilities of measuring the qubit in state \\( |0\\rangle \\) or \\( |1\\rangle \\) are given by \\( |\\alpha|^2 \\) and \\( |\\beta|^2 \\), respectively, with the condition \\( |\\alpha|^2 + |\\beta|^2 = 1 \\).', 
best_rl_response='The mathematical formula that represents a qubit in a superposition of states, including its probability amplitudes α and β, is:\n\nψ = α|0⟩ + β|1⟩ \n\nwhere α and β are complex numbers satisfying |α|² + |β|² = 1. Here, |0⟩ and |1⟩ represent the basis states of the qubit.', simple_similarity=0.9147188527910994, rl_similarity=0.9406503486070138
"""

if __name__ == '__main__':
    test_cmp_rag()

