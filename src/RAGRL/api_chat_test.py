# python3
# Create Date: 2025-04-08
# Author: Scc_hy
# Func: 测试deepseek api
# ================================================================================================

import os 
from openai import OpenAI
import numpy as np 
import json
# from FlagEmbedding import FlagModel
from modelscope.hub.snapshot_download import snapshot_download


help_functions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如北京"
                    }
                },
                "required": ["location"]
            }
        },
     },
]

def get_current_weather(location):
    if location == '东京':
        return'misty'
    return'sunny'


def test_function_call():
    api_key = os.environ['DEEPSEEK_API_KEY']
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    messages = [
        {"role": "system", "content": "如果不知道函数中的具体参数，不要假设。如果知道或者参数为空，请询问用户具体的内容."},
        {"role": "user", "content": "东京天气怎么样？"}
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    print('ORG-Response:', response.choices[0].message.content)
    # 第一次 → 模型“做决策，下指令”。
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        tools=help_functions
    )
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_current_weather":
        args = json.loads(tool_call.function.arguments)  # 注意：生产环境请用 json.loads
        result = get_current_weather(**args)

        # 把结果返回给模型继续对话
        # 第二次 → 模型“拿到执行结果，生成自然语言”。
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        print('\n--------------------------\nFunctionCall-Response:',final_response.choices[0].message.content)


def ds_test():
    api_key = os.environ['DEEPSEEK_API_KEY']

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)


def aiml_test():
    chunks_batch = ['Hello, how are you ?', 'I am fine. thank you.']
    aiml_api_key = os.environ['AIML_API_KEY']
    client = OpenAI(api_key=aiml_api_key, base_url="https://api.aimlapi.com/v1")
    response1 = client.embeddings.create(
        model='BAAI/bge-large-en-v1.5', #'BAAI/bge-en-icl',    
        input=chunks_batch[0]   
    )
    response2 = client.embeddings.create(
        model='BAAI/bge-large-en-v1.5', #'BAAI/bge-en-icl', 
        input=chunks_batch[1]   
    )
    a = np.array([item.embedding for item in response1.data])
    b = np.array([item.embedding for item in response2.data])
    print(f'{a.shape=}')
    s = a @ b.T / (np.linalg.norm(a) * np.linalg.norm(b))
    print(chunks_batch, f'{s=:.5f}')



def deepinfra_test():
    chunks_batch = ['Hello, how are you ?', 'I am fine. thank you.']
    aiml_api_key = os.environ['AIML_API_KEY']
    df_api_key = os.environ['DEEPINFRA_API_KEY']
    client = OpenAI(api_key=df_api_key, base_url="https://api.deepinfra.com/v1/openai")
    response = client.embeddings.create(
        model='BAAI/bge-en-icl',    
        input=chunks_batch,
        encoding_format='float'
    )
    rb = np.array([item.embedding for item in response.data])
    a = rb[0, :]
    b = rb[1, :]
    print(f'{rb.shape=}')
    s = a @ b.T / (np.linalg.norm(a) * np.linalg.norm(b))
    print(chunks_batch, f'{s=:.5f}')


def BAAI_download():
    local_model_dir = '/home/scc/sccWork/devData/sccDisk/local_models'
    model_name = 'BAAI/bge-en-icl'
    snapshot_download(model_id=model_name, cache_dir=local_model_dir)


def test_FlagModel():
    local_model_dir = '/home/scc/sccWork/devData/sccDisk/local_models'
    model_name = 'BAAI/bge-en-icl'
    # 初始化模型
    model = FlagModel(
        os.path.join(local_model_dir, model_name), # 'BAAI/bge-large-zh-v1.5', 
        use_fp16=True
    )

    a = model.encode('Hello, how are you ?')
    b = model.encode('I am fine. thank you.')

    print(a @ b.T / (np.linalg.norm(a) * np.linalg.norm(b)))


def ali_chat_test():
    ali_api_key = os.environ['ALI_API_KEY']
    client = OpenAI(api_key=ali_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    completion = client.chat.completions.create(
        model='qwen-long', #"qwen-turbo", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        max_tokens=512,   
        temperature=1,   
        top_p=0.9,   
        extra_body={   
            "top_k": 50  
        },
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": "你是谁？"}],
    )
    print('qwen-long: ', completion.choices[0].message.content)

    # text-embedding-v3 或 text-embedding-async-v2
    response = client.embeddings.create(
        model='text-embedding-v3',
        input=['你是谁'],
        encoding_format='float'
    )
    embeddings = [item.embedding for item in response.data]
    print('text-embedding-v3', len(embeddings[0]))


if __name__ == '__main__':
    # ds_test()
    # aiml_test()
    # deepinfra_test()
    # ali_chat_test()
    test_function_call()
