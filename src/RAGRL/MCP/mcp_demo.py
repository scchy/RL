# python3
# Create Date: 2025-08-27
# Func: MCP-demo 
# ============================================================================================
import os
import asyncio
import json
from openai import OpenAI
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
CUR_DIR = os.path.dirname(__file__)
OPENAI_API_KEY = os.environ['DEEPSEEK_API_KEY']
OPENAI_API_BASE = "https://api.deepseek.com"


class MCPClientDemo:
    def __init__(self, server_path: str):
        """
        初始化MCP客户端
        :param server_path: MCP服务端脚本路径
        """
        self.server_path = server_path
        self.llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        self.sys_prompt = {"role": "system", "content": "如果不知道函数中的具体参数，不要假设。如果知道或者参数为空，请询问用户具体的内容."}

    async def run(self, user_query: str):
        """
        执行用户查询，对比使用工具和不使用工具的结果
        :param user_query: 用户问题
        :return: 对比结果字典
        """
        # 配置标准IO通信的服务端参数
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        # 建立与MCP服务端的连接
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            # 创建客户端会话
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # 获取服务端注册的所有工具信息
                tools = (await session.list_tools()).tools
                
                # 将MCP工具格式转换为OpenAI函数调用格式
                functions = []
                for tool in tools:
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            # 使用工具的输入模式或默认模式
                            "parameters": tool.inputSchema or {
                                "type": "object",
                                "properties": {
                                    "city_name": {"type": "string", "description": "城市名称"}
                                },
                                "required": ["city_name"]
                            }
                        }
                    })
                
                # -------------------------------
                # 模型调用 + MCP 工具路径
                # -------------------------------
                response_with_tool = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        self.sys_prompt, 
                        {"role": "user", "content": user_query}
                    ],
                    stream=False,
                    tools=functions,
                    tool_choice="auto"
                )
                message_with_tool = response_with_tool.choices[0].message
                result_with_tool = {
                    "model_reply": message_with_tool.content,
                    "tool_called": None,
                    "tool_result": None
                }
                
                # 如果模型决定调用工具
                tool_call = message_with_tool.tool_calls[0]
                if tool_call:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    # 通过MCP会话调用实际工具
                    tool_result = await session.call_tool(tool_name, arguments)
                    result_with_tool.update({
                        "tool_called": tool_name,
                        "tool_arguments": arguments,
                        "tool_result": tool_result
                    })

                
                # -------------------------------
                # 模型不使用 MCP 工具的路径
                # -------------------------------
                # 调用相同模型，但不提供工具信息
                response_no_tool = self.llm.chat.completions.create(
                    model="deepseek-chat",
                    stream=False,
                    messages=[self.sys_prompt, {"role": "user", "content": user_query}],
                )
                message_no_tool = response_no_tool.choices[0].message
                result_no_tool = {
                    "model_reply": message_no_tool.content
                }
                
                # 返回两种调用方式的对比结果
                return {
                    "user_query": user_query,
                    "with_mcp_tool": result_with_tool,
                    "without_tool": result_no_tool
                }

async def main():
    """主函数，演示工具使用与不使用的对比"""
    # 创建MCP客户端，连接到指定服务端
    client = MCPClientDemo(server_path=os.path.join(CUR_DIR, "simple_stdio_mcp.py"))
    # 执行天气查询示例
    result = await client.run("北京的天气怎么样")
   
    # 格式化输出对比结果
    print(">>> 用户提问：", result["user_query"])
    print("\n【使用 MCP 工具】")
    print("模型回复：", result["with_mcp_tool"]["model_reply"])
    if result["with_mcp_tool"]["tool_called"]:
        print("调用工具：", result["with_mcp_tool"]["tool_called"])
        print("工具参数：", result["with_mcp_tool"]["tool_arguments"])
        print("工具结果：", result["with_mcp_tool"]["tool_result"])
    else:
        print("未调用任何工具")
    print("\n【不使用工具】")
    print("模型回复：", result["without_tool"]["model_reply"])


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
