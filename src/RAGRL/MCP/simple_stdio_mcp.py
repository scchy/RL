# python3
# Create Date: 2025-08-27
# Func: MCP-server
#       - Stdio 模式： 这个主要是用来连接你本地电脑上的软件或文件。比如你想让 AI 控制 Blender 这种没有在线服务的软件，就得用 Stdio，它的配置相对复杂一些。
#       - SSE 模式： 这个用来连接线上的、本身就有 API 的服务。比如访问你的谷歌邮箱、谷歌日历等等。SSE 的配置超级简单，基本上就是一个链接搞定。
# ============================================================================================
from fastmcp import FastMCP


app = FastMCP("demo")


@app.tool(name="weather", description="城市天气查询")
def get_weather(city: str):
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"}
    }
    return weather_data.get(city, {"error": "未找到该城市"})


if __name__ == "__main__":
    app.run(transport="stdio")


# if __name__ == "__main__":
#     # 启动HTTP服务，支持流式响应
#     app.run(
#         transport="streamable-http",  # 使用支持流式传输的HTTP协议
#         host="127.0.0.1",            # 监听本地地址
#         port=4200,                   # 服务端口
#         path="/demo",                # 服务路径前缀
#         log_level="debug",           # 调试日志级别
#     )
