import litellm
import os
import json

def test_sglang_completion_with_litellm():
    """
    测试使用 litellm.completion 调用 SGLang 接口，并传递 extra_body 参数。
    """

    messages = [
        {"role": "user", "content": "Hello, how are you today?"}
    ]

    model_name = "/data/models/qwen3_4b"

    # 要传递给 SGLang 的额外参数
    extra_body_params = {
        "temperature": 0.7,
        "max_tokens": 5000,
        "top_p": 0.9,
        "top_k": 3,
        "min_p": 0.3,
    }

    sglang_base_url = "http://172.168.94.88:30000/v1"

    sglang_api_key = "EMPTY"

    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            api_key=sglang_api_key,
            base_url=sglang_base_url,
            extra_body=extra_body_params,
            custom_llm_provider="openrouter" # 明确指定为 OpenAI 兼容提供商
        )

        print("\n--- API 调用成功 ---")
        print("响应类型:", type(response))
        print("完整响应:")
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))

        # 简单检查响应内容
        if response and response.choices:
            print("\n--- 响应内容检查 ---")
            print("模型:", response.model)
            print("第一个助手的回复:", response.choices[0].message.content)
            # 这里无法直接验证 extra_body 中的自定义参数是否被 SGLang 服务器接收并处理
            # 因为 litellm 返回的是标准 ModelResponse 对象，不会包含 extra_body 的回显
            # 但如果 API 调用成功，且 SGLang 行为正常，则说明参数已传递。
        else:
            print("\n--- 警告：响应中没有 choices 或响应为空。---")

    except Exception as e:
        print(f"\n--- API 调用失败 ---")
        print(f"发生错误: {e}")
        print("请检查：")
        print("1. SGLANG_API_KEY 环境变量是否正确设置。")
        print("2. SGLANG_BASE_URL 环境变量是否正确（通常是 https://api.sglang.com/v1）。")
        print("3. SGLang 服务是否正在运行且可访问。")
        print("4. 您使用的模型名称是否在 SGLang 服务中有效。")

if __name__ == "__main__":
    litellm.set_verbose=True
    test_sglang_completion_with_litellm()