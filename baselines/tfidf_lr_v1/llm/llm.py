from dataclasses import dataclass, field
import os
import json
import requests

@dataclass
class LLMConfig:
    "LLM 配置"

    api_key: str = ""  # 请设置环境变量 LLM_API_KEY 或在此处填入
    base_url: str = "https://models.sjtu.edu.cn/api/v1"
    model: str = "deepseek-reasoner"
    temperature: float = 0.3
    max_tokens: int = 512
    timeout: int = 30

def explain(text: str, label: int, evidence: list, config: LLMConfig = None) -> str:
    """
    根据模型预测出的谣言/非谣言判定结果、可解释性核心证据词，
    调用交大官方 API 的 deepseek-reasoner 模型，生成深度的、条理清晰的谣言可解释性分析。
    """
    if config is None:
        config = LLMConfig()
    if not config.api_key:
        config.api_key = os.environ.get("LLM_API_KEY", "")

    label_str = "【谣言 (Rumor)】" if label == 1 else "【非谣言 (Non-Rumor)】"
    
    # 格式化特征词作为 Prompt 的一部分
    evidence_desc = ""
    if not evidence:
        evidence_desc = "无明显特征词激活。"
    else:
        for idx, item in enumerate(evidence):
            evidence_desc += f"- 词语: {item['word']} | TF-IDF: {item['tfidf']:.4f} | 决策权重: {item['weight']:.4f} | 乘积贡献: {item['contribution']:.4f}\n"

    # 设计高质量的学术化中文可解释性 Prompt
    system_prompt = (
        "您是一个精通社会网络、传播学和计算语言学的资深事实核查专家与AI可解释性（XAI）分析助手。\n"
        "请结合传统的逻辑回归模型特征权重以及大语言模型的深层认知能力，深入解析输入推文是否为谣言，"
        "剖析其语言风格、词汇倾向和情感物理流动，生成具有学术论证力度、结构高度清晰的研究报告级解析。"
    )
    
    user_prompt = f"""请针对以下推文进行『可解释谣言检测』研判深度分析：

【输入推文文本】: "{text}"
【经典机器学习逻辑回归预测】: {label_str}
【模型作出的决策核心词证据 (Evidence Words)】:
{evidence_desc}

【分析要求】：
1. 【决策一致性核验】：深入分析模型提出的这几个关键证据词，解读为什么它们在 TF-IDF + 逻辑回归决策中贡献了最大的偏向力。（例如为什么特定词体现了谣言的煽动性、恐慌感，或者真实推文的官方性与陈述性风格）。
2. 【推文特征剖析】：从句法特征、文本修辞、情感煽动倾向等方面深入多层次剖析该推文。
3. 【深度事实核查路径】：提出如果要在实际应用中对本推文进行绝对可靠的核实，应该查证哪些客观信源（如政府公告、权威科教机构或时间序列对比）。
4. 大模型的推理过程应该与分类器结果保持逻辑闭环。请输出一个结构清晰、富有人文洞察与计算机解释学深度的中文分析。
"""

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens
    }
    
    # 如果接口在某地区网络不稳定或 API 存在限流，提供优雅降级或捕获提示
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=config.timeout
        )
        
        if response.status_code == 200:
            resp_data = response.json()
            choices = resp_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                
                # 特别支持 DeepSeek Reasoner 深层思考（Reasoning）内容的输出
                thinking = message.get("reasoning_content", "")
                content = message.get("content", "")
                
                output = ""
                if thinking:
                    output += f"【DeepSeek-Reasoner 思考链】:\n{thinking}\n\n"
                output += content
                return output
            else:
                return "API 返回结构异常，未检出有效的生成文本。"
        else:
            return f"API 请求失败（HTTP {response.status_code}）: {response.text}\n温馨提示: 本模块需要上海交通大学内网权限、校园网环境或其提供的有效跨域API才能支持网络连通。"
            
    except requests.exceptions.RequestException as e:
        return f"接口连接异常: {str(e)}\n\n(提示: 本模块对接的是上海交大校内 DeepSeek 服务接口 '{url}'。请确保您的执行终端具备至该域名的网络路由接入)"

if __name__ == "__main__":
    # 测试样例
    sample_evidence = [
        {"word": "leak", "tfidf": 0.45, "weight": 2.1, "contribution": 0.945},
        {"word": "arrest", "tfidf": 0.38, "weight": 1.8, "contribution": 0.684}
    ]
    print("[LLM Test] 正在测试交大 DeepSeek 接口连接...")
    # 由于可能在沙箱无连通权限，建议捕获其提示
    explanation = explain(
        text="BREAKING NEWS: Massive government leak confirms sudden arrest!",
        label=1,
        evidence=sample_evidence
    )
    print("\n[DeepSeek 专家核查分析生成]：\n")
    print(explanation)