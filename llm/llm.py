"""
LLM 谣言检测结果解释模块
─────────────────────────
接收二分类模型预测结果（文本 + 0/1 标签 + 证据词），
调用大语言模型生成自然语言解释。

使用方式:
    from llm.llm import explain

    reason = explain(text="推文内容...", label=1, evidence=["foo", "bar"])
    print(reason)  # "该推文被判定为谣言，因为..."

支持的后端:
    - 任何兼容 OpenAI chat/completions 接口的服务

环境变量:
    LLM_API_KEY       API 密钥 (可选，已内置默认值)
    LLM_BASE_URL      接口地址 (可选，默认 https://models.sjtu.edu.cn/api/v1)
    LLM_MODEL         模型名称 (可选，默认 deepseek-reasoner)
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LLMConfig:
    """LLM API 配置，支持从环境变量覆盖"""

    api_key: str = "sk-qTuQYYdsMw21GBbJHbQfZA"
    base_url: str = "https://models.sjtu.edu.cn/api/v1"
    model: str = "deepseek-reasoner"
    temperature: float = 0.3
    max_tokens: int = 512
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量读取配置，不存在则使用默认值"""
        api_key = os.environ.get(
            "LLM_API_KEY", "sk-qTuQYYdsMw21GBbJHbQfZA"
        )
        if not api_key:
            raise ValueError("请设置环境变量 LLM_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "LLM_BASE_URL", "https://models.sjtu.edu.cn/api/v1"
            ),
            model=os.environ.get("LLM_MODEL", "deepseek-reasoner"),
        )


# ═══════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个社交媒体（Twitter/X）谣言检测系统的专业解释者。\n"
    "\n"
    "一个二分类模型已经完成了对推文的判断（1=谣言，0=非谣言），"
    "你的任务是为该判断结果生成一段清晰、简洁的中文解释。\n"
    "\n"
    "解释原则：\n"
    "- 如果判定为谣言（label=1）：说明该推文为什么像谣言 —— "
    "例如：包含未经证实的说法、使用煽动性语言、缺乏可信来源、"
    "情绪操控、与已知事实不符等。\n"
    "- 如果判定为非谣言（label=0）：说明该推文为什么看起来可信 —— "
    "例如：报道客观事实、引用了官方或可查证来源、语气中立、信息可验证等。\n"
    "\n"
    "要求：\n"
    "- 解释长度为 2-4 句中文。\n"
    "- 必须针对推文的具体内容进行分析，不要泛泛而谈。\n"
    "- 不要使用“模型认为”“根据预测结果”等字眼，"
    "直接以分析的口吻描述。"
)


# ═══════════════════════════════════════════════════════════════════════
# LLM 解释器
# ═══════════════════════════════════════════════════════════════════════


class RumorExplainerLLM:
    """基于 LLM 的谣言检测结果解释器

    输入: 推文文本 + 已有的二分类标签 (0/1) + 模型提取的证据词
    输出: 自然语言解释文本

    用法:
        explainer = RumorExplainerLLM()
        reason = explainer.explain("推文内容", label=1, evidence=["foo"])
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        # base_url 可能已包含 /chat/completions 路径，也可能只到 /v1
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self._chat_url = base
        else:
            self._chat_url = f"{base}/chat/completions"

    # ── 核心接口 ──────────────────────────────────────────────

    def explain(
        self,
        text: str,
        label: int,
        evidence: Optional[List[str]] = None,
        context: str = "",
    ) -> str:
        """对模型分类结果生成自然语言解释

        Args:
            text: 推文文本
            label: 模型预测标签，0=非谣言, 1=谣言
            evidence: 模型注意力机制提取的关键证据词列表
            context: (可选) RAG 检索到的补充背景知识

        Returns:
            str: 自然语言解释 (2-4 句中文)
        """
        if not text or not text.strip():
            return "输入文本为空，无法生成解释。"

        label_name = "谣言" if label else "非谣言"

        # 构造 user prompt
        user_content = (
            f"分类结果: {label_name} (label={label})\n"
            f"\n"
            f"推文内容: \"{text}\"\n"
        )

        # 附上模型提取的证据词，帮助 LLM 聚焦关键信息
        if evidence:
            evidence_str = ", ".join(evidence)
            user_content += (
                f"\n"
                f"模型关注的证据词: {evidence_str}\n"
            )

        if context.strip():
            user_content += (
                f"\n"
                f"补充背景知识（检索到的相关信息）:\n"
                f"{context}\n"
            )

        user_content += "\n请为该分类结果撰写一段中文分析解释。"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return self._call_api(messages)

    def explain_batch(
        self,
        items: List[Dict],
    ) -> List[str]:
        """批量解释（逐条调用，不并发）

        Args:
            items: 字典列表，每项包含:
                {"text": str, "label": int,
                 "evidence": Optional[List[str]], "context": Optional[str]}

        Returns:
            List[str]: 每条对应的解释文本
        """
        results = []
        for item in items:
            reason = self.explain(
                text=item["text"],
                label=item["label"],
                evidence=item.get("evidence"),
                context=item.get("context", ""),
            )
            results.append(reason)
        return results

    # ── 底层 API 调用 ────────────────────────────────────────

    def _call_api(self, messages: list) -> str:
        """调用兼容 OpenAI chat/completions 接口的 LLM 服务"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            resp = requests.post(
                self._chat_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"LLM API 超时 ({self.config.timeout}s)"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"LLM API 错误 {e.response.status_code}: "
                f"{e.response.text[:300]}"
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"无法连接 {self._chat_url}，请检查 LLM_BASE_URL"
            )


# ═══════════════════════════════════════════════════════════════════════
# 模块级快捷接口 —— 供 main.py 直接调用
# ═══════════════════════════════════════════════════════════════════════

# 全局单例，避免每次调用都重新创建配置
_explainer: Optional[RumorExplainerLLM] = None


def _get_explainer() -> RumorExplainerLLM:
    """懒加载全局解释器实例"""
    global _explainer
    if _explainer is None:
        _explainer = RumorExplainerLLM()
    return _explainer


def explain(
    text: str,
    label: int,
    evidence: Optional[List[str]] = None,
) -> str | None:
    """对谣言检测结果生成 LLM 解释 —— main.py 调用入口

    签名与原有接口完全兼容：label=0 为非谣言，label=1 为谣言。
    evidence 来自模型 attention 提取的 top-5 关键词。

    如果 LLM API 不可用，返回 None 而非抛出异常，
    保证主流程在无网络或 API 故障时仍能正常运行。
    """
    try:
        return _get_explainer().explain(
            text=text, label=label, evidence=evidence
        )
    except Exception:
        # API 不可用时静默降级，main.py 会收到 None
        return None
