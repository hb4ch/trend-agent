"""
Unified LLM Provider for Trend Agent

Supports multiple LLM backends through langchain:
- ZhipuAI: Used for web search (zhipu_search tool)
- SiliconFlow DeepSeek V3.2: Default for all other tasks
"""

import os
import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

logger = logging.getLogger(__name__)

# Model configurations
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4-flash")
ZHIPU_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", os.environ.get("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1"))
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", os.environ.get("DEEPSEEK_API_KEY"))
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")

QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")


class LLMProvider:
    """
    Unified LLM provider supporting multiple backends.

    Usage:
        provider = LLMProvider()

        # Get DeepSeek for general tasks
        llm = provider.get_llm(model="deepseek")

        # Get Zhipu for search tasks
        llm = provider.get_llm(model="zhipu")

        # Simple invoke
        response = provider.invoke("deepseek", "Hello, world!")

        # With prompt template
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        chain = prompt | provider.get_llm("deepseek") | StrOutputParser()
        result = chain.invoke({"topic": "programming"})
    """

    def __init__(self):
        self._llm_cache: Dict[str, BaseChatModel] = {}
        logger.info("LLM Provider initialized")

    def get_llm(self, model: str = "deepseek", temperature: float = 0.2) -> BaseChatModel:
        """
        Get a langchain LLM instance for the specified model.

        Args:
            model: Model name ("deepseek", "zhipu", or "qwen")
            temperature: Sampling temperature

        Returns:
            langchain BaseChatModel instance
        """
        cache_key = f"{model}_{temperature}"

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        llm = self._create_llm(model, temperature)
        self._llm_cache[cache_key] = llm
        return llm

    def _create_llm(self, model: str, temperature: float) -> BaseChatModel:
        """Create a new LLM instance."""

        if model == "zhipu":
            if not ZHIPU_API_KEY:
                raise ValueError("ZHIPUAI_API_KEY not set")
            return ChatZhipuAI(
                model=ZHIPU_MODEL,
                temperature=temperature,
                api_key=ZHIPU_API_KEY,
            )

        elif model == "deepseek":
            if not SILICONFLOW_API_KEY:
                raise ValueError("SILICONFLOW_API_KEY not set")
            # Use ChatOpenAI with SiliconFlow's OpenAI-compatible API
            return ChatOpenAI(
                model=DEEPSEEK_MODEL,
                base_url=SILICONFLOW_BASE_URL,
                api_key=SILICONFLOW_API_KEY,
                temperature=temperature,
                timeout=300,  # 5 minutes timeout for long requests
            )

        elif model == "qwen":
            if not SILICONFLOW_API_KEY:
                raise ValueError("SILICONFLOW_API_KEY not set")
            return ChatOpenAI(
                model=QWEN_MODEL,
                base_url=SILICONFLOW_BASE_URL,
                api_key=SILICONFLOW_API_KEY,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unknown model: {model}. Choose from: deepseek, zhipu, qwen")

    def invoke(
        self,
        model: str,
        messages: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """
        Simple invoke interface.

        Args:
            model: Model name ("deepseek", "zhipu", or "qwen")
            messages: List of message strings (user messages)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to the LLM

        Returns:
            Response text
        """
        llm = self.get_llm(model, temperature)

        # Convert messages to langchain format
        langchain_messages = []
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            langchain_messages.append(HumanMessage(content=msg))

        response = llm.invoke(langchain_messages, **kwargs)
        return response.content

    def invoke_with_prompt(
        self,
        model: str,
        prompt_template: ChatPromptTemplate,
        variables: Dict[str, Any],
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """
        Invoke with a ChatPromptTemplate.

        Args:
            model: Model name
            prompt_template: ChatPromptTemplate instance
            variables: Variables for the template
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Response text
        """
        llm = self.get_llm(model, temperature)
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(variables, **kwargs)


# Global instance
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get the global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider


def get_llm(model: str = "deepseek", temperature: float = 0.2) -> BaseChatModel:
    """
    Convenience function to get an LLM instance.

    Args:
        model: Model name ("deepseek", "zhipu", or "qwen")
        temperature: Sampling temperature

    Returns:
        langchain BaseChatModel instance
    """
    return get_llm_provider().get_llm(model, temperature)


def invoke_llm(
    model: str,
    messages: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """
    Convenience function to invoke an LLM.

    Args:
        model: Model name ("deepseek", "zhipu", or "qwen")
        messages: List of message strings
        system_prompt: Optional system prompt
        temperature: Sampling temperature

    Returns:
        Response text
    """
    return get_llm_provider().invoke(model, messages, system_prompt, temperature)
