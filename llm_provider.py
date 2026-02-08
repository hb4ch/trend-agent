"""
Unified LLM Provider for Trend Agent

Supports multiple LLM backends through langchain:
- ZhipuAI: Used for web search (zhipu_search tool)
- SiliconFlow DeepSeek V3.2: Default for all other tasks
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
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

# =============================================================================
# LLM Logging Configuration
# =============================================================================

# Force LLM input/output to be printed to screen (parsed, not raw JSON)
# Set via environment variable: FORCE_LLM_LOGGING=1
FORCE_LLM_LOGGING = os.environ.get("FORCE_LLM_LOGGING", "").strip() in {"1", "true", "True", "YES", "yes"}

# =============================================================================
# LLM Message Formatting Helpers
# =============================================================================

def format_messages_for_screen(messages: List[Dict[str, str]], model: str, max_length: int = 2000) -> str:
    """
    Format messages for screen display in a readable format.

    Args:
        messages: List of message dicts with "role" and "content" keys
        model: Model name being called
        max_length: Maximum content length to display (truncate if longer)

    Returns:
        Formatted string for screen output
    """
    lines = [
        "",
        "=" * 80,
        f"[{datetime.now().strftime('%H:%M:%S')}] LLM CALL: {model}",
        "=" * 80,
    ]

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        content_type = msg.get("type", "text")  # For tool results etc.

        # Truncate long content for readability
        if len(content) > max_length:
            content_preview = content[:max_length] + f"... [truncated, total {len(content)} chars]"
        else:
            content_preview = content

        lines.append(f"\n--- Message {i+1}: {role.upper()} ---")

        # Handle different content types
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            lines.append(f"Tool Call ID: {tool_call_id}")
            if content_type:
                lines.append(f"Content Type: {content_type}")
            lines.append(f"Result: {content_preview}")
        elif role == "assistant":
            # Check for tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                lines.append(f"Tool Calls: {len(tool_calls)} tool(s)")
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_func = tc.get("function", {})
                        lines.append(f"  - {tc_func.get('name', 'unknown')}(...)")
                    else:
                        lines.append(f"  - {tc}")
            if content:
                lines.append(f"Content: {content_preview}")
            # Check for reasoning_content (DeepSeek thinking mode)
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                reasoning_preview = reasoning[:max_length] + "..." if len(reasoning) > max_length else reasoning
                lines.append(f"Reasoning: {reasoning_preview}")
        else:
            # system, user messages
            lines.append(content_preview)

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_response_for_screen(response: str, model: str, max_length: int = 3000) -> str:
    """
    Format LLM response for screen display in a readable format.

    Args:
        response: The response content
        model: Model name that was called
        max_length: Maximum response length to display (truncate if longer)

    Returns:
        Formatted string for screen output
    """
    if len(response) > max_length:
        response_preview = response[:max_length] + f"\n... [truncated, total {len(response)} chars]"
    else:
        response_preview = response

    lines = [
        "",
        "=" * 80,
        f"[{datetime.now().strftime('%H:%M:%S')}] LLM RESPONSE: {model}",
        "=" * 80,
        response_preview,
        "=" * 80,
    ]
    return "\n".join(lines)


def format_thinking_response_for_screen(result: Dict[str, Any], model: str) -> str:
    """
    Format DeepSeek thinking mode response for screen display.

    Args:
        result: Result dict from invoke_thinking
        model: Model name that was called

    Returns:
        Formatted string for screen output
    """
    content = result.get("content", "")
    reasoning = result.get("reasoning_content", "")
    tool_calls = result.get("tool_calls")
    usage = result.get("usage")

    lines = [
        "",
        "=" * 80,
        f"[{datetime.now().strftime('%H:%M:%S')}] LLM RESPONSE (Thinking Mode): {model}",
        "=" * 80,
    ]

    # Add reasoning content if present
    if reasoning:
        reasoning_preview = reasoning[:2000] + "\n... [truncated]" if len(reasoning) > 2000 else reasoning
        lines.extend([
            "🧠 REASONING PROCESS:",
            "-" * 40,
            reasoning_preview,
            "-" * 40,
        ])

    # Add tool calls if present
    if tool_calls:
        lines.extend([
            "",
            f"🔧 TOOL CALLS: {len(tool_calls)} tool(s)",
        ])
        for tc in tool_calls:
            if hasattr(tc, 'function'):
                lines.append(f"  - {tc.function.name}(...)")
            elif isinstance(tc, dict):
                tc_func = tc.get("function", {})
                lines.append(f"  - {tc_func.get('name', 'unknown')}(...)")

    # Add final content
    content_preview = content[:2000] + "\n... [truncated]" if len(content) > 2000 else content
    lines.extend([
        "",
        "📝 FINAL ANSWER:",
        "-" * 40,
        content_preview,
    ])

    # Add usage info if available
    if usage:
        lines.extend([
            "-" * 40,
            f"Tokens: prompt={getattr(usage, 'prompt_tokens', '?')} + "
            f"completion={getattr(usage, 'completion_tokens', '?')} = "
            f"total={getattr(usage, 'total_tokens', '?')}"
        ])

    lines.append("=" * 80)
    return "\n".join(lines)


def log_to_screen(message: str) -> None:
    """Print message directly to screen (bypasses logging system)."""
    print(message, flush=True)


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
            # Enable thinking mode for DeepSeek V3.2 by using max_completion_tokens
            return ChatOpenAI(
                model=DEEPSEEK_MODEL,
                base_url=SILICONFLOW_BASE_URL,
                api_key=SILICONFLOW_API_KEY,
                temperature=temperature,
                timeout=300,  # 5 minutes timeout for long requests
                max_completion_tokens=8192,  # Required for thinking mode
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
        # Build message dict for logging
        log_messages = []
        if system_prompt:
            log_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            log_messages.append({"role": "user", "content": msg})

        # Log input if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_messages_for_screen(log_messages, model))

        llm = self.get_llm(model, temperature)

        # Convert messages to langchain format
        langchain_messages = []
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            langchain_messages.append(HumanMessage(content=msg))

        response = llm.invoke(langchain_messages, **kwargs)
        response_content = response.content

        # Log output if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_response_for_screen(response_content, model))

        return response_content

    def invoke_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """
        Invoke with message dicts in OpenAI format.

        Args:
            model: Model name ("deepseek", "zhipu", or "qwen")
            messages: List of message dicts with "role" and "content" keys
                     e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to the LLM

        Returns:
            Response text
        """
        # Log input if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_messages_for_screen(messages, model))

        llm = self.get_llm(model, temperature)

        # Convert message dicts to langchain format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                langchain_messages.append(AIMessage(content=content))
            else:
                # Default to human message
                langchain_messages.append(HumanMessage(content=content))

        response = llm.invoke(langchain_messages, **kwargs)
        response_content = response.content

        # Log output if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_response_for_screen(response_content, model))

        return response_content

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


def invoke_llm_messages(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """
    Convenience function to invoke an LLM with message dicts.

    Args:
        model: Model name ("deepseek", "zhipu", or "qwen")
        messages: List of message dicts with "role" and "content" keys
        temperature: Sampling temperature

    Returns:
        Response text
    """
    return get_llm_provider().invoke_messages(model, messages, temperature)


# =============================================================================
# DeepSeek Thinking Mode - Direct OpenAI SDK Wrapper
# =============================================================================

from openai import OpenAI as OpenAIClient
import json
from typing import Literal, Optional, Union, Callable

# DeepSeek thinking mode models
DEEPSEEK_REASONING_MODEL = os.environ.get("DEEPSEEK_REASONING_MODEL", DEEPSEEK_MODEL)  # Default to configured DeepSeek model
DEEPSEEK_THINKING_BUDGET = int(os.environ.get("DEEPSEEK_THINKING_BUDGET", "4096"))  # Default 4K tokens for reasoning

# Special models that support interleaved thinking with tools
DEEPSEEK_V3_THINKING_MODEL = os.environ.get("DEEPSEEK_V3_THINKING_MODEL", "deepseek-chat")  # For V3.2 with thinking enabled


class DeepSeekThinkingClient:
    """
    Direct OpenAI SDK wrapper for DeepSeek thinking mode.

    This properly handles:
    - reasoning_content in responses
    - Multi-turn conversations with thinking mode
    - Interleaved thinking with tool calls
    - Clearing reasoning_content between user turns

    Based on: https://api-docs.deepseek.com/guides/thinking_mode
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        reasoning_model: str = None,
        default_thinking_budget: int = DEEPSEEK_THINKING_BUDGET,
    ):
        """
        Initialize DeepSeek thinking mode client.

        Args:
            api_key: SiliconFlow API key
            base_url: API base URL
            reasoning_model: Model name for thinking mode (e.g., "Pro/deepseek-ai/DeepSeek-R1")
            default_thinking_budget: Default token budget for reasoning
        """
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url
        self.reasoning_model = reasoning_model or DEEPSEEK_REASONING_MODEL
        self.default_thinking_budget = default_thinking_budget

        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY not set")

        self._client = OpenAIClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def invoke_thinking(
        self,
        messages: List[Dict[str, str]],
        thinking_budget: Optional[int] = None,
        max_tokens: int = 8192,
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Union[Literal["auto", "none", "required"], Literal["auto"]] = "auto",
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke DeepSeek in thinking mode with proper reasoning_content handling.

        Args:
            messages: List of message dicts with "role" and "content" keys
            thinking_budget: Maximum tokens for reasoning (None = use default)
            max_tokens: Maximum tokens for final answer
            model: Override model (None = use default reasoning_model)
            tools: List of tool definitions for function calling
            tool_choice: How to use tools ("auto", "none", "required")
            stream: Whether to use streaming mode

        Returns:
            Dict with:
                - content: Final answer
                - reasoning_content: Chain of thought
                - tool_calls: Tool call objects if any
                - usage: Token usage info
                - raw_response: Full response object
        """
        thinking_budget = thinking_budget or self.default_thinking_budget
        model = model or self.reasoning_model

        # Log input if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_messages_for_screen(messages, f"{model} (thinking)"))

        # Prepare request with thinking mode enabled
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            # Enable thinking mode via extra_body
            "extra_body": {
                "thinking_budget": thinking_budget
            },
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"DeepSeek thinking mode error: {e}")
            return {
                "content": f"Error invoking DeepSeek thinking mode: {e}",
                "reasoning_content": "",
                "tool_calls": None,
                "usage": None,
                "raw_response": None,
                "error": str(e),
            }

        # Handle streaming response
        if stream:
            content = ""
            reasoning_content = ""
            tool_calls = None
            usage = None

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    # Accumulate content
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content
                    # Accumulate reasoning_content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                    # Capture tool_calls (from the last chunk)
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        tool_calls = delta.tool_calls
                # Capture usage from the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage

            result = {
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls,
                "usage": usage,
                "raw_response": response,
            }

            logger.debug(f"DeepSeek thinking response (streaming) - "
                        f"content_length={len(result['content'])}, "
                        f"reasoning_length={len(result['reasoning_content'])}")

            # Log output if forced logging is enabled
            if FORCE_LLM_LOGGING:
                log_to_screen(format_thinking_response_for_screen(result, f"{model} (thinking, streaming)"))

            return result

        # Extract response components (non-streaming)
        message = response.choices[0].message
        result = {
            "content": message.content or "",
            "reasoning_content": getattr(message, "reasoning_content", "") or "",
            "tool_calls": message.tool_calls,
            "usage": response.usage,
            "raw_response": response,
        }

        logger.debug(f"DeepSeek thinking response - reasoning_tokens={getattr(response.usage, 'completion_tokens', 0)}, "
                    f"content_length={len(result['content'])}, "
                    f"reasoning_length={len(result['reasoning_content'])}")

        # Log output if forced logging is enabled
        if FORCE_LLM_LOGGING:
            log_to_screen(format_thinking_response_for_screen(result, f"{model} (thinking)"))

        return result

    def clear_reasoning_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Remove reasoning_content from all messages before sending a new user question.

        This is important in multi-turn conversations - the previous reasoning_content
        should NOT be sent back to the API for a new user question.

        Args:
            messages: List of message dicts

        Returns:
            Cleaned messages without reasoning_content
        """
        cleaned = []
        for msg in messages:
            # Only keep the fields we need
            if isinstance(msg, dict):
                cleaned_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content", "")
                }
                # Only add tool_calls if present (for tool responses)
                if "tool_calls" in msg:
                    cleaned_msg["tool_calls"] = msg["tool_calls"]
                # Add tool_call_id for tool messages
                if "tool_call_id" in msg:
                    cleaned_msg["tool_call_id"] = msg["tool_call_id"]
                cleaned.append(cleaned_msg)
            else:
                cleaned.append(msg)
        return cleaned

    def invoke_thinking_with_tools(
        self,
        user_message: str,
        tools: List[Callable],
        max_turns: int = 10,
        thinking_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an interleaved thinking conversation with tool calls.

        This handles the complete flow:
        1. Send user message (with optional system prompt)
        2. Loop through: thinking → tool calls → thinking → tool calls → ...
        3. Return final answer with full trace

        Args:
            user_message: The user's question/request
            tools: List of callable functions (will be converted to tool schemas)
            max_turns: Maximum number of iterations (prevents infinite loops)
            thinking_budget: Tokens allocated for reasoning per turn
            system_prompt: Optional system prompt

        Returns:
            Dict with:
                - final_answer: The final content
                - full_trace: List of all reasoning steps and tool calls
                - total_tokens: Total tokens used
                - error: Any error that occurred
        """
        # Convert Python functions to tool schemas
        tool_schemas = []
        tool_map = {}
        for tool in tools:
            import inspect
            tool_name = tool.__name__
            tool_map[tool_name] = tool

            # Get function signature
            sig = inspect.signature(tool)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                param_type = param.annotation
                if param_name == 'return' or param.default is not param.empty:
                    continue

                # Map Python types to JSON schema types
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                }
                json_type = type_map.get(param_type, "string")

                properties[param_name] = {
                    "type": json_type,
                    "description": param_name,
                }

                if param.default is param.empty:
                    required.append(param_name)

            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.__doc__ or tool_name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })

        # Build initial messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        full_trace = []
        total_tokens = 0

        for turn in range(1, max_turns + 1):
            logger.debug(f"Thinking mode turn {turn}")

            # Invoke with thinking mode
            result = self.invoke_thinking(
                messages=messages,
                thinking_budget=thinking_budget,
                tools=tool_schemas if tool_schemas else None,
            )

            if "error" in result:
                return {
                    "final_answer": result["content"],
                    "full_trace": full_trace,
                    "total_tokens": total_tokens,
                    "error": result["error"],
                }

            # Add to trace
            full_trace.append({
                "turn": turn,
                "reasoning": result["reasoning_content"],
                "content": result["content"],
                "tool_calls": result["tool_calls"],
            })

            # Update token count
            if result["usage"]:
                total_tokens += getattr(result["usage"], "total_tokens", 0)

            # Check if there are tool calls
            if not result["tool_calls"]:
                # No more tools - we have the final answer
                return {
                    "final_answer": result["content"],
                    "full_trace": full_trace,
                    "total_tokens": total_tokens,
                    "error": None,
                }

            # Execute tool calls and prepare next request
            assistant_message = {
                "role": "assistant",
                "content": result["content"],
                "reasoning_content": result["reasoning_content"],
                "tool_calls": result["tool_calls"],
            }
            messages.append(assistant_message)

            # Execute each tool
            for tool_call in result["tool_calls"]:
                tool_name = tool_call.function.name
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call.function.arguments)
                    # Execute the function
                    tool_result = tool_map[tool_name](**arguments)
                    tool_result_str = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                except Exception as e:
                    tool_result_str = json.dumps({"error": str(e)})

                # Add tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_str,
                })

        return {
            "final_answer": "",
            "full_trace": full_trace,
            "total_tokens": total_tokens,
            "error": "Max turns exceeded",
        }


# Global thinking client instance
_deepseek_thinking_client: Optional[DeepSeekThinkingClient] = None


def get_deepseek_thinking_client() -> DeepSeekThinkingClient:
    """Get the global DeepSeek thinking mode client instance."""
    global _deepseek_thinking_client
    if _deepseek_thinking_client is None:
        _deepseek_thinking_client = DeepSeekThinkingClient()
    return _deepseek_thinking_client


def invoke_deepseek_thinking(
    messages: List[Dict[str, str]],
    thinking_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to invoke DeepSeek in thinking mode.

    Args:
        messages: List of message dicts
        thinking_budget: Optional token budget for reasoning

    Returns:
        Dict with content, reasoning_content, tool_calls, usage
    """
    client = get_deepseek_thinking_client()
    return client.invoke_thinking(messages, thinking_budget)
