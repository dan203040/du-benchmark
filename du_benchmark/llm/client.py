# CONFIDENTIAL — submitted for VLDB review only. Redistribution is not permitted.
"""
Minimal LLM Client for DU Benchmark.

Provides a unified interface for calling LLMs (Claude, Gemini, DeepSeek, OpenAI)
to generate DU consensus and run extractors.

Usage:
    from du_benchmark.llm.client import create_llm_client, LLMProvider

    client = create_llm_client(LLMProvider.DEEPSEEK, model="deepseek-chat")
    response = await client.generate(prompt="...", system="...")
"""

import asyncio
import functools
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

_logger = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is a transient API error worth retrying."""
    exc_str = str(exc).lower()
    for code in ("429", "500", "502", "503", "504", "rate limit", "rate_limit",
                  "too many requests", "server error", "connection",
                  "timeout", "timed out", "overloaded"):
        if code in exc_str:
            return True
    cls_name = type(exc).__name__.lower()
    for tag in ("ratelimit", "apierror", "timeout", "connection", "server"):
        if tag in cls_name:
            return True
    return False


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Try to extract a Retry-After value (seconds) from the exception."""
    if hasattr(exc, "response") and hasattr(exc.response, "headers"):
        val = exc.response.headers.get("retry-after") or exc.response.headers.get("Retry-After")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def retry_on_transient(max_retries: int = 3, initial_delay: float = 1.0,
                       backoff_factor: float = 2.0):
    """Decorator that retries an async function on transient API errors."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries or not _is_retryable(exc):
                        raise
                    retry_after = _extract_retry_after(exc)
                    wait = retry_after if retry_after is not None else delay
                    _logger.warning(
                        "Retryable error on attempt %d/%d for %s: %s — "
                        "retrying in %.1fs",
                        attempt + 1, max_retries, fn.__qualname__,
                        exc, wait,
                    )
                    await asyncio.sleep(wait)
                    delay *= backoff_factor
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


DEFAULT_MODELS = {
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.GOOGLE: "gemini-1.5-pro",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.DEEPSEEK: "deepseek-chat",
}


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    usage: Dict[str, int]
    success: bool
    error: Optional[str] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None,
                       seed: Optional[int] = None) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class AnthropicClient(BaseLLMClient):
    """Claude API client using Anthropic SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096, temperature: float = 0.1):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def is_available(self) -> bool:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return api_key is not None and len(api_key) > 0

    @retry_on_transient()
    async def generate(self, prompt: str, system: Optional[str] = None,
                       seed: Optional[int] = None) -> LLMResponse:
        try:
            client = self._get_client()
            messages = [{"role": "user", "content": prompt}]
            kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
            return LLMResponse(
                content=content, model=response.model,
                usage={"input_tokens": response.usage.input_tokens,
                       "output_tokens": response.usage.output_tokens},
                success=True,
            )
        except Exception as e:
            if _is_retryable(e):
                raise
            _logger.error("Non-retryable Anthropic API error: %s", e)
            return LLMResponse(content="", model=self.model, usage={},
                               success=False, error=str(e))


class GoogleClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self, model: str = "gemini-1.5-pro",
                 max_tokens: int = 8192, temperature: float = 0.1):
        self.model = model
        if "2.5" in model or "2.0-flash-thinking" in model:
            self.max_tokens = max(max_tokens, 16384)
        else:
            self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        return self._client

    def is_available(self) -> bool:
        api_key = os.environ.get("GOOGLE_API_KEY")
        return api_key is not None and len(api_key) > 0

    @retry_on_transient()
    async def generate(self, prompt: str, system: Optional[str] = None,
                       seed: Optional[int] = None) -> LLMResponse:
        genai = self._get_client()
        model = genai.GenerativeModel(self.model, system_instruction=system if system else None)
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens, temperature=self.temperature),
            )
        except Exception as e:
            if _is_retryable(e):
                raise
            _logger.error("Non-retryable Gemini API error: %s", e)
            return LLMResponse(content="", model=self.model, usage={},
                               success=False, error=str(e))
        try:
            text = response.text
        except ValueError as e:
            _logger.error("Gemini response has no text: %s", e)
            return LLMResponse(content="", model=self.model, usage={},
                               success=False, error=f"No text in response: {e}")
        return LLMResponse(content=text, model=self.model, usage={}, success=True)


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client using OpenAI-compatible API."""

    def __init__(self, model: str = "deepseek-chat",
                 max_tokens: int = 8192, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DEEPSEEK_API_KEY environment variable not set")
                self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        return api_key is not None and len(api_key) > 0

    @retry_on_transient()
    async def generate(self, prompt: str, system: Optional[str] = None,
                       seed: Optional[int] = None) -> LLMResponse:
        try:
            client = self._get_client()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            create_kwargs: Dict[str, Any] = dict(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            if seed is not None:
                create_kwargs["seed"] = seed
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content if response.choices else ""
            return LLMResponse(
                content=content or "", model=response.model,
                usage={"input_tokens": response.usage.prompt_tokens if response.usage else 0,
                       "output_tokens": response.usage.completion_tokens if response.usage else 0},
                success=True,
            )
        except Exception as e:
            if _is_retryable(e):
                raise
            _logger.error("Non-retryable DeepSeek API error: %s", e)
            return LLMResponse(content="", model=self.model, usage={},
                               success=False, error=str(e))


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4o, GPT-5, etc.)."""

    def __init__(self, model: str = "gpt-4o",
                 max_tokens: int = 8192, temperature: float = 0.0):
        self.model = model
        _is_reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
        self.max_tokens = max(max_tokens, 16384) if _is_reasoning else max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        api_key = os.environ.get("OPENAI_API_KEY")
        return api_key is not None and len(api_key) > 0

    @retry_on_transient()
    async def generate(self, prompt: str, system: Optional[str] = None,
                       seed: Optional[int] = None) -> LLMResponse:
        try:
            client = self._get_client()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            _is_new_model = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
            token_param = "max_completion_tokens" if _is_new_model else "max_tokens"
            create_kwargs: Dict[str, Any] = dict(model=self.model, messages=messages)
            if not _is_new_model:
                create_kwargs["temperature"] = self.temperature
            create_kwargs[token_param] = self.max_tokens
            if seed is not None:
                create_kwargs["seed"] = seed
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content if response.choices else ""
            return LLMResponse(
                content=content or "", model=response.model,
                usage={"input_tokens": response.usage.prompt_tokens if response.usage else 0,
                       "output_tokens": response.usage.completion_tokens if response.usage else 0},
                success=True,
            )
        except Exception as e:
            if _is_retryable(e):
                raise
            _logger.error("Non-retryable OpenAI API error: %s", e)
            return LLMResponse(content="", model=self.model, usage={},
                               success=False, error=str(e))


def create_llm_client(
    provider: LLMProvider = LLMProvider.ANTHROPIC,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Factory function to create an LLM client."""
    if provider == LLMProvider.ANTHROPIC:
        client = AnthropicClient(model=model or DEFAULT_MODELS[LLMProvider.ANTHROPIC], **kwargs)
        if client.is_available():
            return client

    if provider == LLMProvider.GOOGLE:
        client = GoogleClient(model=model or DEFAULT_MODELS[LLMProvider.GOOGLE], **kwargs)
        if client.is_available():
            return client

    if provider == LLMProvider.OPENAI:
        client = OpenAIClient(model=model or DEFAULT_MODELS[LLMProvider.OPENAI], **kwargs)
        if client.is_available():
            return client

    if provider == LLMProvider.DEEPSEEK:
        client = DeepSeekClient(model=model or DEFAULT_MODELS[LLMProvider.DEEPSEEK], **kwargs)
        if client.is_available():
            return client

    raise RuntimeError(
        f"No API key available for provider {provider.value}. "
        f"Set the appropriate environment variable "
        f"(ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY)."
    )
