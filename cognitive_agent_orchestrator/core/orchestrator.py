import aiohttp
import json
import logging
import asyncio
from typing import Dict, Optional, List, Any
from ..utils.config import config

logger = logging.getLogger("core.orchestrator")

class LLMOrchestrator:
    """
    Manages interactions with LLM providers (OpenAI, Anthropic, Groq).
    Provides fallback and unified interface.
    """
    
    def __init__(self):
        self.providers = {
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
            "groq": self._call_groq
        }
        self.default_provider = "openai" if config.openai_api_key else "groq" if config.groq_api_key else "anthropic"
        if not config.validate_llm_keys():
            logger.warning("No API keys detected in configuration (Result set to Mock mode).")
            self.default_provider = "mock"

    async def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", provider: Optional[str] = None) -> Dict:
        """
        Generate a response from the LLM.
        """
        provider = provider or self.default_provider
        
        if provider == "mock":
             return {"success": True, "response": f"[MOCK] Response to: {prompt[:20]}...", "provider": "mock"}

        try:
            handler = self.providers.get(provider)
            if not handler:
                return {"success": False, "error": f"Provider {provider} not supported."}
            
            response_text = await handler(prompt, system_prompt)
            return {
                "success": True, 
                "response": response_text, 
                "provider": provider
            }
        except Exception as e:
            logger.error(f"LLM generation failed ({provider}): {e}")
            return {"success": False, "error": str(e)}

    async def _call_openai(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                if "choices" not in data:
                     raise ValueError(f"OpenAI API Error: {data}")
                return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                if "content" not in data:
                    raise ValueError(f"Anthropic API Error: {data}")
                return data["content"][0]["text"]

    async def _call_groq(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json()
                if "choices" not in data:
                    raise ValueError(f"Groq API Error: {data}")
                return data["choices"][0]["message"]["content"]
