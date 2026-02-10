import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Application configuration loaded from environment variables.
    """
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Storage
    db_path: str = os.getenv("DB_PATH", "agent_memory.db")
    
    # System
    debug_mode: bool = os.getenv("DEBUG", "False").lower() == "true"

    def validate_llm_keys(self) -> bool:
        """Returns True if at least one LLM API key is available."""
        return any([self.openai_api_key, self.anthropic_api_key, self.groq_api_key])

config = Config()
