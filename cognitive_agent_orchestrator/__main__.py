"""Entry point for `python -m cognitive_agent_orchestrator`."""

from .interfaces.cli import main
import asyncio
import sys

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
