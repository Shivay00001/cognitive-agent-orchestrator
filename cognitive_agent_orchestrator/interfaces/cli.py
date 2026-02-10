import asyncio
import logging
import argparse
import sys
from ..core.agent import MultiAgentSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli")

async def main():
    parser = argparse.ArgumentParser(description="Cognitive Agent Orchestrator CLI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    print("Initializing Cognitive Agent Orchestrator...")
    system = MultiAgentSystem()
    print("System ready.")
    
    if args.query:
        print(f"\nProcessing: {args.query}")
        result = await system.process_request(args.query)
        print("\n--- Result ---")
        print(result.get("response"))
        return

    if args.interactive:
        print("\nType 'exit' to quit.")
        while True:
            try:
                query = input("\nUser> ")
                if query.lower() in ["exit", "quit"]:
                    break
                if not query.strip():
                    continue
                    
                print("Thinking...")
                result = await system.process_request(query)
                print(f"\nAgent> {result.get('response')}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

    if not args.query and not args.interactive:
        parser.print_help()

def main_sync():
    """Synchronous wrapper for console_scripts entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
