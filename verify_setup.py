"""
Verification script for Cognitive Agent Orchestrator.
Run: python verify_setup.py
"""

import asyncio
import sys
import os

# Ensure we can import from the package directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_imports():
    """Step 1: Verify all modules import correctly."""
    print("=" * 50)
    print("STEP 1: Verifying imports...")
    errors = []

    try:
        from cognitive_agent_orchestrator.core.orchestrator import LLMOrchestrator
        print("  [OK] core.orchestrator.LLMOrchestrator")
    except ImportError as e:
        errors.append(f"  [FAIL] core.orchestrator: {e}")

    try:
        from cognitive_agent_orchestrator.core.agent import Agent, MultiAgentSystem, ReasoningEngine
        print("  [OK] core.agent (Agent, MultiAgentSystem, ReasoningEngine)")
    except ImportError as e:
        errors.append(f"  [FAIL] core.agent: {e}")

    try:
        from cognitive_agent_orchestrator.core.memory import MemoryManager
        print("  [OK] core.memory.MemoryManager")
    except ImportError as e:
        errors.append(f"  [FAIL] core.memory: {e}")

    try:
        from cognitive_agent_orchestrator.core.tools import SafeToolExecutor, ApprovalManager
        print("  [OK] core.tools (SafeToolExecutor, ApprovalManager)")
    except ImportError as e:
        errors.append(f"  [FAIL] core.tools: {e}")

    try:
        from cognitive_agent_orchestrator.core.learning import HeuristicLearner
        print("  [OK] core.learning.HeuristicLearner")
    except ImportError as e:
        errors.append(f"  [FAIL] core.learning: {e}")

    try:
        from cognitive_agent_orchestrator.core.context import ContextAwareness
        print("  [OK] core.context.ContextAwareness")
    except ImportError as e:
        errors.append(f"  [FAIL] core.context: {e}")

    try:
        from cognitive_agent_orchestrator.utils.config import config
        print("  [OK] utils.config")
    except ImportError as e:
        errors.append(f"  [FAIL] utils.config: {e}")

    try:
        from cognitive_agent_orchestrator.utils.logger import setup_logger
        print("  [OK] utils.logger.setup_logger")
    except ImportError as e:
        errors.append(f"  [FAIL] utils.logger: {e}")

    if errors:
        print("\nImport Errors:")
        for err in errors:
            print(err)
        return False

    print("\n  All imports passed.")
    return True


def verify_memory():
    """Step 2: Verify memory (SQLite) operations."""
    print("\n" + "=" * 50)
    print("STEP 2: Verifying MemoryManager (SQLite)...")

    from cognitive_agent_orchestrator.core.memory import MemoryManager

    # Use a temp DB for testing
    test_db = "test_verify_memory.db"
    try:
        mm = MemoryManager(db_path=test_db)
        print("  [OK] Database initialized")

        # Log an interaction
        interaction_id = mm.log_interaction("Test query", "Test response", importance=0.9)
        print(f"  [OK] Logged interaction: {interaction_id}")

        # Retrieve history
        history = mm.retrieve_relevant_history("Test query")
        assert len(history) > 0, "No history retrieved"
        print(f"  [OK] Retrieved {len(history)} relevant interaction(s)")

        return True
    except Exception as e:
        print(f"  [FAIL] Memory verification failed: {e}")
        return False
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"  [OK] Cleaned up test database")


def verify_learning():
    """Step 3: Verify HeuristicLearner."""
    print("\n" + "=" * 50)
    print("STEP 3: Verifying HeuristicLearner...")

    from cognitive_agent_orchestrator.core.learning import HeuristicLearner

    try:
        learner = HeuristicLearner()
        features = learner.extract_features("The quick brown fox jumps over the lazy dog")
        assert len(features) > 0, "No features extracted"
        print(f"  [OK] Extracted {len(features)} features: {features}")

        initial_score = learner.get_confidence_score(features)
        print(f"  [OK] Initial confidence: {initial_score:.4f}")

        learner.learn_from_feedback(features, positive_feedback=True)
        updated_score = learner.get_confidence_score(features)
        assert updated_score > initial_score, "Score should increase after positive feedback"
        print(f"  [OK] Updated confidence: {updated_score:.4f} (increased after positive feedback)")

        return True
    except Exception as e:
        print(f"  [FAIL] Learning verification failed: {e}")
        return False


def verify_context():
    """Step 4: Verify ContextAwareness."""
    print("\n" + "=" * 50)
    print("STEP 4: Verifying ContextAwareness...")

    from cognitive_agent_orchestrator.core.context import ContextAwareness

    try:
        ctx = ContextAwareness()
        result = ctx.analyze_context("Meeting with John Smith at 14:30 on 2025-03-15, email: john@example.com")

        assert "timestamp" in result, "Missing timestamp"
        assert "detected_entities" in result, "Missing entities"
        assert len(result["detected_entities"]) > 0, "No entities detected"
        assert "temporal_context" in result, "Missing temporal context"

        print(f"  [OK] Detected {len(result['detected_entities'])} entities:")
        for entity in result["detected_entities"]:
            print(f"       - {entity['type']}: {entity['value']}")
        print(f"  [OK] Temporal context: {result['temporal_context']}")

        return True
    except Exception as e:
        print(f"  [FAIL] Context verification failed: {e}")
        return False


async def verify_orchestrator():
    """Step 5: Verify LLMOrchestrator (mock mode)."""
    print("\n" + "=" * 50)
    print("STEP 5: Verifying LLMOrchestrator (mock mode)...")

    from cognitive_agent_orchestrator.core.orchestrator import LLMOrchestrator

    try:
        orch = LLMOrchestrator()
        # Without API keys, should fall back to mock
        result = await orch.generate("Plan a trip to Paris")

        assert result["success"], f"Mock generation failed: {result}"
        assert "response" in result, "Missing response field"
        print(f"  [OK] Provider: {result.get('provider', 'unknown')}")
        print(f"  [OK] Response: {result['response'][:80]}...")

        return True
    except Exception as e:
        print(f"  [FAIL] Orchestrator verification failed: {e}")
        return False


async def verify_multi_agent():
    """Step 6: Verify MultiAgentSystem end-to-end (mock mode)."""
    print("\n" + "=" * 50)
    print("STEP 6: Verifying MultiAgentSystem end-to-end (mock mode)...")

    from cognitive_agent_orchestrator.core.agent import MultiAgentSystem

    test_db = "test_verify_multi_agent.db"
    try:
        system = MultiAgentSystem()
        # Override memory DB path for testing
        system.memory.db_path = test_db
        system.memory._init_database()

        result = await system.process_request("Plan a 3-day trip to Tokyo")

        assert "response" in result, "Missing response"
        assert "query" in result, "Missing query"
        print(f"  [OK] Query: {result['query']}")
        print(f"  [OK] Response received: {str(result.get('response', ''))[:80]}...")

        return True
    except Exception as e:
        print(f"  [FAIL] MultiAgentSystem verification failed: {e}")
        return False
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


async def run_all():
    """Run all verification steps."""
    print("\n" + "=" * 50)
    print("  COGNITIVE AGENT ORCHESTRATOR - VERIFICATION")
    print("=" * 50)

    results = {}

    results["imports"] = verify_imports()
    results["memory"] = verify_memory()
    results["learning"] = verify_learning()
    results["context"] = verify_context()
    results["orchestrator"] = await verify_orchestrator()
    results["multi_agent"] = await verify_multi_agent()

    # Summary
    print("\n" + "=" * 50)
    print("  VERIFICATION SUMMARY")
    print("=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = "PASS" if status else "FAIL"
        print(f"  [{icon}] {name}")

    print(f"\n  Result: {passed}/{total} checks passed.")

    if passed == total:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print("  STATUS: SOME CHECKS FAILED - review output above.")

    return passed == total


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)
