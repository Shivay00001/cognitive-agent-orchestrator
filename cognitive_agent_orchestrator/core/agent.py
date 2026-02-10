import asyncio
import json
import logging
from typing import Dict, List, Any
from .orchestrator import LLMOrchestrator
from .tools import ApprovalManager, SafeToolExecutor
from .memory import MemoryManager

logger = logging.getLogger("core.agent")

class Agent:
    """
    A single autonomous agent with a specific role.
    """
    def __init__(self, role: str, orchestrator: LLMOrchestrator, tools: SafeToolExecutor):
        self.role = role
        self.orchestrator = orchestrator
        self.tools = tools
        
    async def execute(self, task: str, context: Dict) -> Dict:
        """
        Execute a task given context using the LLM.
        """
        system_prompt = f"You are a {self.role}. focused on solving the user's request efficiently."
        prompt = f"Task: {task}\nContext: {json.dumps(context, default=str)}\n\nProvide your analysis and solution:"
        
        result = await self.orchestrator.generate(prompt, system_prompt=system_prompt)
        
        return {
            "agent": self.role,
            "result": result.get("response", "No response generated"),
            "success": result.get("success", False)
        }

class ReasoningEngine:
    """
    Executes parallel reasoning strategies to improve decision quality.
    """
    def __init__(self, orchestrator: LLMOrchestrator):
        self.orchestrator = orchestrator
        
    async def analyze(self, query: str, context: Dict) -> Dict:
        """
        Run parallel analysis streams (Logical, Creative, Critical).
        """
        strategies = ["Logical Analysis", "Critical Evaluation", "Creative Solutioning"]
        tasks = []
        
        for strategy in strategies:
            prompt = f"Analyze the following query using {strategy}:\nQuery: {query}\nContext: {json.dumps(context, default=str)}"
            tasks.append(self.orchestrator.generate(prompt))
            
        results = await asyncio.gather(*tasks)
        
        return {
            strategy: res.get("response") 
            for strategy, res in zip(strategies, results)
        }

class MultiAgentSystem:
    """
    Coordinated system of agents working together.
    """
    def __init__(self):
        self.orchestrator = LLMOrchestrator()
        self.approvals = ApprovalManager()
        self.tools = SafeToolExecutor(self.approvals)
        self.memory = MemoryManager()
        self.reasoning = ReasoningEngine(self.orchestrator)
        
        self.agents = {
            "planner": Agent("Strategic Planner", self.orchestrator, self.tools),
            "researcher": Agent("Research Analyst", self.orchestrator, self.tools),
            "critic": Agent("Quality Reviewer", self.orchestrator, self.tools),
            "verifier": Agent("Quality Verifier", self.orchestrator, self.tools)
        }
        
    async def process_request(self, query: str) -> Dict:
        """
        Main entry point for processing a user request.
        """
        logger.info(f"Processing request: {query}")
        
        # 1. Retrieve Context
        past_interactions = self.memory.retrieve_relevant_history(query)
        context = {
            "query": query,
            "history": past_interactions
        }
        
        # 2. Parallel Reasoning (Optional, for complex tasks)
        # reasoning_results = await self.reasoning.analyze(query, context)
        # context["reasoning"] = reasoning_results
        
        # 3. Multi-Agent Collaboration
        # A simple linear flow for now: Plan -> Research -> Critique -> Final
        
        plan = await self.agents["planner"].execute(f"Create a plan for: {query}", context)
        context["plan"] = plan["result"]
        
        research = await self.agents["researcher"].execute(f"Execute plan items for: {query}", context)
        context["research"] = research["result"]
        
        # 4. Verification
        verification = await self.agents["verifier"].execute(
            "Verify quality and correctness of the plan and research.",
            context
        )
        context["verification"] = verification["result"]
        
        # 5. Synthesize Final Response
        final_response = await self.orchestrator.generate(
            f"Synthesize the final answer based on the plan, research, and verification.\n"
            f"Query: {query}\nPlan: {plan['result']}\nResearch: {research['result']}\n"
            f"Verification: {verification['result']}"
        )
        
        # 6. Save to Memory
        self.memory.log_interaction(query, final_response.get("response", ""), importance=0.8)
        
        return {
            "query": query,
            "response": final_response.get("response"),
            "plan": plan.get("result"),
            "research": research.get("result"),
            "verification": verification.get("result")
        }
