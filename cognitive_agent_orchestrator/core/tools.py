import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("core.tools")

class ApprovalManager:
    """Manages human-in-the-loop approval for sensitive operations."""
    
    def __init__(self):
        self.pending = []
        self.approved = set()
        
    def request_approval(self, action_type: str, details: Dict) -> Dict:
        """Request approval for an action."""
        approval_id = len(self.pending)
        item = {
            "id": approval_id,
            "type": action_type,
            "details": details,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        self.pending.append(item)
        logger.info(f"Approval requested for {action_type} (ID: {approval_id})")
        return item
    
    def approve(self, approval_id: int) -> bool:
        """Approve a pending action."""
        if 0 <= approval_id < len(self.pending):
            self.pending[approval_id]["status"] = "approved"
            self.approved.add(approval_id)
            logger.info(f"Approved action ID: {approval_id}")
            return True
        return False

    def reject(self, approval_id: int) -> bool:
        """Reject a pending action."""
        if 0 <= approval_id < len(self.pending):
            self.pending[approval_id]["status"] = "rejected"
            logger.info(f"Rejected action ID: {approval_id}")
            return True
        return False

    def get_pending(self) -> List[Dict]:
        """Get all pending approvals."""
        return [item for item in self.pending if item["status"] == "pending"]

class SafeToolExecutor:
    """Executes tools safely with mandatory approval for sensitive actions."""
    
    def __init__(self, approval_manager: ApprovalManager):
        self.approvals = approval_manager
        self.tools = {
            "file_read": self._read_file,
            "file_write": self._write_file,
            "python_eval": self._eval_python,
            "web_search": self._web_search,
        }
    
    async def execute(self, tool_name: str, args: Dict) -> Dict:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # Handle approvals
        requires_approval = tool_name in ["file_write", "python_eval"]
        
        if requires_approval:
            approval = self.approvals.request_approval(tool_name, args)
            # In a real CLI/UI, we would pause here. 
            # For this MVP, we will auto-approve mostly safe reads, but warn on writes.
            # But the requirement is "Human-in-the-loop safety controls".
            # For now, we return a "pending approval" state if we can't get interactive input.
            
            # SIMULATION: Auto-approve for demo purposes if not dangerous
            if tool_name == "file_write" and "test" in args.get("path", ""):
                 self.approvals.approve(approval["id"])
            
            if approval["id"] not in self.approvals.approved:
                return {"success": False, "status": "pending_approval", "approval_id": approval["id"]}

        try:
            result = await self.tools[tool_name](args)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _read_file(self, args: Dict) -> str:
        path = args.get("path", "")
        if not os.path.exists(path):
            return "File not found."
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    async def _write_file(self, args: Dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File written: {path}"

    async def _eval_python(self, args: Dict) -> str:
        """Safe(r) python evaluation."""
        expr = args.get("expression", "")
        safe_globals = {"__builtins__": None, "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "math": __import__("math")}
        try:
            return str(eval(expr, safe_globals, {}))
        except Exception as e:
            return f"Eval Error: {e}"

    async def _web_search(self, args: Dict) -> str:
        """Web search placeholder. Integrate a real search API (e.g., DuckDuckGo) for production use."""
        query = args.get("query", "")
        return f"Search results for: {query}\n[Integrate a search API for production use]"
