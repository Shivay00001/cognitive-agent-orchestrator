import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("core.memory")

class MemoryManager:
    """
    Manages persistent storage for agent interactions and knowledge.
    Uses SQLite for robust local storage.
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Interaction Logs (formerly Episodic Memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                query TEXT,
                response TEXT,
                importance REAL,
                context TEXT
            )
        """)
        
        # Knowledge Base (formerly Semantic Memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                concept TEXT PRIMARY KEY,
                definition TEXT,
                confidence REAL,
                last_updated REAL
            )
        """)
        
        # Skills (formerly Procedural Memory)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_store (
                skill_id TEXT PRIMARY KEY,
                skill_name TEXT,
                steps TEXT,
                success_count INTEGER DEFAULT 0,
                last_used REAL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def log_interaction(self, query: str, response: str, importance: float = 0.5) -> str:
        """Log a user-agent interaction."""
        import hashlib
        
        interaction_id = hashlib.md5(f"{query}{datetime.now().timestamp()}".encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interaction_logs (id, timestamp, query, response, importance, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (interaction_id, datetime.now().timestamp(), query, response, importance, json.dumps({})))
        
        conn.commit()
        conn.close()
        return interaction_id

    def retrieve_relevant_history(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant past interactions based on simple keyword matching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        keywords = query.lower().split()
        results = []
        
        cursor.execute("SELECT * FROM interaction_logs ORDER BY timestamp DESC LIMIT 50")
        for row in cursor.fetchall():
            interaction_id, timestamp, q, resp, importance, context = row
            # Basic relevance scoring
            score = sum(1 for kw in keywords if kw in q.lower() or kw in resp.lower())
            if score > 0:
                results.append({
                    "id": interaction_id,
                    "query": q,
                    "response": resp,
                    "importance": importance,
                    "relevance": score,
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat()
                })
        
        conn.close()
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]
