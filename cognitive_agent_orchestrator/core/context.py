import re
import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any

class ContextAwareness:
    """
    Maintains awareness of the current context, including time, entities, and recent history.
    Based on heuristic analysis of text inputs.
    """
    
    def __init__(self):
        self.context_history = deque(maxlen=50)
        self.known_entities = {}
        
    def analyze_context(self, text: str) -> Dict:
        """Analyze text to update and return current context context."""
        
        # 1. Update History
        timestamp = datetime.datetime.now().isoformat()
        self.context_history.append({"text": text, "timestamp": timestamp})
        
        # 2. Extract Entities (Rule-based)
        entities = self._extract_entities(text)
        for e in entities:
            self.known_entities[e['value']] = e['type']
            
        return {
            "timestamp": timestamp,
            "detected_entities": entities,
            "history_depth": len(self.context_history),
            "temporal_context": self._get_time_context()
        }

    def _extract_entities(self, text: str) -> List[Dict]:
        """Simple regex-based entity extraction."""
        entities = []
        patterns = {
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Date': r'\b\d{4}-\d{2}-\d{2}\b',
            'Time': r'\b\d{1,2}:\d{2}\b',
            'CapitalizedTerm': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b' # Simple Proper Noun
        }
        
        for label, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({"type": label, "value": match})
        
        return entities

    def _get_time_context(self) -> str:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12: return "morning"
        if 12 <= hour < 17: return "afternoon"
        if 17 <= hour < 21: return "evening"
        return "night"
