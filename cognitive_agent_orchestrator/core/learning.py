import json
import logging
from collections import defaultdict
from typing import Dict, Any, List

logger = logging.getLogger("core.learning")

class HeuristicLearner:
    """
    Simple heuristic learning module.
    Adjusts weights based on feedback loops, without claiming to be a neural network.
    """
    
    def __init__(self):
        # Maps features -> weight (0.0 to 1.0)
        self.feature_weights = defaultdict(lambda: 0.5)
        self.learning_rate = 0.01

    def learn_from_feedback(self, context_features: List[str], positive_feedback: bool):
        """
        Adjust weights based on simple binary feedback.
        If feedback is positive, increase weights for present features.
        If negative, decrease.
        """
        direction = 1 if positive_feedback else -1
        
        for feature in context_features:
            current = self.feature_weights[feature]
            # Simple update rule
            new_weight = current + (direction * self.learning_rate)
            # Clamp between 0 and 1
            self.feature_weights[feature] = max(0.0, min(1.0, new_weight))
            
        logger.info(f"Updated weights for {len(context_features)} features. Feedback: {'Positive' if positive_feedback else 'Negative'}")

    def get_confidence_score(self, context_features: List[str]) -> float:
        """
        Calculate a rudimentary confidence score based on historical performance of features.
        """
        if not context_features:
            return 0.5
            
        total_weight = sum(self.feature_weights[f] for f in context_features)
        return total_weight / len(context_features)

    def extract_features(self, text: str) -> List[str]:
        """
        Extract simple bag-of-words features for learning context.
        """
        # Simple tokenization
        words = text.lower().split()
        return [w for w in words if len(w) > 3] # Filter short words
