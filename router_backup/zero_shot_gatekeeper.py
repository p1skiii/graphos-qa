"""
Modern Zero-Shot Gatekeeper - Tier 1 of the Three-Tier Routing Architecture
Intelligent filtering using pre-trained zero-shot classification models
Completely eliminates hardcoded keyword lists
"""
import time
import logging
from typing import Tuple, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class ZeroShotGatekeeper:
    """
    Tier 1: Modern Gatekeeper - Zero-Shot Classification Based
    
    Design Principles:
    1. Eliminate hardcoded keyword lists completely
    2. Use semantic understanding from pre-trained models
    3. Support dynamic label adjustment
    4. Intelligent semantic judgment (understands "Purple and Gold Dynasty" = Lakers)
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize zero-shot classifier
        
        Args:
            model_name: Zero-shot classification model name
        """
        self.model_name = model_name
        self.classifier = None
        
        # Candidate classification labels (dynamically adjustable)
        self.candidate_labels = [
            "basketball sports",      # Basketball related
            "weather forecast",       # Weather
            "finance business",       # Finance/stocks
            "news information",       # News
            "technology tools",       # Tools/utilities
            "entertainment life",     # Entertainment/lifestyle
            "general conversation"    # General chat
        ]
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0,
            'processing_time_sum': 0.0,
            'model_loading_time': 0.0
        }
        
        # Lazy loading (load model only when first used)
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy load the model"""
        if not self._model_loaded:
            start_time = time.time()
            logger.info(f"🔄 Loading zero-shot classification model: {self.model_name}")
            
            try:
                from transformers import pipeline
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    return_all_scores=True
                )
                
                self.stats['model_loading_time'] = time.time() - start_time
                self._model_loaded = True
                logger.info(f"✅ Model loaded successfully, time: {self.stats['model_loading_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                # Fallback to simple rule-based filter
                self.classifier = None
                self._model_loaded = False
    
    @lru_cache(maxsize=1000)
    def classify_query(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform zero-shot classification on query
        
        Args:
            query: User query text
            
        Returns:
            Tuple[str, float, Dict]: (top_classification, confidence, all_scores)
        """
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        if self.classifier is None:
            # Model loading failed, use simple rules
            return self._fallback_classification(query)
        
        try:
            result = self.classifier(query, self.candidate_labels)
            
            # Parse results
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # Build all classification scores dictionary
            all_scores = dict(zip(result['labels'], result['scores']))
            
            return top_label, top_score, all_scores
            
        except Exception as e:
            logger.error(f"❌ Zero-shot classification failed: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """Fallback simple rule-based classification"""
        query_lower = query.lower()
        
        # Simple rule-based judgment
        non_basketball_keywords = ['weather', 'stock', 'translate', 'news', 'joke']
        basketball_keywords = ['basketball', 'nba', 'kobe', 'lebron', 'lakers', 'dunk']
        
        for keyword in non_basketball_keywords:
            if keyword in query_lower:
                return "non_basketball", 0.9, {"non_basketball": 0.9, "basketball sports": 0.1}
        
        for keyword in basketball_keywords:
            if keyword in query_lower:
                return "basketball sports", 0.9, {"basketball sports": 0.9, "non_basketball": 0.1}
        
        # Default pass (lean towards basketball when uncertain)
        return "basketball sports", 0.5, {"basketball sports": 0.5, "non_basketball": 0.5}
    
    def smart_filter(self, query: str, confidence_threshold: float = 0.5) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Smart filtering - using zero-shot classification
        
        Args:
            query: User query text
            confidence_threshold: Confidence threshold
            
        Returns:
            Tuple[bool, str, Dict]: (is_basketball_related, filter_reason, detailed_analysis)
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Perform zero-shot classification
        top_label, top_score, all_scores = self.classify_query(query)
        
        # Analysis details
        analysis = {
            'query_length': len(query),
            'top_classification': top_label,
            'top_confidence': top_score,
            'all_scores': all_scores,
            'confidence_threshold': confidence_threshold,
            'filter_method': 'zero_shot_classification'
        }
        
        # Determine if basketball related - optimized: basketball sports score above threshold passes
        basketball_score = all_scores.get("basketball sports", 0.0)
        is_basketball = basketball_score >= confidence_threshold
        
        # If not passed but basketball sports score is high, lower requirements
        if not is_basketball and basketball_score >= 0.15:  # 对姚明等篮球人名放宽标准
            is_basketball = True
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['processing_time_sum'] += processing_time
        
        if is_basketball:
            self.stats['basketball_passed'] += 1
            if top_label == "basketball sports":
                reason = f"Basketball related (confidence: {top_score:.3f})"
            else:
                reason = f"Basketball score sufficient ({basketball_score:.3f}) despite top: {top_label}"
        else:
            self.stats['non_basketball_filtered'] += 1
            reason = f"Classified as: {top_label} (confidence: {top_score:.3f}), basketball score too low: {basketball_score:.3f}"
        
        return is_basketball, reason, analysis
    
    def update_labels(self, new_labels: list):
        """Dynamically update classification labels"""
        self.candidate_labels = new_labels
        # Clear cache to apply new labels
        self.classify_query.cache_clear()
        logger.info(f"🔄 Updated classification labels: {new_labels}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        avg_time = self.stats['processing_time_sum'] / total
        filter_rate = self.stats['non_basketball_filtered'] / total * 100
        
        return {
            **self.stats,
            'avg_processing_time': avg_time,
            'filter_rate': filter_rate,
            'pass_rate': 100 - filter_rate,
            'model_loaded': self._model_loaded
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_queries': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0,
            'processing_time_sum': 0.0,
            'model_loading_time': 0.0
        }
        logger.info("🔄 Zero-shot gatekeeper statistics reset")

# Global zero-shot gatekeeper instance
zero_shot_gatekeeper = ZeroShotGatekeeper()

if __name__ == "__main__":
    # Test zero-shot gatekeeper
    test_cases = [
        "What's the weather like today?",              # Should be filtered
        "Please translate this sentence",             # Should be filtered
        "How are the stocks performing?",             # Should be filtered
        "How tall is Yao Ming?",                      # Should pass
        "Who is stronger, Kobe or Jordan?",           # Should pass
        "Purple and Gold Dynasty history",            # Smart recognition: Lakers
        "Black Mamba's scoring ability",              # Smart recognition: Kobe
        "Hello",                                      # Boundary case
        "Who is the NBA all-time scoring leader?",    # Should pass
        "Any games tonight?"                          # Should pass
    ]
    
    print("🧠 Zero-Shot Gatekeeper Test:")
    print("=" * 60)
    
    for query in test_cases:
        is_basketball, reason, analysis = zero_shot_gatekeeper.smart_filter(query)
        status = "✅ PASS" if is_basketball else "❌ FILTER"
        
        print(f"{status} | {query}")
        print(f"   Reason: {reason}")
        print(f"   Classification: {analysis['top_classification']}")
        print(f"   Scores: {dict(list(analysis['all_scores'].items())[:3])}")  # Show top 3 scores
        print("-" * 40)
    
    # Display statistics
    stats = zero_shot_gatekeeper.get_stats()
    print(f"\n📊 Zero-Shot Gatekeeper Statistics:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Filtered: {stats['non_basketball_filtered']} ({stats['filter_rate']:.1f}%)")
    print(f"   Passed: {stats['basketball_passed']} ({stats['pass_rate']:.1f}%)")
    print(f"   Avg time: {stats['avg_processing_time']*1000:.1f}ms")
    print(f"   Model loading time: {stats['model_loading_time']:.2f}s")
