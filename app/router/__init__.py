"""
Router Module v2.0 - Smart Pre-processor Architecture

This module has been completely redesigned around the Smart Pre-processor concept.
The new architecture provides:

1. Global Language State Manager - Language detection and standardization
2. Unified Intent Classifier - AI-driven intent classification and entity extraction  
3. Smart Post-processor - Response language adaptation

Key Features:
- Unified English processing pipeline
- Eliminates hard-coded rules and regex patterns
- Deep integration with core QueryContext architecture
- Backward compatibility with existing code
"""

# New Smart Pre-processor (Primary Interface)
from .smart_preprocessor import (
    SmartPreProcessor,
    GlobalLanguageStateManager, 
    UnifiedIntentClassifier,
    SmartPostProcessor,
    NormalizedQuery,
    ParsedIntent,
    smart_preprocessor,
    router_compatibility
)

# Legacy imports for backward compatibility (commented out, available in backup)
# from .zero_shot_gatekeeper import zero_shot_gatekeeper
# from .intent_classifier import intent_classifier
# from .intelligent_router import intelligent_router

# Primary exports - New Architecture
__all__ = [
    # Main smart preprocessor
    'SmartPreProcessor',
    'smart_preprocessor',
    
    # Component classes
    'GlobalLanguageStateManager',
    'UnifiedIntentClassifier', 
    'SmartPostProcessor',
    
    # Data structures
    'NormalizedQuery',
    'ParsedIntent',
    
    # Compatibility layer
    'router_compatibility'
]

# Backward compatibility functions
def get_intelligent_router():
    """Legacy function - returns compatibility layer"""
    return router_compatibility

def get_intent_classifier():
    """Legacy function - returns smart preprocessor's classifier"""
    return smart_preprocessor.intent_classifier
