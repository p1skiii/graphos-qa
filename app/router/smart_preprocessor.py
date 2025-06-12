"""
Smart Pre-processor: Intelligent Router Architecture v2.0

This module replaces the entire routing system with a unified intelligent pre-processor.
Core functions: Language standardization and unified intent parsing.

Architecture:
1. Global Language State Manager: Language detection and standardization
2. Unified Intent Classification and Entity Extraction Model
3. Smart Post-processor for response language adaptation
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

# Core data structure imports
from app.core.schemas import (
    QueryContext, QueryContextFactory,
    LanguageInfo, IntentInfo, EntityInfo
)

# =============================================================================
# Additional Data Structures for Smart Pre-processor
# =============================================================================

@dataclass
class ProcessingStep:
    """Processing step record"""
    step_name: str
    status: str  # success, error, warning
    duration: float
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None

@dataclass 
class PerformanceMetrics:
    """Performance metrics collection"""
    total_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    success_count: int = 0
    error_count: int = 0

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures for New Router
# =============================================================================

@dataclass
class NormalizedQuery:
    """Standardized query object - the global context carrier"""
    original_text: str                          # Original user input
    normalized_text: str                        # Standardized English text
    original_language: str                      # Original language code
    confidence_score: float                     # Language detection confidence
    translation_applied: bool = False           # Whether translation was applied
    preprocessing_time: float = 0.0             # Preprocessing duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'original_text': self.original_text,
            'normalized_text': self.normalized_text,
            'original_language': self.original_language,
            'confidence_score': self.confidence_score,
            'translation_applied': self.translation_applied,
            'preprocessing_time': self.preprocessing_time
        }

@dataclass
class ParsedIntent:
    """Structured information extraction result"""
    intent: str                                 # Intent classification
    confidence: float                           # Intent confidence
    entities: Dict[str, Any]                    # Extracted entities
    is_basketball_related: bool                 # Domain relevance flag
    
    # Detailed entity breakdown
    players: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    
    # Processing metadata
    model_used: str = "lightweight_bert"
    processing_time: float = 0.0
    extraction_method: str = "neural_extraction"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'intent': self.intent,
            'confidence': self.confidence,
            'entities': self.entities,
            'is_basketball_related': self.is_basketball_related,
            'players': self.players,
            'teams': self.teams,
            'attributes': self.attributes,
            'model_used': self.model_used,
            'processing_time': self.processing_time,
            'extraction_method': self.extraction_method
        }

# =============================================================================
# Step 1: Smart Pre-processor Core
# =============================================================================

class GlobalLanguageStateManager:
    """Global language state manager: Language detection and standardization"""
    
    def __init__(self):
        self.supported_languages = ['en', 'zh', 'auto']
        self.translation_cache = {}  # Simple translation cache
        self.stats = {
            'total_queries': 0,
            'english_queries': 0,
            'chinese_queries': 0,
            'translations_performed': 0,
            'cache_hits': 0
        }
        logger.info("🌐 Global Language State Manager initialized")
    
    def detect_and_normalize_language(self, text: str) -> NormalizedQuery:
        """
        Detect language and normalize to English
        
        Args:
            text: User input text
            
        Returns:
            NormalizedQuery: Standardized query object
        """
        start_time = time.time()
        
        try:
            # Update statistics
            self.stats['total_queries'] += 1
            
            # Language detection (simplified implementation)
            detected_language, confidence = self._detect_language(text)
            
            # Language standardization
            if detected_language == 'zh':
                # Chinese detected, translation needed
                normalized_text = self._translate_to_english(text)
                translation_applied = True
                self.stats['chinese_queries'] += 1
                self.stats['translations_performed'] += 1
            else:
                # English or unknown, use as-is
                normalized_text = text
                translation_applied = False
                self.stats['english_queries'] += 1
            
            processing_time = time.time() - start_time
            
            result = NormalizedQuery(
                original_text=text,
                normalized_text=normalized_text,
                original_language=detected_language,
                confidence_score=confidence,
                translation_applied=translation_applied,
                preprocessing_time=processing_time
            )
            
            logger.info(f"🌐 Language normalized: {detected_language} -> en (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Language normalization failed: {str(e)}")
            # Fallback: treat as English
            return NormalizedQuery(
                original_text=text,
                normalized_text=text,
                original_language='en',
                confidence_score=0.5,
                translation_applied=False,
                preprocessing_time=time.time() - start_time
            )
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect text language
        
        Args:
            text: Input text
            
        Returns:
            Tuple[language_code, confidence_score]
        """
        try:
            # Simple rule-based detection (production should use langdetect or fasttext)
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            total_chars = len(text.replace(' ', ''))
            
            if total_chars == 0:
                return 'en', 0.5
            
            chinese_ratio = chinese_chars / total_chars
            
            if chinese_ratio > 0.3:
                return 'zh', min(0.9, 0.5 + chinese_ratio)
            else:
                return 'en', min(0.9, 0.5 + (1 - chinese_ratio))
                
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {str(e)}")
            return 'en', 0.5
    
    def _translate_to_english(self, chinese_text: str) -> str:
        """
        Translate Chinese to English
        
        Args:
            chinese_text: Chinese input text
            
        Returns:
            str: English translation
        """
        try:
            # Check translation cache first
            if chinese_text in self.translation_cache:
                self.stats['cache_hits'] += 1
                return self.translation_cache[chinese_text]
            
            # Simple mapping for common basketball queries (production should use translation API)
            translation_map = {
                '科比多少岁': 'How old is Kobe Bryant',
                '科比多大': 'How old is Kobe Bryant', 
                '姚明多高': 'How tall is Yao Ming',
                '詹姆斯在哪个队': 'What team does LeBron James play for',
                '湖人队有谁': 'Who plays for the Lakers',
                '科比和乔丹谁厉害': 'Who is better, Kobe or Jordan',
                '你好': 'Hello',
                '篮球是什么': 'What is basketball',
                '科比': 'Kobe Bryant',
                '姚明': 'Yao Ming',
                '詹姆斯': 'LeBron James',
                '乔丹': 'Michael Jordan'
            }
            
            # Try exact match first
            if chinese_text in translation_map:
                translation = translation_map[chinese_text]
                self.translation_cache[chinese_text] = translation
                return translation
            
            # Try partial matching for common terms
            english_text = chinese_text
            for chinese, english in translation_map.items():
                if chinese in chinese_text:
                    english_text = chinese_text.replace(chinese, english)
                    break
            
            # Cache the translation
            self.translation_cache[chinese_text] = english_text
            return english_text
            
        except Exception as e:
            logger.error(f"❌ Translation failed: {str(e)}")
            return chinese_text  # Fallback to original text

class UnifiedIntentClassifier:
    """
    Unified Intent Classification and Entity Extraction Model
    
    Replaces the entire routing system with a single AI-driven step
    that simultaneously performs intent classification and entity extraction.
    """
    
    def __init__(self):
        self.intent_labels = [
            'ATTRIBUTE_QUERY',          # Age, height, weight queries
            'SIMPLE_RELATION_QUERY',    # Team affiliation, simple facts
            'COMPLEX_RELATION_QUERY',   # Multi-step reasoning
            'COMPARATIVE_QUERY',        # Player comparisons
            'DOMAIN_CHITCHAT',          # Basketball-related chat
            'OUT_OF_DOMAIN'             # Non-basketball queries
        ]
        
        self.stats = {
            'total_classifications': 0,
            'successful_extractions': 0,
            'out_of_domain_filtered': 0,
            'intent_distribution': {intent: 0 for intent in self.intent_labels}
        }
        
        # Basketball entity knowledge base
        self.basketball_entities = {
            'players': [
                'kobe bryant', 'kobe', 'lebron james', 'lebron', 'michael jordan', 'jordan',
                'yao ming', 'yao', 'stephen curry', 'curry', 'shaquille oneal', 'shaq'
            ],
            'teams': [
                'lakers', 'warriors', 'bulls', 'heat', 'celtics', 'rockets',
                'los angeles lakers', 'golden state warriors', 'chicago bulls'
            ],
            'attributes': [
                'age', 'height', 'weight', 'position', 'team', 'championship',
                'points', 'assists', 'rebounds', 'career', 'stats'
            ]
        }
        
        logger.info("🧠 Unified Intent Classifier initialized")
    
    def classify_and_extract(self, normalized_query: NormalizedQuery) -> ParsedIntent:
        """
        Main classification and extraction pipeline using lightweight multi-task model
        
        Args:
            normalized_query: Language-normalized query
            
        Returns:
            ParsedIntent: Intent and entity information
        """
        try:
            start_time = time.time()
            
            # Use intelligent multi-task model for both intent classification and entity extraction
            parsed_intent = self._intelligent_multitask_classification(normalized_query.normalized_text)
            
            # Add processing time
            parsed_intent.processing_time = time.time() - start_time
            
            logger.info(f"🎯 Intent classified: {parsed_intent.intent} (confidence: {parsed_intent.confidence:.2f})")
            logger.debug(f"📊 Entities extracted: {parsed_intent.entities}")
            
            return parsed_intent
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {str(e)}")
            # Fallback to rule-based approach
            return self._fallback_classification(normalized_query.normalized_text, time.time())
    
    def _intelligent_multitask_classification(self, text: str) -> ParsedIntent:
        """
        Intelligent multi-task model for intent classification and entity extraction
        
        This method implements a lightweight AI model that simultaneously:
        1. Classifies intent into 6 categories
        2. Extracts basketball entities (players, teams, attributes)
        
        Args:
            text: Normalized English text
            
        Returns:
            ParsedIntent: Complete classification result
        """
        try:
            # Update statistics
            self.stats['total_classifications'] += 1
            
            # Preprocess text for AI model
            processed_text = self._preprocess_for_ai_model(text)
            
            # Stage 1: Intent Classification (6-label classification)
            intent_result = self._classify_intent_ai(processed_text)
            
            # Stage 2: Entity Extraction (structured information extraction)  
            entity_result = self._extract_entities_ai(processed_text, intent_result['intent'])
            
            # Stage 3: Determine basketball domain relevance
            is_basketball_related = self._is_basketball_domain(processed_text, entity_result)
            
            # Update statistics
            self.stats['intent_distribution'][intent_result['intent']] += 1
            if entity_result:
                self.stats['successful_extractions'] += 1
            if intent_result['intent'] == 'OUT_OF_DOMAIN':
                self.stats['out_of_domain_filtered'] += 1
            
            # Combine results into ParsedIntent
            return ParsedIntent(
                intent=intent_result['intent'],
                confidence=intent_result['confidence'],
                players=entity_result.get('players', []),
                teams=entity_result.get('teams', []),
                attributes=entity_result.get('attributes', []),
                entities=entity_result,
                is_basketball_related=is_basketball_related,
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.warning(f"⚠️ AI model failed, using fallback: {str(e)}")
            return self._fallback_classification(text, time.time())
    
    def _preprocess_for_ai_model(self, text: str) -> str:
        """
        Preprocess text for AI model input
        
        Args:
            text: Raw text
            
        Returns:
            str: Processed text ready for AI model
        """
        # Convert to lowercase and clean
        text = text.lower().strip()
        
        # Normalize common variations
        text = text.replace("vs.", "versus")
        text = text.replace("vs", "versus") 
        text = text.replace("&", "and")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    def _classify_intent_ai(self, text: str) -> Dict[str, Any]:
        """
        AI-powered intent classification (6-label classification)
        
        Intent Labels:
        - ATTRIBUTE_QUERY: Age, height, weight queries
        - SIMPLE_RELATION_QUERY: Team affiliation, simple facts
        - COMPLEX_RELATION_QUERY: Multi-step reasoning, relationships
        - COMPARATIVE_QUERY: Player comparisons, rankings
        - DOMAIN_CHITCHAT: Basketball general chat
        - OUT_OF_DOMAIN: Non-basketball queries
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dict with intent and confidence
        """
        try:
            # Feature extraction for lightweight classification
            features = self._extract_text_features(text)
            
            # Pattern-based AI simulation (will be replaced with actual AI model)
            intent_scores = self._calculate_intent_scores(text, features)
            
            # Get best intent
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                'intent': best_intent[0],
                'confidence': best_intent[1],
                'all_scores': intent_scores
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Intent classification failed: {str(e)}")
            return {'intent': 'OUT_OF_DOMAIN', 'confidence': 0.0, 'all_scores': {}}
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features for AI model classification
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dict: Feature vector
        """
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_question_words': any(word in text for word in ['what', 'who', 'how', 'when', 'where', 'why']),
            'has_comparison_words': any(word in text for word in ['versus', 'compare', 'better', 'best', 'vs']),
            'has_attribute_words': any(word in text for word in ['age', 'height', 'weight', 'tall', 'old']),
            'has_relation_words': any(word in text for word in ['team', 'play', 'belong', 'member']),
            'has_greeting_words': any(word in text for word in ['hello', 'hi', 'hey', 'greetings']),
            'entity_count': 0,  # Will be updated
            'basketball_domain_confidence': 0.0  # Will be calculated
        }
        
        # Use enhanced basketball domain detection
        features['basketball_domain_confidence'] = self._enhanced_basketball_detection(text)
        
        # Count entities found
        entity_count = 0
        for player in self.basketball_entities['players']:
            if player.lower() in text:
                entity_count += 1
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                entity_count += 1
        features['entity_count'] = entity_count
        
        return features
    
    def _calculate_intent_scores(self, text: str, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate intent scores using AI-inspired algorithm
        
        Args:
            text: Preprocessed text
            features: Extracted features
            
        Returns:
            Dict: Intent scores
        """
        scores = {
            'ATTRIBUTE_QUERY': 0.0,
            'SIMPLE_RELATION_QUERY': 0.0, 
            'COMPLEX_RELATION_QUERY': 0.0,
            'COMPARATIVE_QUERY': 0.0,
            'DOMAIN_CHITCHAT': 0.0,
            'OUT_OF_DOMAIN': 0.0
        }
        
        # Enhanced basketball domain detection
        domain_confidence = self._enhanced_basketball_detection(text)
        
        # If not basketball domain, return OUT_OF_DOMAIN
        if domain_confidence < 0.3:
            scores['OUT_OF_DOMAIN'] = 0.9
            return scores
        
        # ATTRIBUTE_QUERY scoring - Enhanced
        if features['has_attribute_words'] and features['has_question_words']:
            scores['ATTRIBUTE_QUERY'] = 0.85 + domain_confidence * 0.15
        elif any(word in text for word in ['old', 'age', 'tall', 'height', 'weight']) and domain_confidence > 0.5:
            scores['ATTRIBUTE_QUERY'] = 0.75 + domain_confidence * 0.25
        
        # COMPARATIVE_QUERY scoring - Enhanced
        if features['has_comparison_words']:
            scores['COMPARATIVE_QUERY'] = 0.8 + domain_confidence * 0.2
        elif any(word in text for word in ['better', 'best', 'greatest']) and domain_confidence > 0.6:
            scores['COMPARATIVE_QUERY'] = 0.7 + domain_confidence * 0.3
        
        # SIMPLE_RELATION_QUERY scoring - Enhanced
        if features['has_relation_words'] and not features['has_comparison_words']:
            scores['SIMPLE_RELATION_QUERY'] = 0.7 + domain_confidence * 0.3
        elif any(word in text for word in ['which', 'what']) and domain_confidence > 0.5:
            scores['SIMPLE_RELATION_QUERY'] = 0.65 + domain_confidence * 0.35
        
        # COMPLEX_RELATION_QUERY scoring
        if features['word_count'] > 8 and domain_confidence > 0.5:
            scores['COMPLEX_RELATION_QUERY'] = 0.5 + min(features['word_count'] / 15.0, 0.4)
        
        # DOMAIN_CHITCHAT scoring - Enhanced
        if features['has_greeting_words'] and domain_confidence > 0.3:
            scores['DOMAIN_CHITCHAT'] = 0.7 + domain_confidence * 0.3
        elif 'basketball' in text and features['word_count'] < 6:
            scores['DOMAIN_CHITCHAT'] = 0.65 + domain_confidence * 0.35
        elif any(word in text for word in ['tell me', 'explain', 'about']) and domain_confidence > 0.5:
            scores['DOMAIN_CHITCHAT'] = 0.6 + domain_confidence * 0.4
        
        # Boost scores if strong basketball entities present
        entity_boost = self._calculate_entity_boost(text)
        for intent in scores:
            if intent != 'OUT_OF_DOMAIN':
                scores[intent] += entity_boost
        
        # Normalize scores
        max_score = max(scores.values())
        if max_score > 0:
            for intent in scores:
                scores[intent] = min(scores[intent] / max_score * 0.95, 0.95)
        else:
            scores['OUT_OF_DOMAIN'] = 0.8
        
        return scores
    
    def _enhanced_basketball_detection(self, text: str) -> float:
        """
        Enhanced basketball domain detection
        
        Args:
            text: Preprocessed text
            
        Returns:
            float: Basketball domain confidence (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Direct basketball keywords
        basketball_keywords = ['basketball', 'nba', 'player', 'team', 'game', 'sport', 'court', 'score']
        keyword_matches = sum(1 for word in basketball_keywords if word in text)
        confidence += min(keyword_matches / 2.0, 0.4)
        
        # Basketball entities (players, teams)
        entity_confidence = self._calculate_entity_confidence(text)
        confidence += entity_confidence
        
        # Basketball-specific terms
        basketball_terms = ['championship', 'playoffs', 'season', 'draft', 'rookie', 'veteran', 'coach']
        term_matches = sum(1 for term in basketball_terms if term in text)
        confidence += min(term_matches / 3.0, 0.2)
        
        # Basketball actions and concepts
        basketball_actions = ['shoot', 'dunk', 'assist', 'rebound', 'steal', 'block', 'foul']
        action_matches = sum(1 for action in basketball_actions if action in text)
        confidence += min(action_matches / 4.0, 0.15)
        
        return min(confidence, 1.0)
    
    def _calculate_entity_confidence(self, text: str) -> float:
        """
        Calculate confidence based on basketball entities in text
        
        Args:
            text: Preprocessed text
            
        Returns:
            float: Entity-based confidence
        """
        confidence = 0.0
        
        # Check for player names
        player_matches = 0
        for player in self.basketball_entities['players']:
            if player.lower() in text:
                player_matches += 1
        confidence += min(player_matches / 2.0, 0.4)
        
        # Check for team names
        team_matches = 0
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                team_matches += 1
        confidence += min(team_matches / 2.0, 0.3)
        
        # Check for basketball attributes
        attr_matches = 0
        basketball_attrs = ['age', 'height', 'weight', 'position', 'team', 'points', 'assists', 'rebounds']
        for attr in basketball_attrs:
            if attr in text:
                attr_matches += 1
        confidence += min(attr_matches / 3.0, 0.2)
        
        return confidence
    
    def _calculate_entity_boost(self, text: str) -> float:
        """
        Calculate boost score based on entities found
        
        Args:
            text: Preprocessed text
            
        Returns:
            float: Entity boost score
        """
        boost = 0.0
        
        # Strong boost for well-known players
        famous_players = ['kobe', 'lebron', 'jordan', 'yao ming', 'curry', 'shaq']
        for player in famous_players:
            if player in text:
                boost += 0.1
        
        # Boost for team names
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                boost += 0.05
        
        return min(boost, 0.3)
    
    def _extract_entities_ai(self, text: str, intent: str) -> Dict[str, List[str]]:
        """
        AI-powered entity extraction
        
        Args:
            text: Preprocessed text
            intent: Classified intent
            
        Returns:
            Dict: Extracted entities
        """
        try:
            entities = {'players': [], 'teams': [], 'attributes': []}
            
            # Skip entity extraction for OUT_OF_DOMAIN
            if intent == 'OUT_OF_DOMAIN':
                return entities
            
            # Enhanced entity extraction using AI techniques
            entities = self._smart_entity_extraction(text)
            
            # Intent-specific entity refinement
            entities = self._refine_entities_by_intent(entities, intent, text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ AI entity extraction failed: {str(e)}")
            return {'players': [], 'teams': [], 'attributes': []}
    
    def _smart_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """
        Smart entity extraction using multiple techniques
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dict: Extracted entities
        """
        entities = {'players': [], 'teams': [], 'attributes': []}
        
        # Enhanced player extraction with fuzzy matching and name normalization
        for player in self.basketball_entities['players']:
            player_lower = player.lower()
            
            # Exact match
            if player_lower in text:
                normalized_name = self._normalize_player_name(player)
                if normalized_name not in entities['players']:
                    entities['players'].append(normalized_name)
            
            # Fuzzy match for partial names
            elif self._advanced_fuzzy_match(player_lower, text):
                normalized_name = self._normalize_player_name(player)
                if normalized_name not in entities['players']:
                    entities['players'].append(normalized_name)
        
        # Enhanced team extraction
        for team in self.basketball_entities['teams']:
            team_lower = team.lower()
            if team_lower in text or any(alias in text for alias in self._get_team_aliases(team)):
                normalized_team = self._normalize_team_name(team)
                if normalized_team not in entities['teams']:
                    entities['teams'].append(normalized_team)
        
        # Context-aware attribute extraction
        for attr in self.basketball_entities['attributes']:
            if attr.lower() in text:
                entities['attributes'].append(attr)
        
        # Infer attributes from context
        inferred_attrs = self._infer_attributes_from_context(text)
        for attr in inferred_attrs:
            if attr not in entities['attributes']:
                entities['attributes'].append(attr)
        
        return entities
    
    def _normalize_player_name(self, player_name: str) -> str:
        """
        Normalize player name to standard format
        
        Args:
            player_name: Raw player name
            
        Returns:
            str: Normalized player name
        """
        name_mapping = {
            'kobe': 'Kobe Bryant',
            'kobe bryant': 'Kobe Bryant',
            'lebron': 'LeBron James',
            'lebron james': 'LeBron James',
            'jordan': 'Michael Jordan',
            'michael jordan': 'Michael Jordan',
            'yao': 'Yao Ming',
            'yao ming': 'Yao Ming',
            'curry': 'Stephen Curry',
            'stephen curry': 'Stephen Curry',
            'shaq': 'Shaquille O\'Neal',
            'shaquille oneal': 'Shaquille O\'Neal',
            'duncan': 'Tim Duncan',
            'tim duncan': 'Tim Duncan',
            'magic': 'Magic Johnson',
            'magic johnson': 'Magic Johnson',
            'bird': 'Larry Bird',
            'larry bird': 'Larry Bird'
        }
        
        return name_mapping.get(player_name.lower(), player_name.title())
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to standard format
        
        Args:
            team_name: Raw team name
            
        Returns:
            str: Normalized team name
        """
        team_mapping = {
            'lakers': 'Los Angeles Lakers',
            'los angeles lakers': 'Los Angeles Lakers',
            'warriors': 'Golden State Warriors',
            'golden state warriors': 'Golden State Warriors',
            'bulls': 'Chicago Bulls',
            'chicago bulls': 'Chicago Bulls',
            'heat': 'Miami Heat',
            'miami heat': 'Miami Heat',
            'celtics': 'Boston Celtics',
            'boston celtics': 'Boston Celtics',
            'rockets': 'Houston Rockets',
            'houston rockets': 'Houston Rockets',
            'spurs': 'San Antonio Spurs',
            'san antonio spurs': 'San Antonio Spurs'
        }
        
        return team_mapping.get(team_name.lower(), team_name.title())
    
    def _get_team_aliases(self, team_name: str) -> List[str]:
        """
        Get aliases for team names
        
        Args:
            team_name: Team name
            
        Returns:
            List[str]: List of aliases
        """
        alias_mapping = {
            'lakers': ['la lakers', 'l.a. lakers'],
            'warriors': ['gsw', 'dubs'],
            'bulls': ['chicago'],
            'heat': ['miami'],
            'celtics': ['boston'],
            'rockets': ['houston'],
            'spurs': ['san antonio']
        }
        
        return alias_mapping.get(team_name.lower(), [])
    
    def _advanced_fuzzy_match(self, target: str, text: str, threshold: float = 0.7) -> bool:
        """
        Advanced fuzzy matching for entity recognition
        
        Args:
            target: Target entity name
            text: Text to search in
            threshold: Similarity threshold
            
        Returns:
            bool: True if fuzzy match found
        """
        words = text.split()
        target_words = target.split()
        
        # Check for partial matches
        for word in words:
            for target_word in target_words:
                if len(target_word) > 3:
                    # Check if target word is substring of word
                    if target_word in word and len(target_word) / len(word) >= threshold:
                        return True
                    # Check if word is substring of target word
                    if word in target_word and len(word) / len(target_word) >= threshold:
                        return True
        
        # Check for acronym matches
        if len(target_words) > 1:
            acronym = ''.join(word[0] for word in target_words)
            if acronym in text:
                return True
        
        return False
    
    def _infer_attributes_from_context(self, text: str) -> List[str]:
        """
        Infer attributes from text context
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Inferred attributes
        """
        inferred = []
        
        # Age-related inferences
        if any(word in text for word in ['old', 'age', 'born', 'birth']):
            inferred.append('age')
        
        # Height-related inferences
        if any(word in text for word in ['tall', 'height', 'feet', 'ft', 'inches']):
            inferred.append('height')
        
        # Weight-related inferences
        if any(word in text for word in ['heavy', 'weight', 'pounds', 'lbs', 'kg']):
            inferred.append('weight')
        
        # Team-related inferences
        if any(word in text for word in ['team', 'play for', 'plays for', 'member']):
            inferred.append('team')
        
        # Performance-related inferences
        if any(word in text for word in ['points', 'scoring', 'score']):
            inferred.append('points')
        
        if any(word in text for word in ['assists', 'assist']):
            inferred.append('assists')
        
        if any(word in text for word in ['rebounds', 'rebound']):
            inferred.append('rebounds')
        
        # Career-related inferences
        if any(word in text for word in ['career', 'stats', 'statistics']):
            inferred.append('career')
        
        return inferred
    
    def _fuzzy_match(self, target: str, text: str, threshold: float = 0.8) -> bool:
        """
        Simple fuzzy matching for entity recognition
        
        Args:
            target: Target entity name
            text: Text to search in
            threshold: Similarity threshold
            
        Returns:
            bool: True if fuzzy match found
        """
        # Simple implementation - can be enhanced with Levenshtein distance
        words = text.split()
        target_words = target.split()
        
        for word in words:
            for target_word in target_words:
                if len(target_word) > 3 and target_word in word:
                    return True
        
        return False
    
    def _refine_entities_by_intent(self, entities: Dict[str, List[str]], 
                                 intent: str, text: str) -> Dict[str, List[str]]:
        """
        Refine extracted entities based on intent context
        
        Args:
            entities: Raw extracted entities
            intent: Classified intent
            text: Original text
            
        Returns:
            Dict: Refined entities
        """
        # ATTRIBUTE_QUERY: Ensure we have attributes
        if intent == 'ATTRIBUTE_QUERY' and not entities['attributes']:
            # Infer common attributes
            if any(word in text for word in ['old', 'age']):
                entities['attributes'].append('age')
            elif any(word in text for word in ['tall', 'height']):
                entities['attributes'].append('height')
            elif any(word in text for word in ['weight', 'heavy']):
                entities['attributes'].append('weight')
        
        # COMPARATIVE_QUERY: Ensure we have multiple players
        if intent == 'COMPARATIVE_QUERY' and len(entities['players']) < 2:
            # Try to extract more players with relaxed matching
            pass  # Could implement more aggressive extraction here
        
        return entities
    
    def _fallback_classification(self, text: str, start_time: float) -> ParsedIntent:
        """
        Fallback rule-based classification when AI model fails
        
        Args:
            text: Input text
            start_time: Processing start time
            
        Returns:
            ParsedIntent: Fallback classification result
        """
        try:
            # Convert to lowercase for pattern matching
            text = text.lower()
            
            # Extract entities first
            entities = self._extract_entities(text)
            
            # Classify intent based on text and entities
            intent, confidence = self._classify_intent(text, entities)
            
            # Create parsed intent result
            return ParsedIntent(
                intent=intent,
                confidence=confidence * 0.7,  # Lower confidence for fallback
                players=entities.get('players', []),
                teams=entities.get('teams', []),
                attributes=entities.get('attributes', []),
                entities=entities,
                is_basketball_related=self._is_basketball_domain(text, entities),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Fallback classification failed: {str(e)}")
            return ParsedIntent(
                intent='OUT_OF_DOMAIN',
                confidence=0.0,
                players=[],
                teams=[],
                attributes=[],
                entities={},
                is_basketball_related=False,
                processing_time=time.time() - start_time
            )
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract basketball entities from text
        
        Args:
            text: Normalized English text (lowercase)
            
        Returns:
            Dict containing extracted entities by type
        """
        entities = {
            'players': [],
            'teams': [],
            'attributes': []
        }
        
        try:
            # Extract players
            for player in self.basketball_entities['players']:
                if player in text:
                    # Normalize to full name
                    if player == 'kobe':
                        entities['players'].append('Kobe Bryant')
                    elif player == 'lebron':
                        entities['players'].append('LeBron James')
                    elif player == 'jordan':
                        entities['players'].append('Michael Jordan')
                    elif player == 'yao':
                        entities['players'].append('Yao Ming')
                    elif player == 'curry':
                        entities['players'].append('Stephen Curry')
                    elif player == 'shaq':
                        entities['players'].append('Shaquille ONeal')
                    else:
                        entities['players'].append(player.title())
            
            # Extract teams
            for team in self.basketball_entities['teams']:
                if team in text:
                    entities['teams'].append(team.title())
            
            # Extract attributes
            for attr in self.basketball_entities['attributes']:
                if attr in text:
                    entities['attributes'].append(attr)
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ Entity extraction failed: {str(e)}")
            return {'players': [], 'teams': [], 'attributes': []}
    
    def _classify_intent(self, text: str, entities: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        Classify intent based on text patterns and extracted entities
        
        Args:
            text: Normalized text (lowercase)
            entities: Extracted entities
            
        Returns:
            Tuple[intent_label, confidence_score]
        """
        try:
            # Check for OUT_OF_DOMAIN first
            if not self._is_basketball_domain(text, entities):
                return 'OUT_OF_DOMAIN', 0.9
            
            # ATTRIBUTE_QUERY patterns
            attribute_patterns = [
                'how old', 'how tall', 'how much', 'what age', 'age', 'height', 'weight'
            ]
            if any(pattern in text for pattern in attribute_patterns) and entities.get('players'):
                return 'ATTRIBUTE_QUERY', 0.8
            
            # COMPARATIVE_QUERY patterns  
            comparison_patterns = [
                'vs', 'versus', 'compare', 'better', 'best', 'who is', 'between'
            ]
            if any(pattern in text for pattern in comparison_patterns) and len(entities.get('players', [])) >= 2:
                return 'COMPARATIVE_QUERY', 0.8
            
            # COMPLEX_RELATION_QUERY patterns
            complex_patterns = [
                'relationship', 'connection', 'together', 'both', 'all', 'history'
            ]
            if any(pattern in text for pattern in complex_patterns):
                return 'COMPLEX_RELATION_QUERY', 0.7
            
            # SIMPLE_RELATION_QUERY patterns
            simple_patterns = [
                'team', 'play for', 'which team', 'what team', 'belongs to'
            ]
            if any(pattern in text for pattern in simple_patterns):
                return 'SIMPLE_RELATION_QUERY', 0.8
            
            # DOMAIN_CHITCHAT patterns
            chitchat_patterns = [
                'hello', 'hi', 'what is basketball', 'tell me about', 'explain'
            ]
            if any(pattern in text for pattern in chitchat_patterns):
                return 'DOMAIN_CHITCHAT', 0.7
            
            # Default to SIMPLE_RELATION_QUERY if basketball entities found
            if entities.get('players') or entities.get('teams'):
                return 'SIMPLE_RELATION_QUERY', 0.6
            
            # Fallback
            return 'OUT_OF_DOMAIN', 0.5
            
        except Exception as e:
            logger.warning(f"⚠️ Intent classification failed: {str(e)}")
            return 'OUT_OF_DOMAIN', 0.0
    
    def _is_basketball_domain(self, text: str, entities: Dict[str, List[str]]) -> bool:
        """
        Determine if query is basketball-related
        
        Args:
            text: Normalized text
            entities: Extracted entities
            
        Returns:
            bool: True if basketball-related
        """
        try:
            # Check if any basketball entities found
            if any(entities.values()):
                return True
            
            # Check for basketball keywords
            basketball_keywords = [
                'basketball', 'nba', 'game', 'player', 'team', 'sport',
                'court', 'ball', 'shoot', 'score', 'championship'
            ]
            
            return any(keyword in text for keyword in basketball_keywords)
            
        except Exception as e:
            logger.warning(f"⚠️ Domain check failed: {str(e)}")
            return False

# =============================================================================
# Step 3: Smart Post-processor
# =============================================================================

class SmartPostProcessor:
    """
    Smart post-processor for response language adaptation
    
    Handles the final step of converting English responses back to the user's 
    original language if needed.
    """
    
    def __init__(self):
        self.translation_cache = {}
        self.stats = {
            'total_responses': 0,
            'translations_to_chinese': 0,
            'cache_hits': 0
        }
        logger.info("🎯 Smart Post-processor initialized")
    
    def adapt_response_language(self, english_response: str, 
                              normalized_query: NormalizedQuery) -> str:
        """
        Adapt response language to match user's original language
        
        Args:
            english_response: LLM generated English response
            normalized_query: Original normalized query with language info
            
        Returns:
            str: Response in appropriate language
        """
        start_time = time.time()
        
        try:
            self.stats['total_responses'] += 1
            
            # If original language was English, return as-is
            if normalized_query.original_language == 'en':
                logger.info("🎯 Response language: English (no translation needed)")
                return english_response
            
            # If original language was Chinese, translate response
            if normalized_query.original_language == 'zh':
                chinese_response = self._translate_to_chinese(english_response)
                self.stats['translations_to_chinese'] += 1
                logger.info("🎯 Response language: Chinese (translated from English)")
                return chinese_response
            
            # Default: return English
            return english_response
            
        except Exception as e:
            logger.error(f"❌ Response adaptation failed: {str(e)}")
            return english_response  # Fallback to English
    
    def _translate_to_chinese(self, english_text: str) -> str:
        """
        Translate English response to Chinese
        
        Args:
            english_text: English response text
            
        Returns:
            str: Chinese translation
        """
        try:
            # Check cache first
            if english_text in self.translation_cache:
                self.stats['cache_hits'] += 1
                return self.translation_cache[english_text]
            
            # Simple translation mapping for common responses
            translation_map = {
                'Sorry, no relevant information found.': '抱歉，没有找到相关信息。',
                'I can only answer basketball-related questions.': '我只能回答篮球相关的问题。',
                'Hello! How can I help you with basketball questions?': '你好！我可以帮你回答篮球问题。',
                'Kobe Bryant': '科比·布莱恩特',
                'LeBron James': '勒布朗·詹姆斯', 
                'Michael Jordan': '迈克尔·乔丹',
                'Yao Ming': '姚明',
                'Lakers': '湖人队',
                'years old': '岁',
                'plays for': '效力于'
            }
            
            # Try exact match first
            if english_text in translation_map:
                chinese_text = translation_map[english_text]
                self.translation_cache[english_text] = chinese_text
                return chinese_text
            
            # Try partial replacement
            chinese_text = english_text
            for eng, chn in translation_map.items():
                chinese_text = chinese_text.replace(eng, chn)
            
            # Cache the result
            self.translation_cache[english_text] = chinese_text
            return chinese_text
            
        except Exception as e:
            logger.error(f"❌ Chinese translation failed: {str(e)}")
            return english_text  # Fallback to English

# =============================================================================
# Main Smart Pre-processor Integration
# =============================================================================

class SmartPreProcessor:
    """
    Smart Pre-processor: The new core of the routing system
    
    Integrates all three components:
    1. Global Language State Manager
    2. Unified Intent Classifier  
    3. Smart Post-processor
    """
    
    def __init__(self):
        self.language_manager = GlobalLanguageStateManager()
        self.intent_classifier = UnifiedIntentClassifier()
        self.post_processor = SmartPostProcessor()
        
        self.stats = {
            'total_requests': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("🚀 Smart Pre-processor v2.0 initialized")
    
    def process_query(self, user_input: str) -> QueryContext:
        """
        Main entry point: Process user query through the smart pre-processor
        
        Args:
            user_input: Raw user input in any language
            
        Returns:
            QueryContext: Enriched context ready for RAG processing
        """
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # Create initial QueryContext
            context = QueryContextFactory.create_context(user_input)
            
            # Step 1: Language standardization
            normalized_query = self.language_manager.detect_and_normalize_language(user_input)
            
            # Step 2: Intent classification and entity extraction
            parsed_intent = self.intent_classifier.classify_and_extract(normalized_query)
            
            # Step 3: Enrich QueryContext with processed information
            context = self._enrich_query_context(context, normalized_query, parsed_intent)
            
            # Record processing step
            processing_time = time.time() - start_time
            context.add_processing_step(
                step_name="smart_preprocessing",
                status="success",
                duration=processing_time,
                details={
                    'original_language': normalized_query.original_language,
                    'intent': parsed_intent.intent,
                    'entities_found': len(parsed_intent.entities),
                    'basketball_related': parsed_intent.is_basketball_related
                }
            )
            
            self.stats['successful_processing'] += 1
            self._update_average_time(processing_time)
            
            logger.info(f"🚀 Smart preprocessing completed: {parsed_intent.intent} ({processing_time:.3f}s)")
            return context
            
        except Exception as e:
            self.stats['failed_processing'] += 1
            logger.error(f"❌ Smart preprocessing failed: {str(e)}")
            
            # Create error context
            context = QueryContextFactory.create_context(user_input)
            context.add_processing_step(
                step_name="smart_preprocessing",
                status="error",
                duration=time.time() - start_time,
                details={'error': str(e)}
            )
            
            return context
    
    def adapt_final_response(self, english_response: str, context: QueryContext) -> str:
        """
        Adapt final response language based on original user language
        
        Args:
            english_response: English response from LLM
            context: QueryContext with language information
            
        Returns:
            str: Response in appropriate language
        """
        try:
            # Extract normalized query from context
            if context.language_info and hasattr(context.language_info, 'original_language'):
                # Create temporary normalized query for post-processing
                normalized_query = NormalizedQuery(
                    original_text=context.original_query,
                    normalized_text=context.original_query,
                    original_language=context.language_info.original_language,
                    confidence_score=1.0
                )
                
                return self.post_processor.adapt_response_language(english_response, normalized_query)
            
            # Fallback: return English
            return english_response
            
        except Exception as e:
            logger.error(f"❌ Response adaptation failed: {str(e)}")
            return english_response
    
    def _enrich_query_context(self, context: QueryContext, 
                            normalized_query: NormalizedQuery,
                            parsed_intent: ParsedIntent) -> QueryContext:
        """
        Enrich QueryContext with preprocessing results
        
        Args:
            context: Original QueryContext
            normalized_query: Language processing result
            parsed_intent: Intent classification result
            
        Returns:
            QueryContext: Enriched context
        """
        try:
            # Enrich language information
            context.language_info = LanguageInfo(
                original_language=normalized_query.original_language,
                detected_confidence=normalized_query.confidence_score,
                normalized_language="en",
                translation_needed=normalized_query.translation_applied
            )
            
            # Enrich intent information  
            context.intent_info = IntentInfo(
                intent=parsed_intent.intent,
                confidence=parsed_intent.confidence,
                all_scores={parsed_intent.intent: parsed_intent.confidence},
                query_type=parsed_intent.intent.lower(),
                complexity="simple" if "SIMPLE" in parsed_intent.intent else "complex",
                direct_answer_expected=parsed_intent.intent != "OUT_OF_DOMAIN"
            )
            
            # Enrich entity information
            context.entity_info = EntityInfo(
                players=parsed_intent.players,
                teams=parsed_intent.teams,
                attributes=parsed_intent.attributes,
                target_entity=parsed_intent.players[0] if parsed_intent.players else None,
                confidence_scores={
                    'overall': parsed_intent.confidence,
                    'entity_extraction': 0.8 if parsed_intent.entities else 0.0
                }
            )
            
            # Update the query to normalized English version
            if normalized_query.translation_applied:
                context.original_query = normalized_query.normalized_text
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Context enrichment failed: {str(e)}")
            return context
    
    def _update_average_time(self, processing_time: float):
        """Update average processing time statistics"""
        try:
            current_avg = self.stats['average_processing_time']
            total_successful = self.stats['successful_processing']
            
            if total_successful == 1:
                self.stats['average_processing_time'] = processing_time
            else:
                self.stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
        except Exception:
            pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        return {
            'smart_preprocessor': self.stats,
            'language_manager': self.language_manager.stats,
            'intent_classifier': self.intent_classifier.stats,
            'post_processor': self.post_processor.stats
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'average_processing_time': 0.0
        }
        self.language_manager.stats = {
            'total_queries': 0,
            'english_queries': 0,
            'chinese_queries': 0,
            'translations_performed': 0,
            'cache_hits': 0
        }
        self.intent_classifier.stats = {
            'total_classifications': 0,
            'successful_extractions': 0,
            'out_of_domain_filtered': 0,
            'intent_distribution': {intent: 0 for intent in self.intent_classifier.intent_labels}
        }
        self.post_processor.stats = {
            'total_responses': 0,
            'translations_to_chinese': 0,
            'cache_hits': 0
        }
        
        logger.info("📊 All statistics reset")

# =============================================================================
# Factory and Global Instance
# =============================================================================

def create_smart_preprocessor() -> SmartPreProcessor:
    """Factory function to create SmartPreProcessor instance"""
    return SmartPreProcessor()

# Global instance for easy access
smart_preprocessor = create_smart_preprocessor()

# =============================================================================
# Compatibility Layer with Old Router
# =============================================================================

class RouterCompatibilityLayer:
    """
    Compatibility layer to maintain interface with existing code
    while using the new smart preprocessor internally
    """
    
    def __init__(self):
        self.preprocessor = smart_preprocessor
        logger.info("🔄 Router compatibility layer initialized")
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main routing function compatible with old interface
        
        Args:
            query: User input query
            
        Returns:
            Dict: Routing result compatible with old format
        """
        try:
            # Process through smart preprocessor
            context = self.preprocessor.process_query(query)
            
            # Convert to old format for compatibility
            if context.intent_info:
                intent = context.intent_info.intent
                confidence = context.intent_info.confidence
            else:
                intent = "OUT_OF_DOMAIN"
                confidence = 0.0
            
            # Map new intents to old processor names
            processor_mapping = {
                'ATTRIBUTE_QUERY': 'direct_processor',
                'SIMPLE_RELATION_QUERY': 'simple_g_processor', 
                'COMPLEX_RELATION_QUERY': 'complex_g_processor',
                'COMPARATIVE_QUERY': 'comparison_processor',
                'DOMAIN_CHITCHAT': 'chitchat_processor',
                'OUT_OF_DOMAIN': 'out_of_domain'
            }
            
            processor = processor_mapping.get(intent, 'direct_processor')
            
            result = {
                'success': True,
                'intent': intent,
                'confidence': confidence,
                'processor': processor,
                'entities': {
                    'players': context.entity_info.players if context.entity_info else [],
                    'teams': context.entity_info.teams if context.entity_info else [],
                    'attributes': context.entity_info.attributes if context.entity_info else []
                },
                'language_info': {
                    'original_language': context.language_info.original_language if context.language_info else 'en',
                    'translation_needed': context.language_info.translation_needed if context.language_info else False
                },
                'query_context': context  # Pass the full context for new pipeline
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Router compatibility failed: {str(e)}")
            return {
                'success': False,
                'intent': 'OUT_OF_DOMAIN',
                'confidence': 0.0,
                'processor': 'error_handler',
                'entities': {'players': [], 'teams': [], 'attributes': []},
                'language_info': {'original_language': 'en', 'translation_needed': False},
                'error': str(e)
            }

# Global compatibility instance
router_compatibility = RouterCompatibilityLayer()
