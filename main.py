"""
Advanced Prompt Anonymization System
A comprehensive solution for anonymizing sensitive data in prompts before sending to LLMs
Combines multiple detection methods and anonymization strategies
"""

import re
import json
import hashlib
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types of PII entities that can be detected"""
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    ORGANIZATION = "ORGANIZATION"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    CUSTOM = "CUSTOM"

class AnonymizationMethod(Enum):
    """Different methods for anonymizing detected entities"""
    REPLACE = "replace"      # Replace with placeholder
    MASK = "mask"           # Mask with asterisks
    HASH = "hash"           # Replace with hash
    REDACT = "redact"       # Remove completely
    SYNTHETIC = "synthetic"  # Replace with synthetic data

@dataclass
class DetectedEntity:
    """Represents a detected PII entity"""
    entity_type: EntityType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""

@dataclass
class AnonymizationRule:
    """Rules for how to anonymize specific entity types"""
    entity_type: EntityType
    method: AnonymizationMethod
    placeholder: Optional[str] = None
    preserve_format: bool = False

class PIIDetector:
    """Multi-method PII detection engine"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.custom_patterns = {}
        
    def _load_patterns(self) -> Dict[EntityType, List[str]]:
        """Load regex patterns for different PII types"""
        return {
            EntityType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            EntityType.PHONE: [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
                r'\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'
            ],
            EntityType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            EntityType.CREDIT_CARD: [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                r'\b\d{13,19}\b'
            ],
            EntityType.IP_ADDRESS: [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ],
            EntityType.URL: [
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
            ],
            EntityType.DATE: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            ]
        }
    
    def add_custom_pattern(self, entity_type: EntityType, pattern: str):
        """Add custom regex pattern for detection"""
        if entity_type not in self.custom_patterns:
            self.custom_patterns[entity_type] = []
        self.custom_patterns[entity_type].append(pattern)
    
    def detect_entities(self, text: str) -> List[DetectedEntity]:
        """Detect PII entities in text using multiple methods"""
        entities = []
        
        # Pattern-based detection
        entities.extend(self._pattern_detection(text))
        
        # Named entity recognition (simulated - in real implementation, use spaCy/transformers)
        entities.extend(self._ner_detection(text))
        
        # Custom pattern detection
        entities.extend(self._custom_pattern_detection(text))
        
        # Remove overlapping entities and sort by confidence
        entities = self._resolve_overlaps(entities)
        
        return entities
    
    def _pattern_detection(self, text: str) -> List[DetectedEntity]:
        """Detect entities using regex patterns"""
        entities = []
        
        all_patterns = {**self.patterns, **self.custom_patterns}
        
        for entity_type, patterns in all_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = DetectedEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # Pattern-based gets medium confidence
                        context=self._get_context(text, match.start(), match.end())
                    )
                    entities.append(entity)
        
        return entities
    
    def _ner_detection(self, text: str) -> List[DetectedEntity]:
        """Simulate named entity recognition (replace with actual NER model)"""
        entities = []
        
        # Simulate person name detection with simple heuristics
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
        ]
        
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                # Filter out common false positives
                if not self._is_likely_person_name(match.group()):
                    continue
                    
                entity = DetectedEntity(
                    entity_type=EntityType.PERSON,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,  # NER gets lower confidence without proper model
                    context=self._get_context(text, match.start(), match.end())
                )
                entities.append(entity)
        
        return entities
    
    def _custom_pattern_detection(self, text: str) -> List[DetectedEntity]:
        """Apply custom patterns for domain-specific detection"""
        entities = []
        
        # Example: Medical record numbers
        medical_patterns = [
            r'\bMRN[-:]?\s*\d{6,}\b',
            r'\bPatient\s+ID[-:]?\s*\d+\b'
        ]
        
        for pattern in medical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = DetectedEntity(
                    entity_type=EntityType.CUSTOM,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,  # Domain-specific patterns get high confidence
                    context=self._get_context(text, match.start(), match.end())
                )
                entities.append(entity)
        
        return entities
    
    def _is_likely_person_name(self, text: str) -> bool:
        """Filter out common false positives for person names"""
        common_false_positives = {
            "New York", "Los Angeles", "San Francisco", "United States",
            "North America", "South America", "Middle East", "Hong Kong"
        }
        return text not in common_false_positives
    
    def _get_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _resolve_overlaps(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        """Remove overlapping entities, keeping the one with highest confidence"""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        resolved = []
        for entity in entities:
            # Check if it overlaps with any entity in resolved list
            overlaps = False
            for resolved_entity in resolved:
                if (entity.start < resolved_entity.end and 
                    entity.end > resolved_entity.start):
                    # Overlap detected - keep the one with higher confidence
                    if entity.confidence > resolved_entity.confidence:
                        resolved.remove(resolved_entity)
                        resolved.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                resolved.append(entity)
        
        return sorted(resolved, key=lambda x: x.start)

class Anonymizer:
    """Advanced anonymization engine with multiple strategies"""
    
    def __init__(self):
        self.anonymization_map = {}  # For de-anonymization
        self.synthetic_generators = self._setup_synthetic_generators()
    
    def _setup_synthetic_generators(self) -> Dict[EntityType, callable]:
        """Setup synthetic data generators for each entity type"""
        return {
            EntityType.PERSON: lambda: f"Person_{len(self.anonymization_map) + 1}",
            EntityType.EMAIL: lambda: f"user{len(self.anonymization_map) + 1}@example.com",
            EntityType.PHONE: lambda: f"555-{str(len(self.anonymization_map) + 1).zfill(3)}-{str(len(self.anonymization_map) + 1).zfill(4)}",
            EntityType.ORGANIZATION: lambda: f"Organization_{len(self.anonymization_map) + 1}",
            EntityType.ADDRESS: lambda: f"{len(self.anonymization_map) + 1} Main St, City, ST 12345"
        }
    
    def anonymize_text(self, text: str, entities: List[DetectedEntity], 
                      rules: Dict[EntityType, AnonymizationRule]) -> Tuple[str, Dict[str, str]]:
        """Anonymize text based on detected entities and rules"""
        
        # Sort entities by start position in reverse order to maintain positions
        entities_sorted = sorted(entities, key=lambda x: x.start, reverse=True)
        
        anonymized_text = text
        anonymization_map = {}
        
        for entity in entities_sorted:
            rule = rules.get(entity.entity_type)
            if not rule:
                continue  # Skip if no rule defined
            
            replacement = self._generate_replacement(entity, rule)
            
            # Store mapping for potential de-anonymization
            entity_id = f"{entity.entity_type.value}_{hashlib.md5(entity.text.encode()).hexdigest()[:8]}"
            anonymization_map[entity_id] = {
                'original': entity.text,
                'replacement': replacement,
                'entity_type': entity.entity_type.value,
                'method': rule.method.value
            }
            
            # Replace in text
            anonymized_text = (anonymized_text[:entity.start] + 
                             replacement + 
                             anonymized_text[entity.end:])
        
        return anonymized_text, anonymization_map
    
    def _generate_replacement(self, entity: DetectedEntity, rule: AnonymizationRule) -> str:
        """Generate replacement text based on anonymization rule"""
        
        if rule.method == AnonymizationMethod.REPLACE:
            if rule.placeholder:
                return rule.placeholder
            else:
                return f"[{entity.entity_type.value}]"
        
        elif rule.method == AnonymizationMethod.MASK:
            if rule.preserve_format:
                return self._mask_preserving_format(entity.text)
            else:
                return "*" * len(entity.text)
        
        elif rule.method == AnonymizationMethod.HASH:
            return hashlib.sha256(entity.text.encode()).hexdigest()[:16]
        
        elif rule.method == AnonymizationMethod.REDACT:
            return ""
        
        elif rule.method == AnonymizationMethod.SYNTHETIC:
            generator = self.synthetic_generators.get(entity.entity_type)
            if generator:
                return generator()
            else:
                return f"[SYNTHETIC_{entity.entity_type.value}]"
        
        return f"[{entity.entity_type.value}]"  # Default fallback
    
    def _mask_preserving_format(self, text: str) -> str:
        """Mask text while preserving format (e.g., keep dashes in phone numbers)"""
        result = ""
        for char in text:
            if char.isalnum():
                result += "*"
            else:
                result += char
        return result

class PromptAnonymizer:
    """Main anonymization system orchestrating detection and anonymization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.detector = PIIDetector()
        self.anonymizer = Anonymizer()
        self.config = config or self._default_config()
        self.audit_log = []
        
    def _default_config(self) -> Dict:
        """Default configuration for anonymization"""
        return {
            'rules': {
                EntityType.PERSON: AnonymizationRule(
                    EntityType.PERSON, 
                    AnonymizationMethod.SYNTHETIC
                ),
                EntityType.EMAIL: AnonymizationRule(
                    EntityType.EMAIL, 
                    AnonymizationMethod.SYNTHETIC
                ),
                EntityType.PHONE: AnonymizationRule(
                    EntityType.PHONE, 
                    AnonymizationMethod.MASK, 
                    preserve_format=True
                ),
                EntityType.SSN: AnonymizationRule(
                    EntityType.SSN, 
                    AnonymizationMethod.MASK
                ),
                EntityType.CREDIT_CARD: AnonymizationRule(
                    EntityType.CREDIT_CARD, 
                    AnonymizationMethod.MASK
                ),
                EntityType.ADDRESS: AnonymizationRule(
                    EntityType.ADDRESS, 
                    AnonymizationMethod.REPLACE, 
                    placeholder="[ADDRESS]"
                ),
            },
            'confidence_threshold': 0.6,
            'enable_audit': True
        }
    
    def anonymize_prompt(self, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main method to anonymize a prompt"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Detect PII entities
        entities = self.detector.detect_entities(prompt)
        
        # Filter by confidence threshold
        filtered_entities = [
            e for e in entities 
            if e.confidence >= self.config['confidence_threshold']
        ]
        
        # Anonymize text
        anonymized_prompt, anonymization_map = self.anonymizer.anonymize_text(
            prompt, filtered_entities, self.config['rules']
        )
        
        # Create audit log entry
        if self.config['enable_audit']:
            audit_entry = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'original_length': len(prompt),
                'anonymized_length': len(anonymized_prompt),
                'entities_detected': len(entities),
                'entities_anonymized': len(filtered_entities),
                'entity_types': list(set(e.entity_type.value for e in filtered_entities))
            }
            self.audit_log.append(audit_entry)
        
        return {
            'session_id': session_id,
            'original_prompt': prompt,
            'anonymized_prompt': anonymized_prompt,
            'entities_detected': [
                {
                    'type': e.entity_type.value,
                    'text': e.text,
                    'confidence': e.confidence,
                    'context': e.context
                } for e in filtered_entities
            ],
            'anonymization_map': anonymization_map,
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'entities_count': len(filtered_entities),
                'anonymization_methods_used': list(set(
                    self.config['rules'][e.entity_type].method.value 
                    for e in filtered_entities 
                    if e.entity_type in self.config['rules']
                ))
            }
        }
    
    def de_anonymize_response(self, response: str, anonymization_map: Dict[str, str]) -> str:
        """De-anonymize LLM response using the anonymization map"""
        de_anonymized = response
        
        for entity_id, mapping in anonymization_map.items():
            replacement = mapping['replacement']
            original = mapping['original']
            
            # Only de-anonymize if the replacement appears in the response
            if replacement in de_anonymized:
                de_anonymized = de_anonymized.replace(replacement, original)
        
        return de_anonymized
    
    def add_custom_rule(self, entity_type: EntityType, rule: AnonymizationRule):
        """Add custom anonymization rule"""
        self.config['rules'][entity_type] = rule
    
    def add_custom_pattern(self, entity_type: EntityType, pattern: str):
        """Add custom detection pattern"""
        self.detector.add_custom_pattern(entity_type, pattern)
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log for compliance and monitoring"""
        return self.audit_log
    
    def export_config(self) -> str:
        """Export current configuration as JSON"""
        # Convert enum values to strings for JSON serialization
        serializable_config = {
            'confidence_threshold': self.config['confidence_threshold'],
            'enable_audit': self.config['enable_audit'],
            'rules': {
                entity_type.value: {
                    'entity_type': rule.entity_type.value,
                    'method': rule.method.value,
                    'placeholder': rule.placeholder,
                    'preserve_format': rule.preserve_format
                }
                for entity_type, rule in self.config['rules'].items()
            }
        }
        return json.dumps(serializable_config, indent=2)

# Example usage and testing
def demo_usage():
    """Demonstrate the anonymization system"""
    
    # Initialize the anonymizer
    anonymizer = PromptAnonymizer()
    
    # Add custom pattern for employee IDs
    anonymizer.add_custom_pattern(EntityType.CUSTOM, r'\bEMP\d{5}\b')
    
    # Example prompt with various PII
    test_prompt = """
    Hi, I'm John Smith and I work at Acme Corporation. 
    My email is john.smith@acme.com and my phone number is (555) 123-4567.
    My employee ID is EMP12345. I live at 123 Main Street, Anytown, ST 12345.
    My SSN is 123-45-6789 and my credit card is 4532-1234-5678-9012.
    Please help me with my account registered on 03/15/2023.
    """
    
    print("Original Prompt:")
    print(test_prompt)
    print("\n" + "="*80 + "\n")
    
    # Anonymize the prompt
    result = anonymizer.anonymize_prompt(test_prompt)
    
    print("Anonymized Prompt:")
    print(result['anonymized_prompt'])
    print("\n" + "="*80 + "\n")
    
    print("Detected Entities:")
    for entity in result['entities_detected']:
        print(f"- {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")
    
    print("\n" + "="*80 + "\n")
    
    # Simulate LLM response that uses anonymized data
    mock_llm_response = "Hello Person_1! I can help you with your account at Organization_1. Please verify your contact at user1@example.com."
    
    print("Mock LLM Response:")
    print(mock_llm_response)
    print("\n" + "="*80 + "\n")
    
    # De-anonymize the response
    de_anonymized_response = anonymizer.de_anonymize_response(
        mock_llm_response, 
        result['anonymization_map']
    )
    
    print("De-anonymized Response:")
    print(de_anonymized_response)
    print("\n" + "="*80 + "\n")
    
    print("Audit Log:")
    for entry in anonymizer.get_audit_log():
        print(f"Session: {entry['session_id'][:8]}... | Entities: {entry['entities_anonymized']} | Types: {entry['entity_types']}")

if __name__ == "__main__":
    demo_usage()