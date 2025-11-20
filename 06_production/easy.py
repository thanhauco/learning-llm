"""
Production - Easy Level
Basic guardrails, PII detection, and error handling
"""

import re
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class ModerationResult:
    """Content moderation result"""
    is_safe: bool
    flagged_categories: List[str]
    confidence: float
    reason: str = ""


class ContentModerator:
    """
    Basic content moderation
    
    Real-world: Use OpenAI Moderation API or Perspective API
    """
    
    # Simple keyword-based moderation (production uses ML models)
    UNSAFE_PATTERNS = {
        "violence": [r"\bkill\b", r"\bharm\b", r"\battack\b"],
        "hate": [r"\bhate\b", r"racist", r"sexist"],
        "sexual": [r"explicit sexual content patterns"],
        "self-harm": [r"suicide", r"self-harm"]
    }
    
    def moderate(self, text: str) -> ModerationResult:
        """
        Moderate content
        
        Returns whether content is safe and why
        """
        text_lower = text.lower()
        flagged = []
        
        for category, patterns in self.UNSAFE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    flagged.append(category)
                    break
        
        is_safe = len(flagged) == 0
        
        return ModerationResult(
            is_safe=is_safe,
            flagged_categories=flagged,
            confidence=0.9 if flagged else 0.95,
            reason=f"Flagged: {', '.join(flagged)}" if flagged else "Content is safe"
        )


class PIIDetector:
    """
    Detect and scrub PII (Personally Identifiable Information)
    
    Real-world: Protect user privacy, comply with GDPR/CCPA
    """
    
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    }
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        found = {}
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches
        
        return found
    
    def scrub(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Remove PII from text"""
        scrubbed = text
        
        for pii_type, pattern in self.PATTERNS.items():
            scrubbed = re.sub(pattern, f"[{pii_type.upper()}]", scrubbed)
        
        return scrubbed
    
    def anonymize(self, text: str) -> tuple[str, Dict]:
        """
        Anonymize PII with reversible mapping
        
        Use when: Need to de-anonymize later
        """
        mapping = {}
        anonymized = text
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            for match in matches:
                # Create deterministic hash
                hash_val = hashlib.md5(match.encode()).hexdigest()[:8]
                placeholder = f"[{pii_type.upper()}_{hash_val}]"
                
                mapping[placeholder] = match
                anonymized = anonymized.replace(match, placeholder)
        
        return anonymized, mapping


class SimpleLogger:
    """
    Simple logging for LLM applications
    
    Real-world: Use structured logging (e.g., structlog, loguru)
    """
    
    def __init__(self, app_name: str = "llm-app"):
        self.app_name = app_name
        self.logs = []
    
    def log(self, level: str, message: str, **kwargs):
        """Log message with metadata"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "app": self.app_name,
            "level": level,
            "message": message,
            **kwargs
        }
        
        self.logs.append(log_entry)
        print(f"[{level}] {message}")
    
    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict]:
        """Get logs, optionally filtered by level"""
        if level:
            return [log for log in self.logs if log["level"] == level]
        return self.logs


class RetryHandler:
    """
    Retry failed API calls with exponential backoff
    
    Real-world: Handle transient failures gracefully
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def retry(self, func: Callable, *args, **kwargs):
        """
        Retry function with exponential backoff
        
        Delay pattern: 1s, 2s, 4s, 8s, ...
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        raise last_exception


if __name__ == "__main__":
    print("=== Content Moderation ===\n")
    
    moderator = ContentModerator()
    
    test_texts = [
        "This is a safe message about AI",
        "I want to harm someone",
        "This contains hate speech"
    ]
    
    for text in test_texts:
        result = moderator.moderate(text)
        print(f"Text: {text}")
        print(f"  Safe: {result.is_safe}")
        print(f"  Flagged: {result.flagged_categories}")
        print(f"  Reason: {result.reason}")
        print()
    
    print("\n=== PII Detection ===\n")
    
    detector = PIIDetector()
    
    text_with_pii = """
    Contact me at john.doe@email.com or call 555-123-4567.
    My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456.
    """
    
    print("Original text:")
    print(text_with_pii)
    
    # Detect PII
    found_pii = detector.detect(text_with_pii)
    print("\nDetected PII:")
    for pii_type, matches in found_pii.items():
        print(f"  {pii_type}: {matches}")
    
    # Scrub PII
    scrubbed = detector.scrub(text_with_pii)
    print("\nScrubbed text:")
    print(scrubbed)
    
    # Anonymize (reversible)
    anonymized, mapping = detector.anonymize(text_with_pii)
    print("\nAnonymized text:")
    print(anonymized)
    print("\nMapping (for de-anonymization):")
    for placeholder, original in mapping.items():
        print(f"  {placeholder} -> {original}")
    
    print("\n=== Logging ===\n")
    
    logger = SimpleLogger("my-llm-app")
    
    logger.info("Application started")
    logger.info("Processing query", user_id="user123", query="What is AI?")
    logger.warning("High latency detected", latency_ms=2500)
    logger.error("API call failed", error="Rate limit exceeded")
    
    print("\nError logs:")
    for log in logger.get_logs("ERROR"):
        print(f"  {log['timestamp']}: {log['message']}")
    
    print("\n=== Retry Handler ===\n")
    
    retry_handler = RetryHandler(max_retries=3, base_delay=0.5)
    
    # Simulate flaky API
    attempt_count = 0
    def flaky_api():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception("API temporarily unavailable")
        return "Success!"
    
    try:
        result = retry_handler.retry(flaky_api)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    print("\n=== Production Checklist ===\n")
    print("✓ Content moderation (prevent harmful outputs)")
    print("✓ PII detection (protect user privacy)")
    print("✓ Structured logging (debug issues)")
    print("✓ Retry logic (handle transient failures)")
    print("✓ Rate limiting (stay within API limits)")
    print("✓ Monitoring (track performance)")
    print("✓ Error handling (graceful degradation)")
    print("✓ Testing (catch issues before production)")
