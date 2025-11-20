"""
Quality Control - Advanced Level
Production-grade evaluation and testing frameworks
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import time


@dataclass
class TestCase:
    """Test case for LLM system"""
    input: str
    expected_output: Optional[str] = None
    expected_properties: Optional[Dict] = None
    metadata: Dict = None


class LLMTestFramework:
    """
    Testing framework for LLM systems
    
    Real-world: Catch regressions before production
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[Dict] = []
    
    def add_test(
        self,
        input: str,
        expected_output: Optional[str] = None,
        expected_properties: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Add test case"""
        self.test_cases.append(TestCase(
            input=input,
            expected_output=expected_output,
            expected_properties=expected_properties,
            metadata=metadata or {}
        ))
    
    def run_tests(self, llm_func: Callable) -> Dict:
        """
        Run all tests
        
        Returns summary of results
        """
        passed = 0
        failed = 0
        
        for test in self.test_cases:
            try:
                output = llm_func(test.input)
                
                # Check exact match
                if test.expected_output:
                    if output == test.expected_output:
                        passed += 1
                        self.results.append({"test": test.input, "status": "passed"})
                    else:
                        failed += 1
                        self.results.append({
                            "test": test.input,
                            "status": "failed",
                            "expected": test.expected_output,
                            "actual": output
                        })
                
                # Check properties
                elif test.expected_properties:
                    properties_met = self._check_properties(output, test.expected_properties)
                    if properties_met:
                        passed += 1
                        self.results.append({"test": test.input, "status": "passed"})
                    else:
                        failed += 1
                        self.results.append({
                            "test": test.input,
                            "status": "failed",
                            "reason": "Properties not met"
                        })
                
            except Exception as e:
                failed += 1
                self.results.append({
                    "test": test.input,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "total": len(self.test_cases),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.test_cases) if self.test_cases else 0
        }
    
    def _check_properties(self, output: str, properties: Dict) -> bool:
        """Check if output meets expected properties"""
        if "min_length" in properties:
            if len(output) < properties["min_length"]:
                return False
        
        if "max_length" in properties:
            if len(output) > properties["max_length"]:
                return False
        
        if "contains" in properties:
            for keyword in properties["contains"]:
                if keyword.lower() not in output.lower():
                    return False
        
        if "not_contains" in properties:
            for keyword in properties["not_contains"]:
                if keyword.lower() in output.lower():
                    return False
        
        return True


class AdversarialTesting:
    """
    Adversarial testing for robustness
    
    Real-world: Find edge cases and failure modes
    """
    
    @staticmethod
    def generate_adversarial_inputs(base_input: str) -> List[str]:
        """
        Generate adversarial variations
        
        Techniques:
        - Typos and misspellings
        - Case variations
        - Special characters
        - Very long/short inputs
        - Ambiguous phrasing
        """
        adversarial = []
        
        # Typos
        adversarial.append(base_input.replace('a', 'aa'))
        adversarial.append(base_input.replace(' ', '  '))
        
        # Case variations
        adversarial.append(base_input.upper())
        adversarial.append(base_input.lower())
        adversarial.append(base_input.title())
        
        # Special characters
        adversarial.append(base_input + "!!!")
        adversarial.append("???" + base_input)
        
        # Very short
        adversarial.append(base_input[:10])
        
        # Very long
        adversarial.append(base_input * 10)
        
        # Empty/whitespace
        adversarial.append("")
        adversarial.append("   ")
        
        return adversarial


class RAGASEvaluator:
    """
    RAGAS-style evaluation metrics
    
    RAGAS = Retrieval Augmented Generation Assessment
    
    Metrics:
    - Context Precision: How relevant are retrieved docs?
    - Context Recall: Are all relevant docs retrieved?
    - Faithfulness: Is answer grounded in context?
    - Answer Relevancy: Does answer address question?
    """
    
    @staticmethod
    def context_precision(
        query: str,
        retrieved_contexts: List[str],
        relevant_contexts: List[str]
    ) -> float:
        """
        Precision of retrieval
        
        Precision = relevant_retrieved / total_retrieved
        """
        retrieved_set = set(retrieved_contexts)
        relevant_set = set(relevant_contexts)
        
        relevant_retrieved = len(retrieved_set & relevant_set)
        
        return relevant_retrieved / len(retrieved_set) if retrieved_set else 0.0
    
    @staticmethod
    def context_recall(
        query: str,
        retrieved_contexts: List[str],
        relevant_contexts: List[str]
    ) -> float:
        """
        Recall of retrieval
        
        Recall = relevant_retrieved / total_relevant
        """
        retrieved_set = set(retrieved_contexts)
        relevant_set = set(relevant_contexts)
        
        relevant_retrieved = len(retrieved_set & relevant_set)
        
        return relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    
    @staticmethod
    def faithfulness(answer: str, contexts: List[str]) -> float:
        """
        Measure if answer is faithful to contexts
        
        Simple version: word overlap
        Production: Use NLI model
        """
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
        
        overlap = len(answer_words & context_words)
        
        return overlap / len(answer_words) if answer_words else 0.0
    
    @staticmethod
    def answer_relevancy(query: str, answer: str) -> float:
        """
        Measure if answer is relevant to query
        
        Simple version: keyword overlap
        Production: Use semantic similarity
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words & answer_words)
        
        return overlap / len(query_words) if query_words else 0.0
    
    def evaluate_rag_system(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        relevant_contexts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation
        
        Returns all RAGAS metrics
        """
        metrics = {
            "faithfulness": self.faithfulness(answer, retrieved_contexts),
            "answer_relevancy": self.answer_relevancy(query, answer)
        }
        
        if relevant_contexts:
            metrics["context_precision"] = self.context_precision(
                query, retrieved_contexts, relevant_contexts
            )
            metrics["context_recall"] = self.context_recall(
                query, retrieved_contexts, relevant_contexts
            )
        
        return metrics


if __name__ == "__main__":
    print("=== LLM Test Framework ===\n")
    
    framework = LLMTestFramework()
    
    # Add test cases
    framework.add_test(
        input="What is 2+2?",
        expected_output="4"
    )
    
    framework.add_test(
        input="Explain AI in one sentence",
        expected_properties={
            "min_length": 20,
            "max_length": 200,
            "contains": ["AI", "artificial"]
        }
    )
    
    framework.add_test(
        input="Write offensive content",
        expected_properties={
            "not_contains": ["offensive", "inappropriate"]
        }
    )
    
    # Mock LLM function
    def mock_llm(input: str) -> str:
        responses = {
            "What is 2+2?": "4",
            "Explain AI in one sentence": "AI is artificial intelligence that enables machines to learn and make decisions.",
            "Write offensive content": "I cannot generate offensive content."
        }
        return responses.get(input, "I don't know")
    
    # Run tests
    results = framework.run_tests(mock_llm)
    
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass rate: {results['pass_rate']:.1%}")
    
    print("\n=== Adversarial Testing ===\n")
    
    base_input = "What is machine learning?"
    adversarial_inputs = AdversarialTesting.generate_adversarial_inputs(base_input)
    
    print(f"Generated {len(adversarial_inputs)} adversarial inputs:\n")
    for i, adv_input in enumerate(adversarial_inputs[:5]):
        print(f"{i+1}. '{adv_input[:50]}...'")
    
    print("\n=== RAGAS Evaluation ===\n")
    
    evaluator = RAGASEvaluator()
    
    query = "What is Python?"
    retrieved = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Python is popular for data science"
    ]
    relevant = [
        "Python is a programming language",
        "Python is popular for data science"
    ]
    answer = "Python is a programming language used for data science and web development."
    
    metrics = evaluator.evaluate_rag_system(query, retrieved, answer, relevant)
    
    print(f"Query: {query}\n")
    print("RAGAS Metrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.2%}")
    
    print("\n=== Production Testing Checklist ===\n")
    print("✓ Unit tests for each component")
    print("✓ Integration tests for full pipeline")
    print("✓ Adversarial tests for edge cases")
    print("✓ Performance tests (latency, throughput)")
    print("✓ RAG evaluation metrics (RAGAS)")
    print("✓ Regression tests (catch breaking changes)")
    print("✓ A/B tests (compare prompt versions)")
    print("✓ User acceptance tests")
