"""
Quality Control - Intermediate Level
Advanced prompt engineering and RAG evaluation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class EvaluationMetric:
    """Evaluation metric result"""
    name: str
    score: float
    details: Dict = None


class PromptOptimizer:
    """
    Optimize prompts through iteration
    
    Real-world: A/B test prompts, measure quality
    """
    
    def __init__(self):
        self.prompt_versions: Dict[str, List[Dict]] = {}
    
    def add_version(self, prompt_name: str, version: str, template: str, metrics: Dict):
        """Add prompt version with metrics"""
        if prompt_name not in self.prompt_versions:
            self.prompt_versions[prompt_name] = []
        
        self.prompt_versions[prompt_name].append({
            "version": version,
            "template": template,
            "metrics": metrics
        })
    
    def get_best_version(self, prompt_name: str, metric: str = "accuracy") -> Dict:
        """Get best performing version"""
        if prompt_name not in self.prompt_versions:
            return None
        
        versions = self.prompt_versions[prompt_name]
        best = max(versions, key=lambda v: v["metrics"].get(metric, 0))
        
        return best


class RAGEvaluator:
    """
    Evaluate RAG system quality
    
    Metrics:
    - Context Relevance: Are retrieved docs relevant?
    - Answer Faithfulness: Is answer grounded in context?
    - Answer Relevance: Does answer address the question?
    """
    
    @staticmethod
    def context_relevance(query: str, contexts: List[str]) -> float:
        """
        Measure if retrieved contexts are relevant to query
        
        Simple version: keyword overlap
        Production: Use LLM to judge relevance
        """
        query_words = set(query.lower().split())
        
        relevance_scores = []
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(query_words & context_words)
            score = overlap / len(query_words) if query_words else 0
            relevance_scores.append(score)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    @staticmethod
    def answer_faithfulness(answer: str, contexts: List[str]) -> float:
        """
        Measure if answer is grounded in provided contexts
        
        Checks if answer statements appear in contexts
        """
        # Split answer into sentences
        answer_sentences = re.split(r'[.!?]+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        grounded_count = 0
        
        for sentence in answer_sentences:
            sentence_words = set(sentence.lower().split())
            
            # Check if sentence words appear in any context
            for context in contexts:
                context_words = set(context.lower().split())
                overlap = len(sentence_words & context_words)
                
                # If >50% of words appear in context, consider grounded
                if overlap / len(sentence_words) > 0.5:
                    grounded_count += 1
                    break
        
        return grounded_count / len(answer_sentences)
    
    @staticmethod
    def answer_relevance(query: str, answer: str) -> float:
        """
        Measure if answer addresses the query
        
        Simple version: keyword overlap
        Production: Use LLM to judge relevance
        """
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words & answer_words)
        score = overlap / len(query_words) if query_words else 0
        
        return min(score, 1.0)
    
    def evaluate_rag(
        self,
        query: str,
        contexts: List[str],
        answer: str
    ) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation
        
        Returns all metrics
        """
        return {
            "context_relevance": self.context_relevance(query, contexts),
            "answer_faithfulness": self.answer_faithfulness(answer, contexts),
            "answer_relevance": self.answer_relevance(query, answer)
        }


class PromptTechniques:
    """
    Advanced prompt engineering techniques
    
    Real-world: Improve output quality
    """
    
    @staticmethod
    def chain_of_thought(question: str, examples: List[Tuple[str, str, str]] = None) -> str:
        """
        Chain-of-thought prompting
        
        Forces model to show reasoning steps
        Improves accuracy on complex tasks by 2-3x
        """
        prompt = "Let's solve this step by step.\n\n"
        
        if examples:
            prompt += "Examples:\n\n"
            for q, reasoning, answer in examples:
                prompt += f"Question: {q}\n"
                prompt += f"Reasoning: {reasoning}\n"
                prompt += f"Answer: {answer}\n\n"
        
        prompt += f"Question: {question}\n"
        prompt += "Reasoning:"
        
        return prompt
    
    @staticmethod
    def self_consistency(question: str, num_samples: int = 5) -> str:
        """
        Self-consistency prompting
        
        Generate multiple answers and take majority vote
        Improves accuracy by 10-20%
        """
        prompt = f"""Answer this question {num_samples} times with different reasoning paths.
Then select the most consistent answer.

Question: {question}

Generate {num_samples} different reasoning paths:"""
        
        return prompt
    
    @staticmethod
    def tree_of_thoughts(problem: str, branches: int = 3) -> str:
        """
        Tree-of-thoughts prompting
        
        Explore multiple solution paths
        Best for complex reasoning tasks
        """
        prompt = f"""Let's explore {branches} different approaches to solve this problem.

Problem: {problem}

Approach 1:
[Think through first approach]

Approach 2:
[Think through second approach]

Approach 3:
[Think through third approach]

Now evaluate which approach is best and provide the final answer:"""
        
        return prompt
    
    @staticmethod
    def react_prompting(task: str) -> str:
        """
        ReAct (Reasoning + Acting) prompting
        
        Interleave reasoning and actions
        Used for tool use and agents
        """
        prompt = f"""Task: {task}

Use this format:
Thought: [your reasoning]
Action: [action to take]
Observation: [result of action]
... (repeat as needed)
Thought: [final reasoning]
Answer: [final answer]

Let's begin:
Thought:"""
        
        return prompt


class OutputValidator:
    """
    Validate LLM outputs
    
    Real-world: Catch bad outputs before showing to users
    """
    
    @staticmethod
    def validate_json(output: str, schema: Dict) -> Tuple[bool, List[str]]:
        """Validate JSON output against schema"""
        import json
        
        errors = []
        
        # Try to parse JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {str(e)}"]
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check types
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                actual_type = type(value).__name__
                
                type_map = {
                    "string": "str",
                    "number": ["int", "float"],
                    "boolean": "bool",
                    "array": "list",
                    "object": "dict"
                }
                
                expected = type_map.get(expected_type, expected_type)
                if isinstance(expected, list):
                    if actual_type not in expected:
                        errors.append(f"Field {field}: expected {expected}, got {actual_type}")
                elif actual_type != expected:
                    errors.append(f"Field {field}: expected {expected}, got {actual_type}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_length(output: str, min_length: int = 10, max_length: int = 1000) -> Tuple[bool, str]:
        """Validate output length"""
        length = len(output)
        
        if length < min_length:
            return False, f"Output too short: {length} < {min_length}"
        
        if length > max_length:
            return False, f"Output too long: {length} > {max_length}"
        
        return True, "Length OK"
    
    @staticmethod
    def validate_no_hallucination(output: str, context: str) -> Tuple[bool, float]:
        """
        Check for hallucinations
        
        Simple version: check if output facts appear in context
        """
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(output_words & context_words)
        coverage = overlap / len(output_words) if output_words else 0
        
        # If <30% overlap, likely hallucinating
        is_valid = coverage > 0.3
        
        return is_valid, coverage


if __name__ == "__main__":
    print("=== Prompt Optimization ===\n")
    
    optimizer = PromptOptimizer()
    
    # Add different prompt versions
    optimizer.add_version(
        "sentiment",
        "v1",
        "Classify sentiment: {text}",
        {"accuracy": 0.75, "latency": 0.5}
    )
    
    optimizer.add_version(
        "sentiment",
        "v2",
        "Analyze the sentiment of this text and classify as positive, negative, or neutral: {text}",
        {"accuracy": 0.82, "latency": 0.6}
    )
    
    optimizer.add_version(
        "sentiment",
        "v3",
        "You are a sentiment analysis expert. Carefully analyze: {text}\nSentiment:",
        {"accuracy": 0.88, "latency": 0.7}
    )
    
    best = optimizer.get_best_version("sentiment", metric="accuracy")
    print(f"Best prompt version: {best['version']}")
    print(f"Accuracy: {best['metrics']['accuracy']:.2%}")
    print(f"Template: {best['template'][:50]}...")
    
    print("\n=== RAG Evaluation ===\n")
    
    evaluator = RAGEvaluator()
    
    query = "What is machine learning?"
    contexts = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language."
    ]
    answer = "Machine learning is a type of artificial intelligence that allows systems to learn and improve from data without being explicitly programmed."
    
    metrics = evaluator.evaluate_rag(query, contexts, answer)
    
    print(f"Query: {query}\n")
    print("Metrics:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.2%}")
    
    print("\n=== Advanced Prompt Techniques ===\n")
    
    techniques = PromptTechniques()
    
    # Chain-of-thought
    print("1. Chain-of-Thought:")
    cot_prompt = techniques.chain_of_thought(
        "If a train travels 60 mph for 2.5 hours, how far does it go?"
    )
    print(cot_prompt[:150] + "...\n")
    
    # ReAct
    print("2. ReAct (Reasoning + Acting):")
    react_prompt = techniques.react_prompting(
        "Find the current weather in San Francisco"
    )
    print(react_prompt[:150] + "...\n")
    
    print("\n=== Output Validation ===\n")
    
    validator = OutputValidator()
    
    # Test JSON validation
    json_output = '{"sentiment": "positive", "confidence": 0.95}'
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["sentiment", "confidence"]
    }
    
    valid, errors = validator.validate_json(json_output, schema)
    print(f"JSON validation: {'✓ Valid' if valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Test length validation
    output = "This is a test output."
    valid, message = validator.validate_length(output, min_length=10, max_length=100)
    print(f"\nLength validation: {'✓ Valid' if valid else '✗ Invalid'} - {message}")
    
    # Test hallucination detection
    context = "The sky is blue. Grass is green."
    output1 = "The sky is blue and beautiful."
    output2 = "The ocean is purple and sparkly."
    
    valid1, coverage1 = validator.validate_no_hallucination(output1, context)
    valid2, coverage2 = validator.validate_no_hallucination(output2, context)
    
    print(f"\nHallucination detection:")
    print(f"  Output 1: {'✓ Grounded' if valid1 else '✗ Hallucination'} (coverage: {coverage1:.1%})")
    print(f"  Output 2: {'✓ Grounded' if valid2 else '✗ Hallucination'} (coverage: {coverage2:.1%})")
    
    print("\n=== Key Takeaways ===\n")
    print("1. Iterate on prompts - v3 can be 20%+ better than v1")
    print("2. Measure RAG quality with multiple metrics")
    print("3. Chain-of-thought improves complex reasoning")
    print("4. Validate outputs before showing to users")
    print("5. Detect hallucinations by checking context overlap")
