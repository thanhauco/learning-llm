"""
Quality Control - Easy Level
Prompt engineering basics and structured outputs
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Reusable prompt template"""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs) -> str:
        """Format template with variables"""
        return self.template.format(**kwargs)


class PromptPatterns:
    """
    Common prompt engineering patterns
    
    Real-world: Improve output quality and consistency
    """
    
    @staticmethod
    def zero_shot(task: str, input_text: str) -> str:
        """
        Zero-shot prompting
        
        Use when: Task is simple and well-defined
        """
        return f"""{task}

Input: {input_text}

Output:"""
    
    @staticmethod
    def few_shot(task: str, examples: List[tuple], input_text: str) -> str:
        """
        Few-shot prompting
        
        Use when: Need to show format or style
        Real-world benefit: 30-50% better accuracy
        """
        examples_str = "\n\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])
        
        return f"""{task}

Examples:
{examples_str}

Input: {input_text}

Output:"""
    
    @staticmethod
    def chain_of_thought(task: str, input_text: str) -> str:
        """
        Chain-of-thought prompting
        
        Use when: Task requires reasoning
        Real-world benefit: 2-3x better on complex tasks
        """
        return f"""{task}

Let's think step by step:

Input: {input_text}

Reasoning:"""
    
    @staticmethod
    def structured_output(task: str, schema: Dict, input_text: str) -> str:
        """
        Structured output prompting
        
        Use when: Need JSON or specific format
        """
        schema_str = json.dumps(schema, indent=2)
        
        return f"""{task}

Output must be valid JSON matching this schema:
{schema_str}

Input: {input_text}

JSON Output:"""


class JSONOutputParser:
    """
    Parse and validate JSON outputs from LLMs
    
    Real-world: LLMs often generate invalid JSON
    """
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        """
        Extract JSON from text
        
        Handles common issues:
        - JSON wrapped in markdown code blocks
        - Extra text before/after JSON
        - Single quotes instead of double quotes
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None
        
        json_str = json_match.group()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common issues
            json_str = json_str.replace("'", '"')
            try:
                return json.loads(json_str)
            except:
                return None
    
    @staticmethod
    def validate_schema(data: Dict, schema: Dict) -> tuple[bool, List[str]]:
        """
        Validate JSON against schema
        
        Simple validation - in production use jsonschema library
        """
        errors = []
        
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                actual_type = type(value).__name__
                
                type_map = {
                    'string': 'str',
                    'number': ['int', 'float'],
                    'boolean': 'bool',
                    'array': 'list',
                    'object': 'dict'
                }
                
                expected = type_map.get(expected_type, expected_type)
                if isinstance(expected, list):
                    if actual_type not in expected:
                        errors.append(f"Field {field}: expected {expected}, got {actual_type}")
                elif actual_type != expected:
                    errors.append(f"Field {field}: expected {expected}, got {actual_type}")
        
        return len(errors) == 0, errors


class PromptTester:
    """
    Test prompts with different inputs
    
    Real-world: Catch edge cases before production
    """
    
    def __init__(self):
        self.test_cases = []
    
    def add_test(self, input_text: str, expected_output: str, description: str = ""):
        """Add test case"""
        self.test_cases.append({
            "input": input_text,
            "expected": expected_output,
            "description": description
        })
    
    def run_tests(self, prompt_func, llm_func) -> Dict:
        """
        Run all tests
        
        Args:
            prompt_func: Function that generates prompt
            llm_func: Function that calls LLM (mocked in this example)
        """
        results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for test in self.test_cases:
            prompt = prompt_func(test["input"])
            # In production, call actual LLM here
            # output = llm_func(prompt)
            
            # For demo, we'll just check prompt format
            if len(prompt) > 0:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(test["description"])
        
        return results


if __name__ == "__main__":
    print("=== Prompt Patterns ===\n")
    
    # Zero-shot
    print("1. Zero-shot:")
    prompt = PromptPatterns.zero_shot(
        "Classify the sentiment as positive, negative, or neutral.",
        "I love this product!"
    )
    print(prompt)
    print()
    
    # Few-shot
    print("2. Few-shot:")
    examples = [
        ("I love this!", "positive"),
        ("This is terrible.", "negative"),
        ("It's okay.", "neutral")
    ]
    prompt = PromptPatterns.few_shot(
        "Classify sentiment:",
        examples,
        "This is amazing!"
    )
    print(prompt[:200] + "...")
    print()
    
    # Chain-of-thought
    print("3. Chain-of-thought:")
    prompt = PromptPatterns.chain_of_thought(
        "Solve this math problem:",
        "If John has 5 apples and gives 2 to Mary, how many does he have left?"
    )
    print(prompt)
    print()
    
    # Structured output
    print("4. Structured output:")
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"}
        },
        "required": ["sentiment", "confidence"]
    }
    prompt = PromptPatterns.structured_output(
        "Analyze sentiment:",
        schema,
        "This product exceeded my expectations!"
    )
    print(prompt[:300] + "...")
    print()
    
    print("\n=== JSON Output Parsing ===\n")
    
    # Test JSON extraction
    test_outputs = [
        '```json\n{"sentiment": "positive", "confidence": 0.95}\n```',
        'Here is the result: {"sentiment": "negative", "confidence": 0.8}',
        "{'sentiment': 'neutral', 'confidence': 0.5}",  # Single quotes
    ]
    
    parser = JSONOutputParser()
    
    for i, output in enumerate(test_outputs):
        print(f"Test {i+1}:")
        print(f"  Raw: {output[:50]}...")
        extracted = parser.extract_json(output)
        print(f"  Extracted: {extracted}")
        
        if extracted:
            valid, errors = parser.validate_schema(extracted, schema)
            print(f"  Valid: {valid}")
            if errors:
                print(f"  Errors: {errors}")
        print()
    
    print("\n=== Prompt Testing ===\n")
    
    tester = PromptTester()
    
    # Add test cases
    tester.add_test(
        "I love this!",
        "positive",
        "Simple positive sentiment"
    )
    tester.add_test(
        "This is terrible.",
        "negative",
        "Simple negative sentiment"
    )
    tester.add_test(
        "",
        "neutral",
        "Empty input edge case"
    )
    
    def mock_llm(prompt):
        return "positive"
    
    results = tester.run_tests(
        lambda inp: PromptPatterns.zero_shot("Classify sentiment", inp),
        mock_llm
    )
    
    print(f"Tests passed: {results['passed']}")
    print(f"Tests failed: {results['failed']}")
    
    print("\n=== Best Practices ===\n")
    print("1. Be specific and clear")
    print("2. Use examples for complex tasks")
    print("3. Request structured output (JSON)")
    print("4. Add validation and error handling")
    print("5. Test with edge cases")
    print("6. Version your prompts")
    print("7. Monitor output quality in production")
