"""
MLOps & Deployment - Easy Level
Model deployment basics and serverless concepts
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    model_path: str
    created_at: str
    metrics: Dict = field(default_factory=dict)
    status: str = "active"  # active, deprecated, archived


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_version: str
    replicas: int = 1
    gpu_type: str = "T4"
    min_replicas: int = 1
    max_replicas: int = 10
    target_latency_ms: float = 1000


class ModelRegistry:
    """
    Model registry for versioning
    
    Real-world: Use MLflow, Weights & Biases, or HuggingFace Hub
    """
    
    def __init__(self):
        self.models: Dict[str, List[ModelVersion]] = {}
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict = None
    ) -> ModelVersion:
        """Register a new model version"""
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {}
        )
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        self.models[model_name].append(model_version)
        
        print(f"Registered {model_name} v{version}")
        return model_version
    
    def get_model(self, model_name: str, version: str = "latest") -> Optional[ModelVersion]:
        """Get specific model version"""
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        if version == "latest":
            return versions[-1]
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model"""
        if model_name not in self.models:
            return []
        
        return [v.version for v in self.models[model_name]]


class AutoScaler:
    """
    Auto-scaling based on load
    
    Real-world: Kubernetes HPA, AWS Auto Scaling
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_replicas = config.replicas
        self.request_queue_size = 0
    
    def scale_decision(self, current_latency: float, queue_size: int) -> int:
        """
        Decide whether to scale up or down
        
        Scale up if:
        - Latency > target
        - Queue is growing
        
        Scale down if:
        - Latency < target / 2
        - Queue is empty
        """
        target_latency = self.config.target_latency_ms
        
        # Scale up
        if current_latency > target_latency or queue_size > 10:
            new_replicas = min(
                self.current_replicas + 1,
                self.config.max_replicas
            )
            if new_replicas > self.current_replicas:
                print(f"Scaling UP: {self.current_replicas} → {new_replicas} replicas")
                self.current_replicas = new_replicas
        
        # Scale down
        elif current_latency < target_latency / 2 and queue_size == 0:
            new_replicas = max(
                self.current_replicas - 1,
                self.config.min_replicas
            )
            if new_replicas < self.current_replicas:
                print(f"Scaling DOWN: {self.current_replicas} → {new_replicas} replicas")
                self.current_replicas = new_replicas
        
        return self.current_replicas


class ABTestManager:
    """
    A/B testing for model deployments
    
    Real-world: Test new models before full rollout
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
    
    def create_experiment(
        self,
        name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5
    ):
        """
        Create A/B test
        
        Args:
            traffic_split: Fraction of traffic to model_b (0.0 to 1.0)
        """
        self.experiments[name] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "metrics_a": {"requests": 0, "latency": [], "errors": 0},
            "metrics_b": {"requests": 0, "latency": [], "errors": 0}
        }
        
        print(f"Created A/B test: {name}")
        print(f"  Model A: {model_a} ({(1-traffic_split)*100:.0f}% traffic)")
        print(f"  Model B: {model_b} ({traffic_split*100:.0f}% traffic)")
    
    def route_request(self, experiment_name: str) -> str:
        """Route request to A or B based on traffic split"""
        import random
        
        exp = self.experiments[experiment_name]
        
        if random.random() < exp["traffic_split"]:
            return exp["model_b"]
        else:
            return exp["model_a"]
    
    def record_metrics(
        self,
        experiment_name: str,
        model: str,
        latency: float,
        error: bool = False
    ):
        """Record metrics for A/B test"""
        exp = self.experiments[experiment_name]
        
        if model == exp["model_a"]:
            metrics = exp["metrics_a"]
        else:
            metrics = exp["metrics_b"]
        
        metrics["requests"] += 1
        metrics["latency"].append(latency)
        if error:
            metrics["errors"] += 1
    
    def get_results(self, experiment_name: str) -> Dict:
        """Get A/B test results"""
        import numpy as np
        
        exp = self.experiments[experiment_name]
        
        metrics_a = exp["metrics_a"]
        metrics_b = exp["metrics_b"]
        
        return {
            "model_a": {
                "requests": metrics_a["requests"],
                "avg_latency": np.mean(metrics_a["latency"]) if metrics_a["latency"] else 0,
                "error_rate": metrics_a["errors"] / metrics_a["requests"] if metrics_a["requests"] > 0 else 0
            },
            "model_b": {
                "requests": metrics_b["requests"],
                "avg_latency": np.mean(metrics_b["latency"]) if metrics_b["latency"] else 0,
                "error_rate": metrics_b["errors"] / metrics_b["requests"] if metrics_b["requests"] > 0 else 0
            }
        }


class DeploymentPlatformComparison:
    """
    Compare different deployment platforms
    
    Real-world: Choose based on your needs
    """
    
    @staticmethod
    def compare_platforms():
        """Compare deployment options"""
        platforms = {
            "Self-hosted (Kubernetes)": {
                "cost": "High (infrastructure + maintenance)",
                "control": "Full control",
                "scaling": "Manual/HPA",
                "setup_time": "Days to weeks",
                "best_for": "Large scale, custom requirements"
            },
            "BentoML": {
                "cost": "Medium (infrastructure only)",
                "control": "High control",
                "scaling": "Built-in auto-scaling",
                "setup_time": "Hours",
                "best_for": "Production ML serving"
            },
            "Modal": {
                "cost": "Pay per use ($0.0001/s GPU)",
                "control": "Medium control",
                "scaling": "Automatic (0 to 1000s)",
                "setup_time": "Minutes",
                "best_for": "Serverless, variable load"
            },
            "Replicate": {
                "cost": "Pay per use + markup",
                "control": "Low control",
                "scaling": "Automatic",
                "setup_time": "Minutes",
                "best_for": "Quick deployment, demos"
            },
            "HuggingFace Inference": {
                "cost": "Pay per use",
                "control": "Low control",
                "scaling": "Automatic",
                "setup_time": "Seconds",
                "best_for": "Standard models, prototyping"
            }
        }
        
        return platforms


if __name__ == "__main__":
    print("=== Model Registry ===\n")
    
    registry = ModelRegistry()
    
    # Register model versions
    registry.register_model(
        "sentiment-classifier",
        "v1.0",
        "s3://models/sentiment-v1.0",
        metrics={"accuracy": 0.85, "f1": 0.83}
    )
    
    registry.register_model(
        "sentiment-classifier",
        "v1.1",
        "s3://models/sentiment-v1.1",
        metrics={"accuracy": 0.88, "f1": 0.86}
    )
    
    # Get latest version
    latest = registry.get_model("sentiment-classifier", "latest")
    print(f"\nLatest version: {latest.version}")
    print(f"Metrics: {latest.metrics}")
    
    # List all versions
    versions = registry.list_versions("sentiment-classifier")
    print(f"All versions: {versions}")
    
    print("\n=== Auto-Scaling ===\n")
    
    config = DeploymentConfig(
        model_version="v1.1",
        replicas=2,
        min_replicas=1,
        max_replicas=5,
        target_latency_ms=500
    )
    
    scaler = AutoScaler(config)
    
    # Simulate load scenarios
    scenarios = [
        {"latency": 800, "queue": 15, "desc": "High load"},
        {"latency": 900, "queue": 20, "desc": "Very high load"},
        {"latency": 600, "queue": 5, "desc": "Medium load"},
        {"latency": 200, "queue": 0, "desc": "Low load"},
        {"latency": 150, "queue": 0, "desc": "Very low load"},
    ]
    
    for scenario in scenarios:
        print(f"{scenario['desc']}:")
        print(f"  Latency: {scenario['latency']}ms, Queue: {scenario['queue']}")
        scaler.scale_decision(scenario['latency'], scenario['queue'])
        print()
    
    print("=== A/B Testing ===\n")
    
    ab_test = ABTestManager()
    
    # Create experiment
    ab_test.create_experiment(
        "model_comparison",
        model_a="v1.0",
        model_b="v1.1",
        traffic_split=0.2  # 20% to new model
    )
    
    # Simulate requests
    print("\nSimulating 100 requests...")
    for i in range(100):
        model = ab_test.route_request("model_comparison")
        
        # Simulate latency (v1.1 is faster)
        if model == "v1.0":
            latency = 500 + np.random.randn() * 50
        else:
            latency = 400 + np.random.randn() * 40
        
        ab_test.record_metrics("model_comparison", model, latency)
    
    # Get results
    results = ab_test.get_results("model_comparison")
    print("\nA/B Test Results:")
    print(f"Model v1.0:")
    print(f"  Requests: {results['model_a']['requests']}")
    print(f"  Avg latency: {results['model_a']['avg_latency']:.1f}ms")
    print(f"  Error rate: {results['model_a']['error_rate']:.2%}")
    
    print(f"\nModel v1.1:")
    print(f"  Requests: {results['model_b']['requests']}")
    print(f"  Avg latency: {results['model_b']['avg_latency']:.1f}ms")
    print(f"  Error rate: {results['model_b']['error_rate']:.2%}")
    
    if results['model_b']['avg_latency'] < results['model_a']['avg_latency']:
        print("\n✓ Model v1.1 is faster! Consider full rollout.")
    
    print("\n=== Platform Comparison ===\n")
    
    platforms = DeploymentPlatformComparison.compare_platforms()
    
    for name, details in platforms.items():
        print(f"{name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
        print()
    
    print("=== Deployment Checklist ===\n")
    print("✓ Model versioning and registry")
    print("✓ Auto-scaling based on load")
    print("✓ A/B testing for new models")
    print("✓ Monitoring and alerting")
    print("✓ Rollback capability")
    print("✓ Cost optimization")
    print("✓ Security and authentication")
    print("✓ CI/CD pipeline")
    
    print("\n=== Cost Comparison ===\n")
    print("Example: Serving LLaMA 7B")
    print("\nSelf-hosted (A100 80GB):")
    print("  - Cost: $3/hour = $2,160/month")
    print("  - Utilization: 24/7")
    print("  - Best for: Consistent high load")
    
    print("\nModal (Serverless):")
    print("  - Cost: $0.0001/second GPU")
    print("  - Utilization: Pay only when used")
    print("  - Example: 10% utilization = $216/month")
    print("  - Best for: Variable/bursty load")
    
    print("\nReplicate:")
    print("  - Cost: ~$0.0002/second (includes markup)")
    print("  - No infrastructure management")
    print("  - Best for: Quick deployment, demos")
