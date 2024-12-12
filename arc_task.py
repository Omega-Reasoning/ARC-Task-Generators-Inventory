from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any

@dataclass
class ARCTask:
    observation_chain: List[str]
    reasoning_chain: List[str]
    task_variables: Dict[str, Any]
    transform_code: str
    train_test_data: Dict[str, List[Dict[str, np.ndarray]]]

    def __str__(self) -> str:
        output = []
        
        # Input Observations
        output.append("Input Observations:")
        for obs in self.observation_chain:
            output.append(f"- {obs}")

        # Reasoning Steps
        output.append("\nReasoning Steps:")
        for i, step in enumerate(self.reasoning_chain, 1):
            output.append(f"{i}. {step}")

        # Transform Code
        output.append("\nTransform Input Matrices:")
        output.append(str(self.transform_code))

        # Training Data
        output.append("\nExample Training Data:")
        for i, pair in enumerate(self.train_test_data["train"]):
            output.append(f"\nTraining pair {i+1}:")
            output.append("Input:")
            output.append(str(pair["input"]))
            output.append("Output:")
            output.append(str(pair["output"]))

        # Test Data
        output.append("\nTest Data:")
        for i, pair in enumerate(self.train_test_data["test"]):
            output.append("Test Input:")
            output.append(str(pair["input"]))
            output.append("Test Output:")
            output.append(str(pair["output"]))

        return "\n".join(output)