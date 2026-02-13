from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any
import json
import csv
import os
from datetime import datetime
import shortuuid
from typing import Dict, List, Any


class ARCDataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return super().default(obj)

@dataclass
class ARCTask:
    input_reasoning_chain: List[str]
    transformation_reasoning_chain: List[str]
    task_variables: Dict[str, Any]
    code: str
    data: Dict[str, List[Dict[str, np.ndarray]]]
    generator_name: str = ""

    @staticmethod
    def write_csv_header(filename: str) -> None:
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id',
                    'created_at',
                    'generator_name',
                    'input_reasoning_chain',
                    'transformation_reasoning_chain',
                    'task_variables',
                    'code',
                    'data'
                ])

    def append_to_csv(self, filename: str) -> None:
        """Append task as a new row to CSV file."""
        # Create file with header if it doesn't exist
        self.write_csv_header(filename)

        # Prepare row data
        row = [
            shortuuid.uuid(),
            datetime.now().isoformat(),
            self.generator_name,
            json.dumps(self.input_reasoning_chain),
            json.dumps(self.transformation_reasoning_chain),
            json.dumps(self.task_variables, cls=ARCDataEncoder),
            self.code,
            json.dumps(self.data, cls=ARCDataEncoder)
        ]

        # Append to CSV
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def __str__(self) -> str:
        output = []
        
        # Input Observations
        output.append("Input Observations:")
        for obs in self.input_reasoning_chain:
            output.append(f"- {obs}")

        # Reasoning Steps
        output.append("\nReasoning Steps:")
        for i, step in enumerate(self.transformation_reasoning_chain, 1):
            output.append(f"{i}. {step}")

        # Transform Code
        output.append("\nTransform Input Matrices:")
        output.append(str(self.code))

        # Training Data
        output.append("\nExample Training Data:")
        for i, pair in enumerate(self.data["train"]):
            output.append(f"\nTraining pair {i+1}:")
            output.append("Input:")
            output.append(str(pair["input"]))
            output.append("Output:")
            output.append(str(pair["output"]))

        # Test Data
        output.append("\nTest Data:")
        for i, pair in enumerate(self.data["test"]):
            output.append("Test Input:")
            output.append(str(pair["input"]))
            output.append("Test Output:")
            output.append(str(pair["output"]))

        return "\n".join(output)