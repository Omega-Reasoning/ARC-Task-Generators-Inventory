from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Any

@dataclass
class ARCTask:
    input_reasoning_chain: List[str]
    transformation_reasoning_chain: List[str]
    task_variables: Dict[str, Any]
    code: str
    data: Dict[str, List[Dict[str, np.ndarray]]]

    @staticmethod
    def write_csv_header(filename: str) -> None:
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id',
                    'created_at',
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

        # Convert numpy arrays to lists for JSON serialization
        data_serializable = {
            set_name: [
                {
                    'input': pair['input'].tolist(),
                    'output': pair['output'].tolist()
                }
                for pair in pairs
            ]
            for set_name, pairs in self.data.items()
        }

        # Prepare row data
        row = [
            shortuuid.uuid(),
            datetime.now().isoformat(),
            json.dumps(self.input_reasoning_chain),
            json.dumps(self.transformation_reasoning_chain),
            json.dumps(self.task_variables),
            json.dumps(self.code),
            json.dumps(data_serializable)
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