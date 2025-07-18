import argparse
import csv
import os
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from execution import get_generator_class, execute_transform_code
from utilities import visualize_grid

class ARCGeneratorTestRunner:
    def __init__(self, csv_file: str, generator_file: str = None):
        self.csv_file = csv_file
        self.generator_file = generator_file
        
    def parse_grid(self, grid_str: str) -> np.ndarray:
        """Convert a JSON string representation of a grid to numpy array."""
        try:
            grid = json.loads(grid_str)
            return np.array(grid)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid grid format: {grid_str}")

    def parse_taskvars(self, taskvars_str: str) -> Dict[str, Any]:
        """Convert comma-separated key:value pairs to dict."""
        result = {}
        # Split by comma and strip whitespace
        pairs = [pair.strip() for pair in taskvars_str.split(',')]
        
        for pair in pairs:
            if not pair:  # Skip empty pairs
                continue
            # Split by : and strip whitespace
            key, value = [x.strip() for x in pair.split(':')]
            
            # Try to convert value to int if possible
            try:
                value = int(value)
            except ValueError:
                # If conversion fails, keep as string
                pass
                
            result[key] = value
            
        return result

    def run_test(self, generator_file: str, taskvars: str, input_grid: str, output_grid: str) -> bool:
        """Run a single test case."""
        try:
            # Parse input and output grids
            input_array = self.parse_grid(input_grid)
            expected_output = self.parse_grid(output_grid)
            task_variables = self.parse_taskvars(taskvars)

            print(f"\nRunning {generator_file}:")
            print(f"\tTask variables: {task_variables}")

            # Instantiate generator
            try:
                generator_class = get_generator_class(generator_file)
                generator = generator_class()
            except Exception as e:
                print("Error during generator instantiation:")
                import traceback
                print(traceback.format_exc())
                return False

            # Get transformation function
            try:
                transform_code_str = generator._partial_evaluation_code(
                    generator.transform_input, 
                    task_variables
                )

                # Check for remaining taskvars and print error message
                pattern = r"taskvars\[(['\"])(.*?)\1\]"
                matches = re.findall(pattern, transform_code_str)

                if matches:
                    referenced_vars = [match[1] for match in matches]  
                    print("\tWARNING: Instantiated transform code references the following taskvars:", 
                        ", ".join(referenced_vars))

                print("\tSuccessfully created transform_code")
                # print(transform_code_str)
                
                transform_code = execute_transform_code(transform_code_str, generator)
                
            except Exception as e:
                print("Error during transform code creation:")
                import traceback
                print(f"Instantiated transform code: {transform_code}")
                print(traceback.format_exc())
                return False

            # Apply transformation
            try:
                actual_output = transform_code(generator, input_array)
            except Exception as e:
                print("Error during transform execution:")
                print(f"Transform code type: {type(transform_code)}")
                print(f"Taskvars: {task_variables}")
                print(f"Instantiated transform code: {transform_code_str}")
                print(traceback.format_exc())
                return False

            # Compare results
            if not np.array_equal(actual_output, expected_output):
                print(f"\nTest failed for generator: {generator_file}")
                print("Task variables:", task_variables)
                print("\nInput grid:")
                print(visualize_grid(input_array))
                print("\nExpected output:")
                print(visualize_grid(expected_output))
                print("\nActual output:")
                print(visualize_grid(actual_output))
                return False

            return True

        except Exception as e:
            print(f"\nUnexpected error in test for generator {generator_file}:")
            import traceback
            print(traceback.format_exc())
            return False

    def run_all_tests(self) -> tuple[int, int]:
        """Run all tests from the CSV file."""
        total_tests = 0
        passed_tests = 0

        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Skip if generator_file filter is set and doesn't match
                    if self.generator_file and row['generator_file'] != self.generator_file:
                        continue

                    total_tests += 1
                    if self.run_test(
                        row['generator_file'],
                        row['taskvars'],
                        row['input_grid'],
                        row['output_grid']
                    ):
                        passed_tests += 1
                        print(f"\tTest passed for {row['generator_file']}")

        except FileNotFoundError:
            print(f"Error: Could not find test file: {self.csv_file}")
            return 0, 0

        return total_tests, passed_tests

def main():
    parser = argparse.ArgumentParser(description='Run tests for ARC task generators')
    parser.add_argument('--csv-file', type=str, default='tests/generator_tests.csv',
                      help='Path to CSV file containing tests')
    parser.add_argument('--generator', type=str,
                      help='Specific generator file to test')

    args = parser.parse_args()

    runner = ARCGeneratorTestRunner(args.csv_file, args.generator)
    total, passed = runner.run_all_tests()

    print(f"\nTest Summary:")
    print(f"Total tests: {total}")
    print(f"Passed tests: {passed}")
    print(f"Failed tests: {total - passed}")
    
    # Return non-zero exit code if any tests failed
    exit(0 if total == passed else 1)

if __name__ == "__main__":
    main()
