from arc_task_generator import ARCTaskGenerator
from arc_training.task007bbfb7 import ARCTask007bbfb7

if __name__ == "__main__":

    generator = ARCTask007bbfb7()
    task = generator.create_task()
    print(task)
    ARCTaskGenerator.visualize_train_test_data(task.train_test_data)

