# ARC Task Generators

A project for writing generators for Abstraction and Reasoning Corpus (ARC) tasks.

## Prerequisites

- Python 3.12 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omega-Reasoning/ARC-Task-Generators.git
cd ARC-Task-Generators
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Use generators

You have three options to use the generators described below:
* test them in a Streamlit app (app.py)
* run a single generator via commandline (test.py)
* create a full dataset (dataset_generator.py) 

### Streamlit

Run the Streamlit app to interactively test and visualize task generators:
```bash
streamlit run app.py
```

This allows you to select a generator, create new tasks using and view the output.

### Command line

#### Generate an ARC task 
```bash
python test.py arc_training/task007bbfb7.py 
```

#### Generate and visualize a task
```bash
python test.py arc_training/task007bbfb7.py -v
```
