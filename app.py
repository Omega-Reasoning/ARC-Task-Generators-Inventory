import sys
import streamlit as st
import os
import importlib.util
import inspect
import numpy as np

def load_task_generator(file_path):
    """Dynamically load a task generator class from a Python file."""
    try:
        # Get the module name from the file path
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Add the directory containing the module to Python's path
        module_dir = os.path.dirname(os.path.abspath(file_path))
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Load the spec
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            st.error(f"Could not load specification for module: {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Find the class that inherits from ARCTaskGenerator
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                any(base.__name__ == 'ARCTaskGenerator' for base in obj.__bases__)):
                return obj
                
        st.error(f"No ARCTaskGenerator class found in {file_path}")
        return None
        
    except Exception as e:
        st.error(f"Error loading module {file_path}: {str(e)}")
        return None

def display_matrix_fixed_size(matrix, title):
    """Display a 2D matrix with colors."""
    # Define the RGB color mapping
    colors = [
        (0, 0, 0),          # black for empty
        (30, 147, 255),     # blue
        (249, 60, 49),      # red
        (79, 204, 48),      # green
        (255, 220, 0),      # yellow
        (153, 153, 153),    # grey
        (229, 58, 163),     # pink
        (255, 137, 27),     # orange
        (135, 216, 241),    # cyan
        (146, 18, 49)       # maroon
    ]

    st.subheader(title)
    html = '<style>table {border-collapse: collapse} td {width: 30px; height: 30px; text-align: center; border: 1px solid black;}</style>'
    html += '<table>'
    for row in matrix:
        html += '<tr>'
        for cell in row:
            # Get the RGB color for the cell value
            if 0 <= cell < len(colors):
                r, g, b = colors[cell]
                color = f'background-color: rgb({r}, {g}, {b})'
                # Use white text for dark backgrounds
                text_color = 'white' if (r + g + b) < 382 else 'black'  # 382 is roughly half of 255*3
                style = f'{color}; color: {text_color}'
            else:
                style = ''
            html += f'<td style="{style}">{cell}</td>'
        html += '</tr>'
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)

def display_matrix(matrix, title):
    """Display a 2D matrix with colors."""
    # Calculate cell size based on matrix dimensions
    num_cols = len(matrix[0])
    # cell_size = min(30, max(10, int(300 / num_cols)))  # Adjust cell size based on matrix width
    cell_size = 30
    
    colors = [
        (0, 0, 0),          # black for empty
        (30, 147, 255),     # blue
        (249, 60, 49),      # red
        (79, 204, 48),      # green
        (255, 220, 0),      # yellow
        (153, 153, 153),    # grey
        (229, 58, 163),     # pink
        (255, 137, 27),     # orange
        (135, 216, 241),    # cyan
        (146, 18, 49)       # maroon
    ]

    st.subheader(title)
    html = f'''
        <style>
            table {{ border-collapse: collapse; margin: auto; }}
            td {{ 
                width: {cell_size}px; 
                height: {cell_size}px; 
                text-align: center; 
                border: 1px solid black;
                font-size: {max(8, cell_size//2)}px;
            }}
        </style>
    '''
    html += '<table>'
    for row in matrix:
        html += '<tr>'
        for cell in row:
            if 0 <= cell < len(colors):
                r, g, b = colors[cell]
                color = f'background-color: rgb({r}, {g}, {b})'
                text_color = 'white' if (r + g + b) < 382 else 'black'
                style = f'{color}; color: {text_color}'
            else:
                style = ''
            html += f'<td style="{style}">{cell}</td>'
        html += '</tr>'
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide") 
    st.title("ARC Task Generator Viewer")

    # Get list of Python files in the project
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get current directory
    python_files = []
    for root, dirs, files in os.walk(base_path):
        # Skip the base directory itself
        if root != base_path:
            for file in files:
                if file.endswith('.py') and 'task' in file.lower():
                    python_files.append(os.path.join(root, file))

    if not python_files:
        st.error("No task generator files found!")
        return

    # Create display names (relative paths) for the files
    display_files = [os.path.relpath(f, base_path) for f in python_files]

    # File selection
    current_file_idx = st.session_state.get('current_file_idx', 0)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    generator_changed = False
    
    with col1:
        if st.button("Previous"):
            current_file_idx = (current_file_idx - 1) % len(python_files)
            st.session_state.current_file_idx = current_file_idx
            generator_changed = True
            
    with col2:
        previous_file = st.session_state.get('previous_file')
        selected_display_file = st.selectbox(
            "Select Task Generator",
            display_files,
            index=current_file_idx
        )
        # Convert display file back to full path
        selected_file = python_files[display_files.index(selected_display_file)]
        if previous_file != selected_file:
            st.session_state.previous_file = selected_file
            generator_changed = True
        
    with col3:
        if st.button("Next"):
            current_file_idx = (current_file_idx + 1) % len(python_files)
            st.session_state.current_file_idx = current_file_idx
            generator_changed = True

    # Load and instantiate the generator
    generator_class = load_task_generator(selected_file)
    if generator_class is None:
        return
    generator = generator_class()

    generate_button = st.button("Generate New Task")

    # Generate new task when generator changes or button is clicked
    if generator_changed or generate_button:
        task = generator.create_task()
        
        # Store in session state
        st.session_state.task = task
        st.session_state.train_test_data = task.train_test_data
        st.session_state.input_reasoning = task.input_reasoning_chain
        st.session_state.transform_reasoning = task.transformation_reasoning_chain
        st.session_state.transform_code = task.transform_code
        st.session_state.task_vars = task.task_variables

    # Display task information if available
    if 'task' in st.session_state:
        st.header("Task Information")
        # Convert the dict to a formatted string, handling np.int64 values
        task_vars_formatted = ", ".join(f"{k}: {int(v) if hasattr(v, 'item') else v}" 
                                    for k, v in st.session_state.task_vars.items())
        st.text("task variables: " + task_vars_formatted)

        # Display reasoning chains
        st.subheader("Input Reasoning Chain")
        for item in st.session_state.input_reasoning:
            st.write(f"• {item}")
            
        st.subheader("Transformation Reasoning Chain")
        for item in st.session_state.transform_reasoning:
            st.write(f"• {item}")

        # Loop over both train and test examples
        for dataset_type in ["train", "test"]:
            st.header(f"{dataset_type.title()} Examples")
            for idx, example in enumerate(st.session_state.train_test_data[dataset_type]):
                st.subheader(f"Example {idx + 1}")
                
                # Calculate column ratio based on matrix sizes
                input_cols = len(example["input"][0])
                output_cols = len(example["output"][0])
                total_cols = input_cols + output_cols
                col1_ratio = input_cols / total_cols
                col2_ratio = output_cols / total_cols
                
                col1, col2 = st.columns([col1_ratio, col2_ratio])
                with col1:
                    display_matrix(example["input"], "Input")
                with col2:
                    display_matrix(example["output"], "Output")


        # Display source code
        st.header("Source Code")
        #with open(selected_file, 'r') as f:
        #    st.code(f.read(), language='python')
        st.code(st.session_state.transform_code, language='python')

if __name__ == "__main__":
    main()
