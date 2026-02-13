import sys
import streamlit as st
import os
from datetime import datetime
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
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create a dictionary to store files by folder
    folder_files = {}
    for root, dirs, files in os.walk(base_path):
        if root != base_path:
            # Get task files with their modification times
            task_files = []
            for f in files:
                if f.endswith('.py') and 'task' in f.lower():
                    file_path = os.path.join(root, f)
                    mod_time = os.path.getmtime(file_path)
                    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    task_files.append({
                        'name': f,
                        'mod_time': mod_time,
                        'display_name': f"{f} (modified: {mod_time_str})"
                    })
            
            if task_files:  # Only add folders that contain task files
                # Sort files by modification time (newest first)
                task_files.sort(key=lambda x: x['mod_time'], reverse=True)
                rel_path = os.path.relpath(root, base_path)
                folder_files[rel_path] = task_files

    if not folder_files:
        st.error("No task generator files found!")
        return

    # Initialize session state for folder and file selection
    if 'current_folder' not in st.session_state:
        st.session_state.current_folder = list(folder_files.keys())[0]
    if 'current_file_idx' not in st.session_state:
        st.session_state.current_file_idx = 0

    if 'generator_changed' not in st.session_state:
        st.session_state.generator_changed = False
    # generator_changed = False

    selected_folder = st.selectbox(
        "select folder and task generator (generators sorted descending by last modification date on disk)",
        options=list(folder_files.keys()),
        index=list(folder_files.keys()).index(st.session_state.current_folder),
        key='folder_selectbox'
    )

    # Update current folder and reset file index if folder changes
    if selected_folder != st.session_state.current_folder:
        st.session_state.current_folder = selected_folder
        st.session_state.current_file_idx = 0
        st.session_state.generator_changed = True
        # generator_changed = True
        st.rerun()

    current_files = folder_files[selected_folder]

    if 'button_clicked' in st.session_state:
        if st.session_state.button_clicked == 'prev':
            st.session_state.current_file_idx = (st.session_state.current_file_idx - 1) % len(current_files)
            st.session_state.generator_changed = True
            # generator_changed = True
        elif st.session_state.button_clicked == 'random':
            st.session_state.current_file_idx = np.random.randint(0, len(current_files))
            st.session_state.generator_changed = True
            # generator_changed = True
        elif st.session_state.button_clicked == 'next':
            st.session_state.current_file_idx = (st.session_state.current_file_idx + 1) % len(current_files)
            st.session_state.generator_changed = True
            # generator_changed = True
        del st.session_state.button_clicked

    selected_file = st.selectbox(
        "Select Task Generator",
        options=[f['display_name'] for f in current_files],
        index=st.session_state.current_file_idx,
        key='file_selectbox',
        label_visibility="collapsed"
    )

    generate_col, buffer_col, prev_col, random_col, next_col = st.columns([2, 6, 1, 1, 1])

    with generate_col:
        generate_button = st.button("Generate New Task")

    with prev_col:
        if st.button("Previous"):
            st.session_state.button_clicked = 'prev'
            st.rerun()

    with random_col:
        if st.button("Random"):
            st.session_state.button_clicked = 'random'
            st.rerun()

    with next_col:
        if st.button("Next"):
            st.session_state.button_clicked = 'next'
            st.rerun()

    # Update current file index when selectbox changes
    selected_display_name = selected_file
    current_file_idx = next((i for i, f in enumerate(current_files) if f['display_name'] == selected_display_name), 0)

    if st.session_state.current_file_idx != current_file_idx:
        st.session_state.current_file_idx = current_file_idx
        st.session_state.generator_changed = True
        # generator_changed = True
        st.rerun()

    # Convert to full path
    selected_file_path = os.path.join(base_path, selected_folder, current_files[st.session_state.current_file_idx]['name'])

    # Load and instantiate the generator
    generator_class = load_task_generator(selected_file_path)
    if generator_class is None:
        return
    generator = generator_class()

    # Generate new task when generator changes or button is clicked
    if st.session_state.generator_changed or generate_button:
        task = generator.create_task()
        
        # Store in session state
        st.session_state.task = task
        st.session_state.train_test_data = task.data
        st.session_state.input_reasoning = task.input_reasoning_chain
        st.session_state.transform_reasoning = task.transformation_reasoning_chain
        st.session_state.transform_code = task.code
        st.session_state.task_vars = task.task_variables

        st.session_state.generator_changed = False

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
        st.code(st.session_state.transform_code, language='python')

if __name__ == "__main__":
    main()
