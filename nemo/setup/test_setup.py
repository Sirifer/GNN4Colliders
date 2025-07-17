import os
import importlib.util
import sys

def test_imports(directories):
    """
    Test importing all Python files in the specified directories.
    
    Parameters:
    - directories: List of directory paths to test.
    """
    print("Testing Conda environment...")

    for directory in directories:
        print(f"\nChecking directory: {directory}")
        
        # Check if the directory exists
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            continue
        
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            # Only consider Python files
            if filename.endswith(".py"):
                filepath = os.path.join(directory, filename)
                module_name = os.path.splitext(filename)[0]  # Remove .py extension
                
                try:
                    # Dynamically import the module
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print(f"Successfully imported: {filepath}")
                except Exception as e:
                    # Print the file and the error message if import fails
                    print(f"Failed to import: {filepath}")
                    print(f"Error: {e}")

if __name__ == "__main__":
    # Automatically append the current directory to sys.path
    current_directory = os.getcwd()
    sys.path.append(current_directory)
    print(f"Current directory added to sys.path: {current_directory}")

    # List of directories to check
    directories = ["scripts", "root_gnn_base", "models"]
    test_imports(directories)