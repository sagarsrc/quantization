# File: notebooks/notebook_setup.py
import os
import sys
import importlib
from pathlib import Path


def setup_project():
    """
    Set up the Python path to include the project root directory and enable module auto-reloading.
    This allows importing from 'src' and auto-reloads changes to source files.
    """
    # Find the project root (parent dir of notebooks)
    current_file = Path(os.path.abspath(__file__))
    project_root = current_file.parent.parent

    # Add project root to Python path if it's not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")

    # Configure auto-reloading of modules
    try:
        # Only run this in IPython environment
        ipython = get_ipython()

        # Load the autoreload extension
        ipython.run_line_magic("load_ext", "autoreload")

        # Configure autoreload to reload all modules every time before executing the code
        ipython.run_line_magic("autoreload", "2")

        print("Autoreload enabled: Source file changes will be automatically reloaded")
    except (NameError, ImportError):
        print("Not running in IPython environment, autoreload not enabled")

    print("Project setup complete. You can now import from 'src'")

    return project_root


# Run setup when imported
project_root = setup_project()
