import subprocess
import os
import sys  # <-- ADDED: This module lets us find the active Python environment

# Define the sequence of files to execute (and their folder)
folder = "Coding"
scripts = [
    'data_cleaning.py', 
    'find_optimal_hyperparameters.py', 
    'estimation_aipw.py',
    'propensity_score_check.py'
    'propensity_score_check.py', 
    'sensitivity_placebo.py', 
    'sensitivity_overlap.py', 
    'tables_and_figures_econml.py', 
    'tables_and_figures_sensitivity.py'
]

for script in scripts:
    # Access the file
    script_path = os.path.join(folder, script)

    print(f"--- Starting {script} ---")
    
    # check=True forces it to stop if any script fails with an error
    # FIX: Replaced 'python' with sys.executable
    subprocess.run([sys.executable, script_path], check=True) 
    
    print(f"--- Finished {script} ---\n")