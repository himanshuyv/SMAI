# Took this code from chatGPT to run multiple python scripts of knn, regression and regularization

import subprocess

python_scripts = ['knn.py', 'regression.py', 'regularization.py']

for script in python_scripts:
    print(f'Running {script}')
    try:
        results =  subprocess.run(['py', f'./{script}'], check=True, capture_output=True, text=True)
        print(f'Outputs for {script} are\n {results.stdout}')
    except subprocess.CalledProcessError as e:
        print(f'Error running {script}')
        print(f'Error: {e}')
    except FileNotFoundError:
        print(f'Error running {script}')
        print(f'Error: File not found')
    print(f'Completed {script}')
    print()

print('All scripts have been run successfully')