import subprocess
from datetime import datetime

if __name__ == '__main__':
    # List of scripts to run
    scriptsToRun = [
        'IRProject3Create.py',
        'IRProject3DVI.py',
        'IRProject3QueryProcessArchive.py'
    ]

    # Dictionary to store the status of each script
    status = {}

    for script in scriptsToRun:
        try:
            subprocess.run(["python", script], check=True)
            status[script] = "Success at " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S")
        except subprocess.CalledProcessError as e:
            status[script] = f"Failed: {str(e)} at " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S")
        except Exception as e:
            status[script] = f"Error: {str(e)} at " + datetime.now().strftime("%d.%m.%Y_%I.%M.%S")

    # Print the results
    for script, result in status.items():
        print(f"{script}: {result}")