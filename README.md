Creating a virtual environment(name of virtual environment, 3rd argument, is "venv"):
1. (Into cmd): python -m venv venv
2. (Into cmd): venv\Scripts\activate.bat

Creating a virtual environment with system packages(packages already on machine)(name of virtual environment, 3rd argument, is "venv"):
1. (Into cmd): python -m venv venv --system-site-packages
2. (Into cmd): venv\Scripts\activate.bat

Exit venv:
1. deactivate

See packages(including system packages):
1. pip list 

See packages(local to virtual environment):
1. pip list --local

Get requirements.txt:
1. pip freeze

Install Requirements from requirements.txt:
1. pip install -r requirements.txt

Remove virtual environment(name of virtual environment, 1st argument, is "venv"):
1. (Into cmd): rmdir venv /s
