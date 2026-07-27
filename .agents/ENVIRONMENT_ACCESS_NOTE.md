# Environment Access Note

## Project `.venv` Python interpreter

- Project virtual environment: `.venv\`
- The sandbox can read `.venv\Scripts\python.exe` and its files.
- The sandbox cannot access the host Python installation referenced by this virtual environment:
  `C:\Users\13603\AppData\Local\Programs\Python\Python310\python.exe`
- Running `.venv\Scripts\python.exe` in the default sandbox fails with:
  `No Python at "C:\Users\13603\AppData\Local\Programs\Python\Python310\python.exe"`
- Accessing the host path may also return `Access is denied`. This is a sandbox isolation restriction; it does not mean that the project or virtual-environment directory is missing.
- If a future task needs to run the project, tests, or dependency installation, request approval for host-side Python execution or recreate the virtual environment with a Python interpreter available inside the sandbox.
