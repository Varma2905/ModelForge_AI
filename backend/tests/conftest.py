import os
import sys

# Allow `from app...` imports when pytest is invoked from any working directory.
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
