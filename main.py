import subprocess
import sys
subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
with open('ACP/mainACP.py', 'r', encoding="utf-8") as f:
    exec(f.read())
with open('AEF/mainAEF.py', 'r', encoding="utf-8") as f:
    exec(f.read())