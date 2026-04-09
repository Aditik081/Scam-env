import uvicorn
from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.get("/run")
def run_inference():
    import subprocess
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True
    )
    return {"logs": result.stdout, "errors": result.stderr}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()