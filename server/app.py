import uvicorn
from fastapi import FastAPI
import sys
import os

# Root directory ko path mein add karein taaki inference.py mil sake
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import app as inference_app

app = inference_app

def main():
    """Main entry point for the server as required by OpenEnv validator."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()