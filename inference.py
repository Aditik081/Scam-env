import os
import sys
import time
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from env import normal_agent, ScamEnv

# FastAPI app initialize karein (Scaler ko response chahiye hota hai)
app = FastAPI()

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN") 
MODEL_NAME = os.getenv("MODEL_NAME")

client = None
if API_BASE_URL and HF_TOKEN and MODEL_NAME:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

def predict(text):
    if client is None:
        return normal_agent(text)
        
    prompt = f"Classify as 'scam' or 'safe': {text}\nAnswer ONLY one word."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )
        ans = response.choices[0].message.content.strip().lower()
        return "scam" if "scam" in ans else "safe"
    except:
        return normal_agent(text)

def run_episode(env):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        text = obs["text"]
        prediction = predict(text) if len(text) >= 25 else normal_agent(text)
        obs, reward, done, _ = env.step(prediction)
        total_reward += reward
        steps += 1
    return total_reward, steps

# --- SCALER SPECIFIC ENDPOINTS ---

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/reset")
def reset_env():
    # Scaler jab 'reset()' call karega, hum usey 200 OK bhejenge
    return {"status": "success", "message": "Environment Reset"}

@app.get("/run_task")
def main():
    print("START")
    try:
        env = ScamEnv()
        episodes = 20
        total_reward = 0
        total_steps = 0

        for _ in range(episodes):
            ep_reward, ep_steps = run_episode(env)
            total_reward += ep_reward
            total_steps += ep_steps

        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        print("END")
        print(round(avg_reward, 4))
        return {"score": round(avg_reward, 4), "status": "END"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face default port 7860 use karta hai
    uvicorn.run(app, host="0.0.0.0", port=7860)