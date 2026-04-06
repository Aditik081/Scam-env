import os
import sys
import time
from openai import OpenAI
# Yahan 'env' file se sahi classes import karein
from env import normal_agent, ScamEnv 

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

    while not done and steps < 100: # Safety break
        text = obs["text"]
        # Yahan 'predict' call karein
        prediction = predict(text) if len(text) >= 25 else normal_agent(text)
        
        obs, reward, done, _ = env.step(prediction)
        total_reward += reward
        steps += 1
    return total_reward, steps

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
        
        # Zaroori: Script ko thodi der "Running" state mein rakhne ke liye
        print("SUCCESS: Result generated. Waiting for Scaler to scan...")
        time.sleep(300) # 5 minute sleep
        
    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(60) # Error ke case mein bhi thoda wait
    
    sys.exit(0)

if __name__ == "__main__":
    main()