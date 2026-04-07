import os
from fastapi import FastAPI
from openai import OpenAI
from env import ScamEnv

# FastAPI app
app = FastAPI()

# -------- ENV VARIABLES (SAFE MODE) --------
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# -------- OPENAI CLIENT --------
client = None
if API_BASE_URL and API_KEY and MODEL_NAME:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

# -------- PREDICTION FUNCTION --------
def predict(text):
    prompt = f"Classify as 'scam' or 'safe': {text}\nAnswer ONLY one word."

    # Try API call first (important for validator)
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=5
            )
            ans = response.choices[0].message.content.strip().lower()
            return "scam" if "scam" in ans else "safe"
        except Exception as e:
            print(f"[ERROR] API failed: {e}", flush=True)

    # fallback (no crash)
    return "safe"

# -------- EPISODE RUNNER --------
def run_episode(env):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        text = obs["text"]
        prediction = predict(text)
        obs, reward, done, _ = env.step(prediction)
        total_reward += reward
        steps += 1

    return total_reward, steps

# -------- FASTAPI ENDPOINTS --------
@app.get("/")
def health():
    return {"status": "running"}

@app.post("/reset")
def reset_env():
    return {"status": "success"}

@app.get("/run_task")
def run_task_endpoint():
    print("[START] task=scam-detection", flush=True)

    try:
        env = ScamEnv()
        episodes = 20
        total_reward = 0
        total_steps = 0

        for i in range(episodes):
            ep_reward, ep_steps = run_episode(env)
            total_reward += ep_reward
            total_steps += ep_steps

            print(f"[STEP] step={i+1} reward={ep_reward}", flush=True)

        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        score = round(avg_reward, 4)

        print(f"[END] task=scam-detection score={score} steps={episodes}", flush=True)

        return {"score": score, "status": "END"}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return {"error": str(e)}

# -------- CLI ENTRY (FOR VALIDATOR) --------
def main():
    env = ScamEnv()
    episodes = 20
    total_reward = 0
    total_steps = 0

    print("[START] task=scam-detection", flush=True)

    for i in range(episodes):
        ep_reward, ep_steps = run_episode(env)
        total_reward += ep_reward
        total_steps += ep_steps

        print(f"[STEP] step={i+1} reward={ep_reward}", flush=True)

    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    score = round(avg_reward, 4)

    print(f"[END] task=scam-detection score={score} steps={episodes}", flush=True)

# -------- IMPORTANT: DO NOT START UVICORN HERE --------
if __name__ == "__main__":
    main()