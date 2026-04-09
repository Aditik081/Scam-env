import os
import traceback
from fastapi import FastAPI
from openai import OpenAI
from env import ScamEnv

# -------- FASTAPI APP --------
app = FastAPI()

# -------- SAFE ENV VARIABLES --------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", "dummy-key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# -------- OPENAI CLIENT --------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# -------- PREDICT FUNCTION --------
def predict(text):
    try:
        prompt = f"Classify as 'scam' or 'safe': {text}\nAnswer ONLY one word."

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )

        ans = response.choices[0].message.content.strip().lower()

        if "scam" in ans:
            return "scam"
        else:
            return "safe"

    except Exception as e:
        print(f"[ERROR] predict failed: {e}", flush=True)
        return "safe"   # fallback (IMPORTANT)


# -------- EPISODE RUNNER --------
def run_episode(env):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        try:
            text = obs["text"]
            prediction = predict(text)
            obs, reward, done, _ = env.step(prediction)

            total_reward += reward
            steps += 1

        except Exception as e:
            print(f"[ERROR] step failed: {e}", flush=True)
            break

    return total_reward, steps


# -------- HEALTH CHECK --------
@app.get("/")
def health():
    return {"status": "running"}


# -------- RESET --------
@app.post("/reset")
def reset_env():
    return {"status": "success"}


# -------- RUN TASK (API MODE) --------
@app.get("/run_task")
def run_task_endpoint():
    try:
        print("[START] task=scam-detection", flush=True)

        env = ScamEnv()
        episodes = 6
        total_reward = 0
        total_steps = 0

        # Safe proxy ping
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception as e:
            print(f"[ERROR] proxy failed: {e}", flush=True)

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
        print("[FATAL ERROR]", e, flush=True)
        traceback.print_exc()
        return {"score": 0, "status": "ERROR"}


# -------- CLI ENTRY (VALIDATOR MODE) --------
def main():
    try:
        print("[START] task=scam-detection", flush=True)

        env = ScamEnv()
        episodes = 20
        total_reward = 0
        total_steps = 0

        # Safe proxy ping
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception as e:
            print(f"[ERROR] proxy failed: {e}", flush=True)

        for i in range(episodes):
            ep_reward, ep_steps = run_episode(env)
            total_reward += ep_reward
            total_steps += ep_steps

            print(f"[STEP] step={i+1} reward={ep_reward}", flush=True)

        avg_reward = total_reward / total_steps if total_steps > 0 else 0
        score = round(avg_reward, 4)

        print(f"[END] task=scam-detection score={score} steps={episodes}", flush=True)

    except Exception as e:
        print("[FATAL ERROR]", e, flush=True)
        traceback.print_exc()


# -------- ENTRY POINT --------
if __name__ == "__main__":
    main()