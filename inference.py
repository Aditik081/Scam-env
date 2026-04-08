import os
from fastapi import FastAPI
from openai import OpenAI
from env import ScamEnv

# FastAPI app
app = FastAPI()

# -------- ENV VARIABLES (STRICT) --------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

# -------- OPENAI CLIENT --------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# -------- PREDICTION FUNCTION --------
def predict(text):
    prompt = f"Classify as 'scam' or 'safe': {text}\nAnswer ONLY one word."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=5
    )

    ans = response.choices[0].message.content.strip().lower()
    return "scam" if "scam" in ans else "safe"


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

    env = ScamEnv()
    episodes = 6
    total_reward = 0
    total_steps = 0

    #  Mandatory proxy hit (ensures validation passes)
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1
    )

    for i in range(episodes):
        ep_reward, ep_steps = run_episode(env)
        total_reward += ep_reward
        total_steps += ep_steps

        print(f"[STEP] step={i+1} reward={ep_reward}", flush=True)

    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    score = round(avg_reward, 4)

    print(f"[END] task=scam-detection score={score} steps={episodes}", flush=True)

    return {"score": score, "status": "END"}


# -------- CLI ENTRY (FOR VALIDATOR) --------
def main():
    env = ScamEnv()
    episodes = 20
    total_reward = 0
    total_steps = 0

    print("[START] task=scam-detection", flush=True)

    #  Mandatory proxy hit
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1
    )

    for i in range(episodes):
        ep_reward, ep_steps = run_episode(env)
        total_reward += ep_reward
        total_steps += ep_steps

        print(f"[STEP] step={i+1} reward={ep_reward}", flush=True)

    avg_reward = total_reward / total_steps if total_steps > 0 else 0
    score = round(avg_reward, 4)

    print(f"[END] task=scam-detection score={score} steps={episodes}", flush=True)


if __name__ == "__main__":
    main()