import os
from openai import OpenAI
from env.env import ScamEnv
from env.env import normal_agent   # 🔥 fallback

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def get_prediction(text):
    prompt = f"""
You are a scam detection system.

Classify the message strictly as:
- scam
- safe

Rules:
- OTP without link/request = safe
- Asking money, clicking links, urgency = scam

Message: {text}

Answer ONLY one word: scam or safe
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )

        ans = response.choices[0].message.content.strip().lower()

        # 🔥 clean output handling
        if "scam" in ans:
            return "scam"
        elif "safe" in ans:
            return "safe"
        else:
            return normal_agent(text)   # fallback

    except:
        return normal_agent(text)       # fallback


def run_episode(env):
    obs = env.reset()
    done = False

    total_reward = 0
    steps = 0

    while not done:
        text = obs["text"]

        # 🔥 short text → better handled by rules
        if len(text) < 25:
            prediction = normal_agent(text)
        else:
            prediction = get_prediction(text)

        obs, reward, done, _ = env.step(prediction)

        total_reward += reward
        steps += 1

    return total_reward, steps


def main():
    env = ScamEnv()

    episodes = 20   # 🔥 more stable score
    total_reward = 0
    total_steps = 0

    for _ in range(episodes):
        ep_reward, ep_steps = run_episode(env)
        total_reward += ep_reward
        total_steps += ep_steps

    avg_reward = total_reward / total_steps if total_steps > 0 else 0

    # ✅ ONLY output (important for submission)
    print(round(avg_reward, 4))


if __name__ == "__main__":
    main()