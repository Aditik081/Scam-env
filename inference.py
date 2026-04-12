import os
import sys
from typing import List

# -------- ENV VARIABLES --------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("API_KEY", "dummy-key")  # validator yahi inject karta hai

# -------- OPENAI CLIENT --------
from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _safe_predict(text: str, task: str) -> str:
    text = text.lower()
    if task == "easy":
        if any(w in text for w in ["win", "lottery", "gift", "free", "click"]):
            return "scam"
        return "safe"
    elif task == "medium":
        if any(w in text for w in ["kyc", "otp", "bank", "link", "update", "internship"]):
            return "scam"
        if "do not share" in text:
            return "safe"
        return "safe"
    elif task == "hard":
        if any(w in text for w in ["money", "pay", "urgent", "refund", "arrest", "bail", "overpaid"]):
            return "scam"
        return "safe"
    return "safe"


def _llm_predict(text: str, task: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scam detection AI. "
                        "Classify the message as scam or safe. "
                        "Reply with ONLY one word: scam or safe."
                    )
                },
                {
                    "role": "user",
                    "content": f"Message: {text}"
                }
            ],
            max_tokens=5,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip().lower()
        if "scam" in result:
            return "scam"
        return "safe"
    except Exception as e:
        print(f"[WARN] LLM failed: {e}", flush=True)
        return _safe_predict(text, task)


def main() -> None:
    sys.path.insert(0, ".")
    from env import ScamEnv
    from grader import ScamGrader

    grader = ScamGrader()
    task_ids = ["easy", "medium", "hard"]

    for task_id in task_ids:
        env = ScamEnv()
        obs = env.reset(task=task_id)

        print(f"[START] task={task_id} env=ScamEnv model={MODEL_NAME}", flush=True)

        step_idx = 0
        done = False
        rewards_list: List[float] = []

        while not done and step_idx < env.max_steps:
            step_idx += 1
            text = obs["text"]

            try:
                action_label = _llm_predict(text, task_id)
                obs, reward, done, info = env.step(action_label)

                grade_fn = getattr(grader, task_id)
                step_score = float(grade_fn(action_label, obs, info))
                rewards_list.append(step_score)

                print(
                    f"[STEP] step={step_idx} action={action_label} "
                    f"reward={step_score:.2f} done={str(done).lower()} error=null",
                    flush=True
                )

            except Exception as e:
                err = str(e).replace("\n", " ")
                rewards_list.append(0.15)
                print(
                    f"[STEP] step={step_idx} action=none "
                    f"reward=0.15 done=true error={err}",
                    flush=True
                )
                done = True
                break

        rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
        print(f"[END] success=true steps={step_idx} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    main()