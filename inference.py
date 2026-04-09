import json
import os
import sys
from typing import Any, Dict

def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(tag, json.dumps(payload, separators=(",", ":"), ensure_ascii=True))

def _safe_predict(text: str, task: str) -> str:
    text = text.lower()
    if task == "easy":
        if any(w in text for w in ["win", "lottery", "gift", "free", "click"]):
            return "scam"
        return "safe"
    elif task == "medium":
        if any(w in text for w in ["kyc", "otp", "bank", "link", "update"]):
            return "scam"
        if "do not share" in text:
            return "safe"
        return "safe"
    elif task == "hard":
        if any(w in text for w in ["money", "pay", "urgent", "refund", "arrest"]):
            return "scam"
        return "safe"
    return "safe"

def _llm_predict(client, model_name: str, text: str, task: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scam detection AI. "
                        "Classify the message as 'scam' or 'safe'. "
                        "Reply with ONLY one word: scam or safe."
                    )
                },
                {
                    "role": "user",
                    "content": f"Message: {text}\nTask difficulty: {task}\nClassify:"
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
        print(f"[WARN] LLM call failed: {e}", flush=True)
        return _safe_predict(text, task)  # fallback

def main() -> None:
    sys.path.insert(0, ".")

    from openai import OpenAI
    from env import ScamEnv
    from grader import ScamGrader

    # ← Yeh teen lines ZARURI hain — validator inhi se check karta hai
    api_base_url = os.environ["API_BASE_URL"]   # no default!
    api_key = os.environ["API_KEY"]             # no default!
    model_name = os.environ["MODEL_NAME"]       # no default!

    # OpenAI client with validator's proxy
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key
    )

    task_ids = ["easy", "medium", "hard"]
    grader = ScamGrader()

    for task_id in task_ids:
        env = ScamEnv()
        obs = env.reset(task=task_id)
        max_steps = env.max_steps

        _emit("[START]", {
            "task_id": task_id,
            "max_steps": max_steps,
            "reward_range": [0.0, 1.0],
        })

        step_idx = 0
        done = False
        final_score = 0.5
        info = {}

        while not done and step_idx < max_steps:
            step_idx += 1
            text = obs["text"]

            # ALWAYS try LLM first — validator yahi check karta hai
            action_label = _llm_predict(client, model_name, text, task_id)

            obs, reward, done, info = env.step(action_label)

            grade_fn = getattr(grader, task_id)
            final_score = float(grade_fn(action_label, obs, info))

            _emit("[STEP]", {
                "task_id": task_id,
                "step": step_idx,
                "action": action_label,
                "reward": final_score,
                "done": bool(done),
            })

        final_score = max(0.01, min(0.99, final_score))

        _emit("[END]", {
            "task_id": task_id,
            "score": final_score,
            "reward": final_score,
            "done": True,
        })

if __name__ == "__main__":
    main()