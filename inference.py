import json
import os
import sys
from typing import Any, Dict, List


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


def _choose_action_via_llm(*, client, model_name, task_id, step, obs):
    system = (
        "You are a scam detection agent.\n"
        "Return ONLY JSON with one action.\n"
        '{"action_type":"classify","label":"scam|safe"}\n'
    )
    user = {"task_id": task_id, "step": step, "observation": obs}
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def main() -> None:
    sys.path.insert(0, ".")

    from openai import OpenAI
    from env import ScamEnv
    from grader import ScamGrader

    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    client = None
    if api_base_url and model_name and hf_token:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)

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

            # Try LLM, fallback to rule-based
            action_label = None
            if client is not None:
                try:
                    payload = _choose_action_via_llm(
                        client=client,
                        model_name=model_name,
                        task_id=task_id,
                        step=step_idx,
                        obs=obs,
                    )
                    action_label = payload.get("label")
                except Exception:
                    action_label = None

            if action_label not in ("scam", "safe"):
                action_label = _safe_predict(text, task_id)

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

        # Clamp strictly between 0 and 1
        final_score = max(0.01, min(0.99, final_score))

        _emit("[END]", {
            "task_id": task_id,
            "score": final_score,
            "reward": final_score,
            "done": True,
        })


if __name__ == "__main__":
    main()