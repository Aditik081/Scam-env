import random
from typing import Tuple, Dict

random.seed()


class ScamEnv:

    def __init__(self):

        self.current_step = 0
        self.max_steps = 6

        # EASY / MEDIUM / HARD TASKS
        self.data = [
            # -------- EASY --------
            {"text": "your package will arrive by 8pm", "actual": "safe", "task": "easy"},
            {"text": "You won $1000 lottery click link", "actual": "scam", "task": "easy"},
            {"text": "Hi your electricity bill is paid", "actual": "safe", "task": "easy"},
            {"text": "Claim your free gift now", "actual": "scam", "task": "easy"},

            # -------- MEDIUM --------
            {"text": "You are selected for internship join session", "actual": "scam", "task": "medium"},
            {"text": "Update your KYC immediately", "actual": "scam", "task": "medium"},
            {"text": "your otp is 4578, do not share", "actual": "safe", "task": "medium"},
            {"text": "complete your bank details using link", "actual": "scam", "task": "medium"},

            # -------- HARD --------
            {"text": "Unknown number asking money", "actual": "scam", "task": "hard"},
            {"text": "Pay 299 for offer letter refund later", "actual": "scam", "task": "hard"},
            {"text": "Overpaid, share bank details for refund", "actual": "scam", "task": "hard"},
            {"text": "Grandson arrested, need bail money urgent", "actual": "scam", "task": "hard"},
        ]

        self.current_data = None
        self.current_task = None

    def reset(self, task=None):
        self.current_step = 0

        if task is None:
            self.current_task = random.choice(["easy", "medium", "hard"])
        else:
            self.current_task = task

        for_task = [x for x in self.data if x["task"] == self.current_task]
        self.current_data = random.choice(for_task)

        return {
            "text": self.current_data["text"],
            "task": self.current_task
        }

    def step(self, prediction: str) -> Tuple[Dict, float, bool, Dict]:

        self.current_step += 1
        done = False

        correct_actual = self.current_data["actual"]

        # reward (0–1)
        reward = 1.0 if prediction == correct_actual else 0.0

        if self.current_step >= self.max_steps:
            done = True

        # next sample
        for_task = [x for x in self.data if x["task"] == self.current_task]
        self.current_data = random.choice(for_task)

        observation = {
            "text": self.current_data["text"],
            "task": self.current_task
        }

        info = {
            "actual": correct_actual   # 🔥 REQUIRED FOR GRADER
        }

        return observation, reward, done, info

    def state(self):
        return {
            "step": self.current_step,
            "task": self.current_task,
            "text": self.current_data["text"] if self.current_data else None
        }