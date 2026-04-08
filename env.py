import random
from typing import Tuple, Dict

random.seed()


class ScamEnv:

    def __init__(self):

        self.current_step = 0
        self.max_steps = 6

        #  GROUPED INTO 3 TASKS
        self.data = [
            # -------- SCAM DETECTION --------
            {"text": "your package with delivery id-10976 arrive by 8pm", "actual": "safe", "task": "scam_detection"},
            {"text": "Woah! you won $1000 lottery click link", "actual": "scam", "task": "scam_detection"},
            {"text": "You are selected for internship join session", "actual": "scam", "task": "scam_detection"},
            {"text": "Hi your electricity bill of $10, ignore if paid", "actual": "safe", "task": "scam_detection"},

            # -------- PHISHING DETECTION --------
            {"text": "complete your bank details using link", "actual": "scam", "task": "phishing_detection"},
            {"text": "Update your KYC immediately or account suspended", "actual": "scam", "task": "phishing_detection"},
            {"text": "Overpaid, share bank details for refund", "actual": "scam", "task": "phishing_detection"},
            {"text": "your otp is 4578, do not share", "actual": "safe", "task": "phishing_detection"},

            # -------- SPAM DETECTION --------
            {"text": "Unknown number asking money", "actual": "scam", "task": "spam_detection"},
            {"text": "Pay 299 for offer letter refund later", "actual": "scam", "task": "spam_detection"},
            {"text": "Grandson arrested, need bail money urgent", "actual": "scam", "task": "spam_detection"},
            {"text": "Won Walmart gift card claim now", "actual": "scam", "task": "spam_detection"},
        ]

        self.current_data = None
        self.current_task = None

    def reset(self, task=None):
        self.current_step = 0

        #  PICK TASK (IMPORTANT)
        if task is None:
            self.current_task = random.choice([
                "scam_detection",
                "phishing_detection",
                "spam_detection"
            ])
        else:
            self.current_task = task

        # pick data from that task
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

        #  GRADER (important for validator)
        if prediction == correct_actual:
            reward = 1.0
        else:
            reward = -0.5

        if self.current_step >= self.max_steps:
            done = True
            return {}, reward, done, {}

        # next sample from same task
        for_task = [x for x in self.data if x["task"] == self.current_task]
        self.current_data = random.choice(for_task)

        observation = {
            "text": self.current_data["text"],
            "task": self.current_task
        }

        return observation, reward, done, {}

    def state(self):
        return {
            "step": self.current_step,
            "task": self.current_task,
            "text": self.current_data["text"] if self.current_data else None
        }