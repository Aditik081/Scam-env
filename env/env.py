import random
from typing import Tuple, Dict


class ScamEnv:

    def __init__(self):
       

        #step tracking
        self.current_step = 0
        self.max_steps = 6

        self.data=[
            {"text":"your package with delivery id-10976 arrive by 8pm", "actual":"safe","level":"easy"},
            {"text":"Woah! you won you won $1000 lottery for claiming it click the link below","actual":"scam","level":"easy"},
            {"text":"You are selected for internship first join the session time is given below","actual":"scam","level":"medium"},
            {"text":"Hi your electricity bill of $10, Avoid if you lready paid", "actual":"safe","level":"easy"},
            {"text":"complete your bank details using link","actual":"scam","level":"medium"},
            {"text":"Update your KYC immediately or your account will suspended","actual":"scam","level":"hard"},
            {"text":"your otp is 4578, do not share with anyone","actual":"safe","level":"easy"},
            {"text":"Message from Unknown number of family and relative asking money","actual":"scam","level":"hard"},
            {"text":"Dear Candidate you have selected for offer letter pay 299 and we refund it after you recieve it.","actual":"scam","level":"hard"},
            {"text":"Our records show you overpaid for (a product or service). Kindly supply your bank routing and account number to receive your refund.","actual":"scam","level":"hard"},
            {"text":"Urgent Your grandson was arrested last night in maxico. Need Bail money immediately to State Police ","actual":"scam","level":"hard"},
            {"text":"Congratulations! You've won the Walmart Gift card of $1000 go to link tp claim now.","actual":"scam","level":"medium"},
            


        ]

        self.current_data = None

    def reset(self,level=None):
        self.current_step=0  #new game come to 0
        if level is None:
            level = random.choice(["easy","medium","hard"])

        for_level = [x for x in self.data if x["level"]==level]
        self.current_data = random.choice(for_level)


        
        
        return {
            "text": self.current_data["text"],
            "has_link": "link" in self.current_data["text"].lower(),
            "has_urgent_words": "urgent" in self.current_data["text"].lower()
}

    def step(self, prediction:str)->Tuple[Dict,float,bool,Dict]:
        
        self.current_step += 1
        done = False

        correct_actual = self.current_data["actual"]
        
        if prediction == correct_actual:
            if self.current_data["level"] == "easy":
                reward = 0.6
            elif self.current_data["level"]=="medium":
                reward = 0.8
            else:
                reward = 1.0
        else: 
            reward = 0.0

        if self.current_step >= self.max_steps:
           done = True
        else:
           for_level = [x for x in self.data if x["level"] == self.current_data["level"]]
           self.current_data = random.choice(for_level)
    

        
         
        
        observation ={
            "text": self.current_data["text"],
            "has_link": "link" in self.current_data["text"].lower(),
            "has_urgent_words": "urgent" in self.current_data["text"].lower()
        }

        info = {}

        return observation, reward, done, info

def normal_agent(text):
    text = text.lower()

    if "otp" in text:
        if "link" in text or "click" in text or "pay" in text:
            return "scam"
        return "safe"

    scam_keywords = [
        "lottery", "kyc", "win",
        "urgent", "money", "click", "link",
        "suspended", "bank", "verify",
        "pay", "offer", "selected", "prize",
        "gift", "claim", "reward"
    ]

    for word in scam_keywords:
        if word in text :
            return "scam"

    if "pay" in text and ("offer" in text or "job" in text):
        return "scam"
            
    
    return "safe"



env = ScamEnv()

# 1. reset check

obs = env.reset()
done = False

while not done:
    text = obs["text"]
    prediction = normal_agent(text)

   

    obs,reward,done, _ = env.step(prediction)
   
# print("First message:", obs)

# # 2. manually prediction do
# obs, reward, done, _ = env.step("scam")
# print("After 1 step:", obs, reward, done)

# obs, reward, done, _ = env.step("safe")
# print("After 2 step:", obs, reward, done)

# obs, reward, done, _ = env.step("scam")
# print("After 3 step:", obs, reward, done)

# obs, reward, done, _ = env.step("safe")
# print(reward)

# result = env.reset()
# print(result)

