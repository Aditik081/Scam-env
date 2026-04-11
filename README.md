# Scam Detection OpenEnv

A real-world simulation environment for detecting scam messages using hybrid AI agents (rule-based + LLM).

## Overview

This environment simulates real-world scam detection scenarios where an agent must classify incoming messages as:

* **scam**
* **safe**

The system includes multi-level difficulty tasks and a reward mechanism that evaluates agent performance based on correctness and complexity.



## 🎯 Objective

Build an intelligent agent that can:

* Identify scam patterns (phishing, urgency, financial fraud)
* Distinguish legitimate messages (OTP alerts, service notifications)
* Perform consistently across difficulty levels



## Environment Design

🔹 Observation Space

Each step returns:

```json
{
  "text": "message content",
  "has_link": true/false,
  "has_urgent_words": true/false
}
```


### Action Space

Agent must output:

```
"scam" OR "safe"
```


### Difficulty Levels

 Level           Description                            

 Easy      Obvious scams or clearly safe messages 
 Medium    Slightly ambiguous cases               
 Hard      Realistic, deceptive scam scenarios    



| Condition       | Reward |
|-----------------|--------|
| Correct (easy)  | 0.90   |
| Correct (medium)| 0.85   |
| Correct (hard)  | 0.80   |
| Wrong (easy)    | 0.10   |
| Wrong (medium)  | 0.15   |
| Wrong (hard)    | 0.20   |

This reward structure provides **partial learning signals** and penalizes incorrect predictions.

---

## Environment API

### reset(level=None)

* Starts a new episode
* Optionally specify difficulty level

### `step(action)`

Returns:

(observation, reward, done, info)


##  Baseline Agent (Inference)

The baseline agent uses a **hybrid approach**:

1. **Rule-based system**

   * Handles short/simple messages
   * Detects keywords like OTP, links, urgency

2. **LLM-based classification**

   * Uses Hugging Face router (Mistral-7B)
   * Handles complex or ambiguous cases

3. **Fallback mechanism**

   * If API fails → rule-based prediction ensures stability



##  Evaluation

* Runs over multiple episodes (default: 20)
* Computes average reward across steps
* Final output:


0.82 (example score)




##  Deployment (Docker)

### Build

```bash
docker build -t scam-env .
```

### Run

```bash
docker run -e HF_TOKEN=your_token \
           -e API_BASE_URL=https://router.huggingface.co/v1 \
           -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
           scam-env
```


##  Hugging Face Spaces

* Fully deployable as a Docker Space
* Automatically runs `inference.py`
* Outputs reproducible score


## Setup (Local)

```bash
pip install -r requirements.txt
python inference.py
```


## Real-World Relevance

This environment simulates real scam detection tasks such as:

* Phishing attacks
* Fake job offers
* Fraudulent payment requests
* Urgency-based social engineering


## Constraints

* No UI required (evaluation is backend-based)
* Deterministic reward structure
* Agent must generalize across difficulty levels


##  Future Improvements

* Multi-modal inputs (images, audio scams)
* Advanced NLP feature extraction
* Fine-tuned classification models



##  Author

Developed as part of an OpenEnv-based AI evaluation system for real-world problem solving.



##  Key Highlights

* Real-world use case (not synthetic)
* Multi-level tasks (easy → hard)
* Hybrid agent (rule + LLM)
* Robust fallback mechanism
* Meaningful reward shaping
* Fully reproducible baseline

## 🔗 Live Demo
Hugging Face Space: https://huggingface.co/spaces/aditik08/hackathon


