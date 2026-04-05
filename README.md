# Scam Detection RL Environment

## Overview

A reinforcement learning environment for classifying messages as **scam** or **safe**.
It simulates real-world fraud scenarios such as phishing, fake offers, and urgent scams.

## Features

* Multi-level tasks: easy, medium, hard
* Feature-based observations (link detection, urgency detection)
* Reward-based evaluation (0–1 range)
* Hybrid inference (LLM + rule-based fallback)

## Actions

* `scam`
* `safe`

## Output

The environment returns a score between **0 and 1** based on prediction accuracy.

## Usage

Run locally:

```bash
python inference.py
```

Run with Docker:

```bash
docker build -t scam-env .
docker run scam-env
```
