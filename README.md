---

title: Dataops Env
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false

---

# 🧠 DataOps Gym

### ⚡ The First Hallucination-Aware Data Cleaning Environment

> ❌ Most systems ask: *“Did you fix the data?”*
> ✅ We ask: *“Did you think before fixing?”*

---

# 🚨 THE PROBLEM

**60–80% of a data scientist’s time is spent cleaning data.**

But current systems:

* blindly fix values
* hallucinate corrections
* ignore contradictions
* break real-world logic
  
---

> 💡 **Wrong data is worse than missing data.**

---

# 🧠 WHAT THIS PROJECT DOES

DataOps Gym is a **step-based OpenEnv environment** where an AI agent:

1. Detects semantic inconsistencies
2. Fixes data **only when confident**
3. Outputs **"cannot determine"** when uncertain
4. Maintains **cross-record consistency**
5. Learns through **reward-based feedback**

---

Each step teaches the agent:

* when to fix ✅
* when to abstain ⚠️
* when to say “I don’t know” 🧠

---

# 🧩 ACTION SPACE

All actions must follow strict JSON format:

```json
{
  "action_type": "detect_issue | fix_value | cannot_determine | skip",
  "record_id": "string",
  "field": "string",
  "value": "string",
  "confidence": 0.0
}
```

---

## 🔥 Key Innovation

👉 `cannot_determine` is a **first-class action**

---

# 🧠 WHY THIS IS DIFFERENT

| Traditional Systems | DataOps Gym            |
| ------------------- | ---------------------- |
| Fix everything      | Fix only when safe     |
| Always answer       | Can abstain            |
| Ignore confidence   | Confidence-aware       |
| Single-row logic    | Cross-record reasoning |
| Output-based        | Behavior-based         |

---

# 💰 REWARD SYSTEM

---

## ✅ Rewards

* correct reasoning
* safe corrections
* correct uncertainty
* consistency across records

---

## ❌ Penalties

* hallucinated fixes 🚫
* overconfidence 🚫
* over-correction 🚫
* inconsistency 🚫

---

### 🔥 Core Principle

> **“Better to not fix than to fix incorrectly.”**

---

# 📊 FINAL SCORING (0–1)

```text
task_score =
  0.5 * normalized_record_score
+ 0.2 * (1 - hallucination_rate)
+ 0.15 * uncertainty_accuracy
+ 0.15 * consistency_score
```

---

# 📉 METRICS

| Metric                  | Description            |
| ----------------------- | ---------------------- |
| 🧠 Hallucination Rate   | Wrong invented fixes   |
| ⚖️ Uncertainty Accuracy | Correct abstentions    |
| 🔗 Consistency Score    | Cross-record reasoning |

---

# 🧪 TASKS
> ⚡ Each task is carefully designed to evaluate **reasoning, restraint, and reliability** — not just accuracy.

---

## 🟢 EASY — *Foundational Data Hygiene*

<p align="left">
  <b>“Can the agent fix obvious issues without breaking anything?”</b>
</p>

* Basic inconsistencies
* Missing values
* Duplicate records

---

## 🟡 MEDIUM — *Contextual Reasoning & Ambiguity*

<p align="left">
  <b>“Can the agent reason across records and handle uncertainty?”</b>
</p>

* Cross-table inconsistencies
* Identity ambiguity
* Data normalization

---

## 🔴 HARD — *Real-World Data Chaos*

<p align="left">
  <b>“Can the agent survive contradictions, missing context, and unsolvable data?”</b>
</p>

* Multi-table conflicts
* Temporal inconsistencies
* Non-fixable contradictions

---

> 🔥 **Difficulty is not about complexity — it's about uncertainty.**

| Level  | Focus |
|--------|------|
| 🟢 Easy   | Precision on clear signals |
| 🟡 Medium | Reasoning under ambiguity |
| 🔴 Hard   | Decision-making under uncertainty |

---

# 🧪 EXAMPLE FAILURE LOG

```json
{
  "record_id": "T3",
  "error_type": "hallucination",
  "details": "assigned value without evidence",
  "confidence": 0.9
}
```

---

# 🚀 QUICK START

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run Server

```bash
python -m server.app
```

---

## Run Baseline

```bash
python inference.py
```

---

## Example Output

```text
easy   → 0.73
medium → 0.55
hard   → 0.38
```

> ⚠️ Replace with your actual results

---

# 🌐 API ENDPOINTS

| Endpoint  | Description       |
| --------- | ----------------- |
| `/reset`  | Start new episode |
| `/step`   | Take action       |
| `/state`  | Get current state |
| `/health` | Health check      |

---

# 🐳 DOCKER

```bash
docker build -t dataops-gym .
docker run -p 7860:7860 dataops-gym
```

---

# 🧠 DESIGN PRINCIPLES

1. Prefer uncertainty over hallucination
2. Penalize confident mistakes
3. Avoid over-correction
4. Enforce cross-record consistency
5. Reward safe reasoning

---

# 🏆 BENCHMARK (EXPECTED)

| Task   | Score       |
| ------ | ----------- |
| Easy   | 0.65 – 0.85 |
| Medium | 0.45 – 0.65 |
| Hard   | 0.05 – 0.40 |

---

# 📌 USE CASES

* AI data pipelines
* automated ETL validation
* financial data cleaning
* healthcare record validation
* LLM safety benchmarking

---

# 🏁 FINAL TAKEAWAY

> 🧠 **The future of AI is not about answering everything.**
> ⚡ **It’s about knowing when NOT to answer.**

---

# 🔥 TAGLINE

> **“We built a system that teaches AI when NOT to change data.”**

---




