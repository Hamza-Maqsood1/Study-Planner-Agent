# 📚 AI Study Planner Agent

![Framework](https://img.shields.io/badge/Framework-Chainlit-purple)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Technique](https://img.shields.io/badge/Technique-Pomodoro%20%2B%20Weighted%20Scheduling-green)
![UI](https://img.shields.io/badge/UI-Interactive%20Chat-orange)

## 📌 Overview

An intelligent study scheduling assistant that generates **personalized, time-blocked study plans** based on subjects, available time, and user-defined priorities. Built with a custom weighted scheduling algorithm and Pomodoro-style session management, deployed via **Chainlit** as an interactive chat interface.

---

## 🎯 Problem Statement

- Students struggle to allocate study time effectively across multiple subjects
- Generic schedules ignore individual subject priorities and learning patterns
- Goal: Build an intelligent planner that generates optimized, personalized schedules in real time through a conversational interface

---

## ⚙️ Core Algorithm

### 1. Priority-Based Time Distribution
```python
# Weights normalized across subjects
weights = normalize_weights(priorities)  # numpy normalization
per_subject = distribute_time(total_minutes, weights, min_block=25)
```
- User assigns priority scores (e.g., `Math:3, Python:2, AI:4`)
- Weights normalized using NumPy higher priority = more allocated time
- Minimum block size enforced (25 min) to prevent ineffective micro-sessions

### 2. Pomodoro Session Splitting
- **Focus block:** 45 minutes of deep work
- **Short break:** 10 minutes after each focus block
- **Long break:** 20 minutes after every 3 study blocks
- Remaining time handled gracefully no wasted minutes

### 3. Schedule Generation
- Precise start/end times calculated using `datetime` + `timedelta`
- Full schedule output as Pandas DataFrame
- Rendered as Markdown table in Chainlit UI

### 4. Memory Persistence
- Last generated plan saved to `planner_memory.json`
- Session-aware: retrieve previous plans with `last` command

---

## 💬 Chat Commands

| Command | Action |
|---|---|
| `plan` | Start interactive wizard enter subjects, time, priorities |
| `example` | See a demo 3-hour plan (Math, Python, AI) |
| `last` | Retrieve last saved plan |
| `save` | Save current plan to memory |
| `reset` | Clear saved memory |

---

## 📋 Example Output

```
Input: Math:3, Python:2, AI:4 | 180 minutes | Start: 09:00

| Start | End   | Min | Type  | Subject |
|-------|-------|-----|-------|---------|
| 09:00 | 09:45 | 45  | study | AI      |
| 09:45 | 09:55 | 10  | break |         |
| 09:55 | 10:40 | 45  | study | AI      |
| 10:40 | 10:50 | 10  | break |         |
| 10:50 | 11:20 | 20  | break |         | ← long break
| 11:20 | 12:05 | 45  | study | Math    |
...
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **UI:** Chainlit (interactive chat interface)
- **Scheduling:** Custom weighted algorithm (NumPy)
- **Data Handling:** Pandas DataFrame
- **Persistence:** JSON memory storage

---

## 🚀 How to Run

```bash
# Install dependencies
pip install chainlit numpy pandas

# Run the app
chainlit run study_planner.py
```

Open browser at `http://localhost:8000` type `plan` or `example` to start.

---

## 📁 Repository Structure

```
├── study_planner.py        # Main app scheduling algorithm + Chainlit UI
├── planner_memory.json     # Auto-generated stores last plan
└── README.md               # Documentation
```

---

## 🔮 Future Work

- Integrate with Google Calendar API for direct schedule export
- Add spaced repetition algorithm (SM-2) for revision scheduling
- Build subject difficulty estimator using past performance data
- Deploy as hosted Chainlit app

---

## 👤 Author

**Hamza Maqsood**
BS Artificial Intelligence University of Management and Technology, Lahore
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/hamza-maqsood1)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Hamza-Maqsood1)
