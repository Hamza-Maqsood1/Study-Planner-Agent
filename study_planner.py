import re
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import chainlit as cl

MEMORY_PATH = Path("planner_memory.json")
QUOTES = [
    "Focus on the next small win.",
    "Consistency beats intensity.",
    "Short breaks keep the mind sharp.",
    "Done is better than perfect.",
    "Tiny progress compounds."
]

def load_memory():
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_plan": []}

def save_memory(mem: dict):
    MEMORY_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")

def save_last_plan(df: pd.DataFrame):
    mem = load_memory()
    mem["last_plan"] = df.to_dict(orient="records")
    save_memory(mem)

def load_last_plan() -> pd.DataFrame:
    mem = load_memory()
    if mem.get("last_plan"):
        return pd.DataFrame(mem["last_plan"])
    return pd.DataFrame()

def normalize_weights(priorities: Dict[str, float]) -> Dict[str, float]:
    arr = np.array(list(priorities.values()), dtype=float)
    if arr.sum() == 0:
        return {k: 1.0/len(priorities) for k in priorities}
    arr = arr / arr.sum()
    return {k: float(v) for k, v in zip(priorities.keys(), arr)}

def distribute_time(total_minutes: int, weights: Dict[str, float], min_block: int = 25) -> Dict[str, int]:
    if total_minutes <= 0:
        return {k: 0 for k in weights}
    raw = {k: int(total_minutes * w) for k, w in weights.items()}
    for k in raw:
        if raw[k] == 0 and total_minutes >= min_block:
            raw[k] = min_block
    diff = total_minutes - sum(raw.values())
    keys = list(weights.keys())
    i = 0
    while diff != 0 and keys:
        k = keys[i % len(keys)]
        if diff > 0:
            raw[k] += 1
            diff -= 1
        else:
            if raw[k] > min_block:
                raw[k] -= 1
                diff += 1
        i += 1
    return raw

def split_into_sessions(minutes: int, focus_len: int = 50, short_break: int = 10) -> List[Tuple[int, str]]:
    plan = []
    remaining = minutes
    while remaining > 0:
        if remaining <= focus_len:
            plan.append((remaining, 'study'))
            remaining = 0
        else:
            plan.append((focus_len, 'study'))
            remaining -= focus_len
            if remaining > 0:
                b = min(short_break, remaining)
                plan.append((b, 'break'))
                remaining -= b
    return plan

def build_schedule(subjects: List[str],
                   priorities: Dict[str, float],
                   total_minutes: int,
                   start_time: datetime = None,
                   focus_len: int = 45,
                   short_break: int = 10,
                   long_break_every: int = 3,
                   long_break_len: int = 20) -> pd.DataFrame:
    start_time = start_time or datetime.now().replace(second=0, microsecond=0)
    weights = normalize_weights(priorities)
    per_subject = distribute_time(total_minutes, weights, min_block=min(25, total_minutes))
    rows = []
    cur = start_time
    study_block_counter = 0
    for subj in subjects:
        sessions = split_into_sessions(per_subject[subj], focus_len, short_break)
        for dur, kind in sessions:
            end = cur + timedelta(minutes=dur)
            rows.append({
                "start": cur.strftime("%H:%M"),
                "end": end.strftime("%H:%M"),
                "duration_min": dur,
                "type": kind,
                "subject": subj if kind == 'study' else ""
            })
            cur = end
            if kind == 'study':
                study_block_counter += 1
                if study_block_counter % long_break_every == 0:
                    l_end = cur + timedelta(minutes=long_break_len)
                    rows.append({
                        "start": cur.strftime("%H:%M"),
                        "end": l_end.strftime("%H:%M"),
                        "duration_min": long_break_len,
                        "type": "break",
                        "subject": ""
                    })
                    cur = l_end
    return pd.DataFrame(rows)

def pick_quote() -> str:
    return random.choice(QUOTES)

@cl.on_chat_start
async def start():
    msg = (
        "**AI Study Planner Agent**\n\n"
        "Type `plan` to create a new schedule.\n"
        "Commands: `last` (show last plan), `save` (save plan), `reset` (clear memory), `example` (demo)."
    )
    await cl.Message(content=msg).send()

async def wizard() -> dict:
    await cl.Message(content="List your subjects and priorities like `Math:3, Python:2, AI:4` (higher = more time).").send()
    subjects_event = await cl.AskUserMessage(content="Subjects & priorities:", timeout=600).send()
    sp_text = (subjects_event and subjects_event["content"] or "").strip()
    pairs = [p.strip() for p in sp_text.split(",") if p.strip()]
    subjects, priorities = [], {}
    for p in pairs:
        if ":" in p:
            name, val = p.split(":", 1)
            name = name.strip()
            try:
                pr = float(val.strip())
            except ValueError:
                pr = 1.0
            subjects.append(name)
            priorities[name] = pr
        else:
            name = p.strip()
            subjects.append(name)
            priorities[name] = 1.0
    if not subjects:
        subjects = ["Math", "Python", "AI"]
        priorities = {"Math": 3, "Python": 2, "AI": 4}

    await cl.Message(content="How many **total minutes** do you have today? (e.g., 180)").send()
    mins_event = await cl.AskUserMessage(content="Total minutes:", timeout=600).send()
    try:
        total_minutes = int((mins_event and mins_event["content"] or "180").strip())
    except ValueError:
        total_minutes = 180

    await cl.Message(content="Optional start time in 24h `HH:MM` (or leave blank for now).").send()
    st_event = await cl.AskUserMessage(content="Start time (optional):", timeout=600).send()
    st_txt = (st_event and st_event["content"] or "").strip()
    start_dt = None
    if st_txt:
        try:
            hh, mm = [int(x) for x in st_txt.split(":")]
            now = datetime.now().replace(second=0, microsecond=0)
            start_dt = now.replace(hour=hh, minute=mm)
        except Exception:
            start_dt = None

    return {"subjects": subjects, "priorities": priorities, "total_minutes": total_minutes, "start_dt": start_dt}

def df_to_markdown(df: pd.DataFrame) -> str:
    lines = ["| Start | End | Min | Type | Subject |", "|---|---:|---:|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['start']} | {r['end']} | {int(r['duration_min'])} | {r['type']} | {r['subject']} |")
    return "\n".join(lines)

@cl.on_message
async def handle_message(message: str):
    text = message.strip().lower()

    if text == "example":
        subjects = ["Math", "Python", "AI"]
        priorities = {"Math": 3, "Python": 2, "AI": 4}
        df = build_schedule(subjects, priorities, 180)
        save_last_plan(df)
        md = df_to_markdown(df)
        await cl.Message(content=f"🗓 **Example Plan (3h)**\n\n{md}\n\n**Quote:** _{pick_quote()}_").send()
        return

    if text == "last":
        df = load_last_plan()
        if df.empty:
            await cl.Message(content="No saved plan yet. Type `plan` to create one.").send()
            return
        md = df_to_markdown(df)
        await cl.Message(content=f"🗂 **Last Saved Plan**\n\n{md}").send()
        return

    if text == "save":
        df = load_last_plan()
        if df.empty:
            await cl.Message(content="Nothing to save yet. Create a plan first (`plan`).").send()
            return
        save_last_plan(df)
        await cl.Message(content="💾 Plan saved.").send()
        return

    if text == "reset":
        save_memory({"last_plan": []})
        await cl.Message(content="🧹 Cleared saved plan.").send()
        return

    if text == "plan":
        params = await wizard()
        df = build_schedule(params["subjects"], params["priorities"], params["total_minutes"], start_time=params["start_dt"])
        save_last_plan(df)
        md = df_to_markdown(df)
        await cl.Message(content=f"🗓 **Your Study Plan**\n\n{md}\n\n**Quote:** _{pick_quote()}_").send()
        return

    await cl.Message(content="Type `plan` to create a study plan, or `example` to see a demo.").send()
