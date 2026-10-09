"""Per-turn latency table from the bot's Modal logs.

Usage:
    modal app logs multilingual-voice-ai --since 30m --tail 20000 --timestamps > bot.log
    python tools/turn_latency.py bot.log

Groups log events into user turns and reports, per turn: end-to-end delay
(user stops speaking -> bot audio starts), time to end the turn, and the
STT / LLM / TTS time-to-first-output from Pipecat's metrics.
"""
import re, sys, statistics as st
from datetime import datetime

# Must match the VAD stop_secs in server/bot/bot.py.
VAD_STOP_SECS = 0.3
pat = re.compile(r"^\S+ \S+ (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| \w+ +\| [\w.]+:\w+:\d+ - (.*)$")
ev = []
for line in open(sys.argv[1]):
    m = pat.match(line.rstrip("\n"))
    if m:
        ev.append((datetime.fromisoformat(m.group(1)), m.group(2)))

turns, cur, bot_speaking = [], None, False
for t, msg in ev:
    if "Starting. Waiting for StartFrame" in msg:
        session_start = t
    if "User started speaking" in msg:
        if cur is None:
            cur = {"start": t, "segments": [], "texts": [], "interrupt": bot_speaking}
    elif cur is not None and "STT received audio segment" in msg:
        cur["segments"].append(t)
    elif cur is not None and msg.startswith("Received transcription:"):
        cur["texts"].append(msg.split(":", 1)[1].strip())
    elif cur is not None and "STT" in msg and "TTFB" in msg:
        cur.setdefault("stt", []).append(float(re.search(r"TTFB: ([\d.]+)", msg).group(1)))
    elif cur is not None and "User stopped speaking" in msg:
        cur["turn_end"] = t
        cur["strategy"] = re.search(r"strategy: (\S+?)#?\d*\)", msg).group(1) if "strategy" in msg else ""
    elif cur is not None and "OpenAILLMService" in msg and "TTFB" in msg:
        cur["llm"] = float(re.search(r"TTFB: ([\d.]+)", msg).group(1))
    elif cur is not None and "TTSService#" in msg and "TTFB" in msg and "tts" not in cur:
        cur["tts"] = float(re.search(r"TTFB: ([\d.]+)", msg).group(1))
    elif "Empty user turn" in msg and cur is not None:
        cur["empty"] = True
    if "Bot started speaking" in msg:
        bot_speaking = True
        if cur is not None and "turn_end" in cur:
            cur["bot_start"] = t
            turns.append(cur); cur = None
    elif "Bot stopped speaking" in msg:
        bot_speaking = False

rows = []
print(f"{'#':>2} {'e2e':>5} {'endpt':>5} {'stt':>5} {'llm':>5} {'tts':>5} {'segs':>4} {'int':>3}  transcript")
for i, c in enumerate(turns, 1):
    speech_end = c["segments"][-1].timestamp() - VAD_STOP_SECS if c["segments"] else c["turn_end"].timestamp()
    e2e = c["bot_start"].timestamp() - speech_end
    endpt = c["turn_end"].timestamp() - speech_end  # speech end -> turn declared over (includes STT)
    stt = max(c.get("stt", [0]))
    rows.append((e2e, endpt, stt, c.get("llm", 0), c.get("tts", 0)))
    print(f"{i:>2} {e2e:5.2f} {endpt:5.2f} {stt:5.2f} {c.get('llm',0):5.2f} {c.get('tts',0):5.2f} {len(c['segments']):>4} {'Y' if c['interrupt'] else '':>3}  {' | '.join(c['texts'])[:90]}")

if rows:
    def summ(k, name):
        v = sorted(r[k] for r in rows)
        p90 = v[min(len(v)-1, int(round(0.9*(len(v)-1))))]
        print(f"{name:<34} median={st.median(v):.2f}s  p90={p90:.2f}s  min={v[0]:.2f}s  max={v[-1]:.2f}s")
    print(f"\nturns={len(rows)}")
    summ(0, "End-to-end (speech end -> bot audio)")
    summ(1, "Speech end -> turn over (incl. STT)")
    summ(2, "STT TTFB (slowest segment)")
    summ(3, "LLM TTFB")
    summ(4, "TTS TTFB (first sentence)")
