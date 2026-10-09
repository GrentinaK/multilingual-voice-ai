# Findings

Running notes from building a real-time voice assistant on Modal, grouped
into runtime, infrastructure, and developer tools. Each finding gives the
symptom we saw, the cause, and what we did about it.

Last updated: 2026-10-09 (English baseline, Pipecat implementation; two
measured test sessions and the first cold-start fix).

## Current setup

| Part | Choice | Where it runs |
|---|---|---|
| Orchestration | Pipecat 1.12.0, SmallWebRTC transport | Modal CPU container (4 vCPU), one per session |
| Speech-to-text | faster-whisper `large-v3-turbo`, fp16, forced English | Modal L4 GPU, reached over a websocket tunnel |
| LLM | Qwen3-4B-Instruct-2507 on vLLM 0.21.0 | Modal L4 GPU, OpenAI-compatible endpoint |
| Text-to-speech | Kokoro-82M (`af_heart` voice) via kokoro-onnx | In-process, inside the bot container (CPU) |
| Turn-taking | Silero VAD (0.3 s stop) + Pipecat smart-turn v3 | In-process |
| Client | Pipecat voice-ui-kit console (React) | Static files served from Modal |

Not in use, kept for the multilingual work: Chatterbox multilingual TTS on an
L4 GPU (`server/tts.py`, `modal_chatterbox_service.py`).

## How the setup evolved

The first working version (2026-10-07) and the current one (2026-10-09):

| | Initial setup | Current setup |
|---|---|---|
| Language | Swahili and English, Whisper picks per segment | English, Whisper forced to English |
| Text-to-speech | Chatterbox multilingual, own L4 GPU container | Kokoro-82M, in the bot container on CPU |
| LLM image | Weights downloaded from Hugging Face at every start; FlashInfer compiled a kernel at every start | Weights built into the image (pinned revision); FlashInfer sampler off |
| Dependencies | Unpinned; a rebuild pulled in Pipecat 1.12 | All pinned to verified versions |
| VAD and turn detection | Silently ignored (set where 1.12 no longer reads them) | Set on the user aggregator |
| STT container timeout | Modal default, 300 s (sessions died at 5 min) | 30 min |

What that changed, as measured:

| Measure | Initial setup | Current setup |
|---|---|---|
| **LLM container start → ready** | **281 s** | **121–138 s** |
| Click Connect → ready, everything cold | ~5 min 20 s | not yet measured end to end; the LLM part is now ~2 min |
| Click Connect → ready, GPUs warm | ~12 s (server side) | 17 s (browser side) |
| Click Connect → ready, only Whisper cold | — | 32 s (browser side) |
| TTS, time to first audio | 2.5–3.2 s (Chatterbox) | 0.7–1.1 s median (Kokoro) |
| User stops → bot speaks, median | ~5.7 s (one Chatterbox turn) | 4.4–4.6 s |

"Server side" is from the bot's logs; "browser side" is from the client's
event log and includes starting the bot container. The cold end-to-end
number for the current setup still needs a test after the LLM has scaled to
zero.

## Latency measurements

Two test sessions, English, natural conversation with pauses, trailing off,
and interruptions. Numbers come from the bot's logs and Pipecat metrics; the
parser is described under Developer tools.

### Test 1 (2026-10-08): one cold start (13 turns) and one warm reconnect (4 turns)

#### Session start (click Connect → agent ready)

| | Time | Where it goes |
|---|---|---|
| Cold (nothing running) | **~5 min 20 s** | almost all waiting for the LLM container (see Infrastructure) |
| Warm (STT and LLM containers still up) | **~12 s** | ~8 s booting a fresh bot container (one per session), the rest WebRTC setup |

#### Per turn (17 turns)

| Measure | Median | p90 | Min | Max |
|---|---|---|---|---|
| **End-to-end** (user stops speaking → bot audio starts) | **4.43 s** | 6.19 s | 1.53 s | 6.82 s |
| Speech end → turn declared over (includes STT) | 1.26 s | 3.30 s | 0.52 s | 3.39 s |
| STT, per segment (warm, forced English) | 0.25 s | 0.28 s | | 0.44 s |
| LLM, time to first token | 0.41 s | 0.62 s | 0.39 s | 1.97 s (first turn) |
| TTS, time to first audio | 1.07 s | 2.13 s | | 2.46 s |

"User stops speaking" is estimated as the end of the last VAD segment minus
the 0.3 s VAD pause.

What the numbers say:
- **Deciding the turn is over is the biggest and most variable cost.** It
  takes either ~0.5 s or ~3.3 s, nothing in between (see Runtime: smart-turn
  fallback). Removing the 3.3 s cases alone would bring the median close to
  2.5 s.
- **TTS is second**, and depends on which machine the bot lands on (see
  Infrastructure: CPU varies between containers).
- **STT and the LLM are fast once warm** (0.25 s and 0.41 s medians).
- Interruptions (3 turns) worked: the bot stopped and answered the new turn.

### Test 2 (2026-10-09): two warm-GPU sessions (18 turns)

The LLM was still warm from a benchmark, so neither session was a full cold
start; in the first, Whisper started cold.

| Measure | Median | p90 | Min | Max |
|---|---|---|---|---|
| **End-to-end** (user stops speaking → bot audio starts) | **4.55 s** | 5.57 s | 1.53 s | 6.32 s |
| Speech end → turn declared over (includes STT) | 3.28 s | 3.37 s | 0.59 s | 3.43 s |
| STT, per segment | 0.37 s | 0.48 s | 0.29 s | 0.67 s |
| LLM, time to first token | 0.37 s | 0.52 s | 0.32 s | 1.61 s (first turn) |
| TTS, time to first audio | 0.70 s | 1.71 s | | 3.10 s |

| Session start (browser side) | Time |
|---|---|
| Whisper cold, LLM warm | 32 s (16 s to "connected", 16 s more for Whisper) |
| Everything warm | 17 s (almost all starting the bot container) |

What this session added:
- **Turn detection dominated**: 11 of 18 turns waited out the 3 s smart-turn
  fallback (6 of 17 in test 1), so the median turn-end time rose to 3.28 s.
- **The conversation itself held up**: "Can you say that again?" got a repeat,
  and "Finish what you wanted to tell me about going out first" picked the
  earlier topic back up after an interruption. The user interrupted 7 times
  and every one worked.
- **TTS speed was even across both containers** (17–19 ms per character),
  unlike test 1 (13–14 vs 27–52).
- **The long-sentence problem became a reliability problem** (see Runtime:
  audio contexts), and a fragmented transcript reversed a meaning (see
  Runtime: VAD pause length).

### Earlier comparison: Chatterbox vs Kokoro

The TTS switch was the biggest single gain:

| | Chatterbox (L4 GPU, separate container) | Kokoro (4 vCPU, in-process) |
|---|---|---|
| First audio, short sentence | 2.5–3.2 s | 0.52 s |
| First audio, 74-character sentence | ~3 s | 1.05 s |
| Model load | 30–40 s | 1.7 s |
| First generation after load | 23 s (warm-up) | no warm-up seen |

One Chatterbox turn measured about 5.7 s end-to-end, against the 4.43 s
median above.

## Runtime (Pipecat)

### Silent breaking changes are the main risk
The bot image installed `pipecat-ai` without a version. A rebuild picked up
1.12, and several things broke without any error:

- **VAD and turn detection were ignored.** `TransportParams(vad_analyzer=...,
  turn_analyzer=...)` is accepted but does nothing in 1.12 (the extra fields are
  dropped). Symptom: speech was never detected and the STT never ran. Fix: pass
  `vad_analyzer` and `user_turn_strategies` to `LLMUserAggregatorParams`.
- **The TTS interface changed.** `run_tts(text)` became `run_tts(text,
  context_id)`, and audio now flows through per-utterance *audio contexts*
  (`create_audio_context`, `append_to_audio_context`, `remove_audio_context`).
  Symptom: `run_tts() takes 2 positional arguments but 3 were given` on every
  reply.

Fix for the whole class: pin exact versions in every image (done), and when
upgrading, diff the changelog and test each custom service.

### WebRTC renegotiation needs signalling support
If the browser connects before the pipeline has finished starting, and the
browser's offer includes video slots (the Pipecat client always adds them),
Pipecat asks the browser to renegotiate. The browser then sends a second offer
with the same `pc_id`. Our `/offer` endpoint started a new bot for every offer,
so the browser ended up paired with a second bot whose data channel never
opened.

Symptom: client shows connected, agent never does; logs show two `POST /offer`
about 1 s apart and `Data channel not established within 10s`. Fix: route
offers that carry a `pc_id` to the existing bot (through a Modal Queue named
after the connection) and call `connection.renegotiate()`. The browser also
sends ICE candidates with `PATCH /offer`, which needs a handler.

### The pipeline must start within 20 seconds
`PipelineTask` stops the pipeline if startup takes longer than
`start_timeout_secs` (default 20 s). Our STT and TTS services opened their
remote connections during startup, so a cold GPU container blew the limit and
the session dropped right after connecting. Fix: connect to remote services
(and wait for the LLM) in parallel *before* starting the pipeline.

### Audio contexts close if no audio arrives in time
A TTS audio context is closed when no audio arrives within
`stop_frame_timeout_s`; audio arriving later is dropped. A non-streaming TTS
that returns a whole sentence at once (Chatterbox, about 3 s) needs this set
well above its per-sentence time; ours was 1 s.

It caught Kokoro too, which uses the default limit of 3 s. In test 2 the LLM
produced a 209-character first sentence; Kokoro took 3.07 s, Pipecat gave up
at 3 s ("TTS context … completed with no audio", shown as an error in the
client), and the audio arrived 0.04 s later. Pipecat reopened the context and
the reply played, but only just. Kokoro's time grows with sentence length, so
long sentences are both slow to start and at risk of being dropped. Fixes: a
longer `stop_frame_timeout_s` for Kokoro, and keeping the LLM's sentences
short (the prompt asks for one or two short sentences; Qwen doesn't always
comply).

### VAD pause length trades speed for transcript quality
With a segmented STT, each VAD segment is transcribed separately.
- At 0.2–0.3 s, sentences are split into fragments ("setting up my" /
  "Multilingo." / "Voici AI"), which Whisper mis-hears more often.
- At 0.6 s, transcripts are cleaner, but every reply starts later.

The smart-turn model still decides when the whole turn is over, so the LLM gets
the full turn either way; only transcription accuracy suffers. Currently 0.3 s.
In both tests one turn was split into as many as 8 segments.

Accuracy loss can change the meaning. In test 2 the user said things had not
been working smoothly; Whisper heard "and things have not been worried." and
"working smoothly." as separate segments, and the bot answered "That's great
to hear!". Options: a longer pause (fewer fragments, later replies),
re-transcribing the whole turn once it ends (more accurate, adds delay), or a
streaming STT that keeps context across the turn.

### Smart-turn's 3-second fallback makes turn-ending bimodal
When the smart-turn model judges a turn complete, the bot moves on about
0.5 s after the user stops. When it judges the turn *incomplete* (the user
trailed off: "I wonder what I should do today…", "So…"), it waits for
`SmartTurnParams.stop_secs`, which defaults to **3 s**. Together with the
0.3 s VAD pause, that's the 3.3 s seen in 6 of 17 turns in test 1 and 11 of
18 in test 2, and those are most of the slow end-to-end turns. The model also
errs the other way: "Okay, I forgot to…" was judged complete and answered
after 0.65 s, cutting the user off. Lowering `stop_secs` (e.g. to 1.5–2 s)
trades fewer long waits for more cases of cutting the user off; not yet
tuned.

### Barge-in: the server reacts in ~5 ms; the delay is in detecting speech
The user noticed the bot takes a moment to stop when interrupted. From the
logs, once Pipecat registers "user started speaking", the bot stops within
3–14 ms (one case 288 ms) across 9 interruptions. After that, at most one
40 ms chunk of audio is still queued for sending: SmallWebRTC's sender waits
for each chunk to go out before queuing the next.

So the delay is before detection, which the logs can't time (they don't
record when the user actually began speaking). Known contributors:
- Silero VAD needs **0.2 s** of confident speech (`start_secs`, confidence
  0.7, minimum volume 0.6) before it fires. These are Pipecat's defaults,
  unchanged by us.
- The user was on the **laptop speakers**. Browser echo cancellation lowers
  the mic while the bot plays, which can delay reaching VAD's thresholds.
- The user's audio travels to Modal's us-east region first.

Moving VAD into the user aggregator (required by Pipecat 1.12) didn't change
these. To separate them: barge in with headphones (if faster, echo
cancellation is the main factor), then try `start_secs` ≈ 0.1 s (faster, but
coughs and noise interrupt more).

### The LLM doesn't know today's date or recent events
Asked the date, Qwen answered "Thursday, June 20, 2024" in test 1 and
"April 5, 2024" in test 2, each time with confidence, and named a president
from its training data. Its training data ends in 2024, so it has
no way to know either. Two different problems, two different fixes:
- **The date** doesn't need a search: the server knows the current date and
  can put it in the system prompt at the start of each session.
- **Recent facts** (news, who holds an office) need a live source, such as a
  web-search tool the LLM can call. Qwen3 supports tool calling through vLLM.

### Known issues, not yet fixed
- Custom services log `TTSSettings/STTSettings ... NOT_GIVEN` errors at
  startup; 1.12 expects settings fields to be initialised. Harmless so far.
- Shutting down a session with our websocket services takes about 13 s and
  reports dangling tasks.

## Infrastructure (Modal)

### The LLM cold start is ~5 minutes, and model size is a small part of it
Measured on two cold vLLM containers (vLLM 0.21, Qwen3-4B, L4); total
4 min 50 s and 4 min 48 s. Second boot:

| Step | Time | What drives it |
|---|---|---|
| Container scheduled and started | 8 s | Modal |
| Python imports and model config | 54 s | Software: vLLM and PyTorch are imported twice (API server and engine process) from a fresh disk |
| Downloading weights (7.5 GB) | 31 s | **Model size** |
| Loading weights onto the GPU | 4 s | **Model size** |
| `torch.compile` | 58 s | Software: vLLM compiles the model graph for speed |
| **FlashInfer compiling its sampling kernel** | **108 s** | Software: FlashInfer builds a GPU kernel from source on first use |
| KV cache, CUDA graphs, server start | ~10 s | |
| **Total** | **4 min 48 s** | |

The 108 s step logs nothing. It was identified from file timestamps inside
the running container: FlashInfer's `cached_ops/sampling` folder was written
the exact second vLLM resumed logging. It happens because the image is
NVIDIA's CUDA *devel* image, which includes the compiler, so FlashInfer
compiles instead of failing over to a prebuilt kernel. vLLM only uses it for
top-p/top-k sampling.

Root cause, in order of weight:
1. **The serving stack does one-time preparation at every start (~60%).**
   `torch.compile` and FlashInfer's kernel build are investments that pay off
   on a long-running server. They're unrelated to model size.
2. **The architecture repeats all of it every time.** With scale-to-zero,
   each container starts from the image alone: no downloaded weights, no
   compiled graphs, no compiled kernels survive from the previous container.
   This is the reason the work is repeated, not the work itself.
3. **Model size is only ~12%** (download plus load, 35 s). A bigger model
   would grow this part, but not the rest.

#### Step A: weights in the image, FlashInfer sampler off (2026-10-09)
Changes in `server/llm.py`: download the weights during the image build
(pinned model revision), set `HF_HUB_OFFLINE=1` at runtime, and set
`VLLM_USE_FLASHINFER_SAMPLER=0`. Measured with `tools/llm_coldstart.py` and
the container logs:

| | Before | Step A |
|---|---|---|
| Container start → vLLM ready | 281 s | **121 s, 138 s** (two boots) |
| Model loading (download + load) | 41 s | 7–8 s |
| FlashInfer kernel build | 108 s | gone |
| `torch.compile` | 58 s | 53–61 s (unchanged) |
| Generation speed | 28.6 tok/s | 28.8 tok/s |
| Time to first token (5 requests) | 0.149 s | 0.241 s |

About 55% off the cold start, with no change in generation speed. The time
to first token moved from 0.15 s to 0.24 s, but with five samples and one
outlier it may be noise; not confirmed. What's left is mostly the imports
(~50 s) and `torch.compile` (~55–60 s).

Measuring this taught two things about Modal:
- **A deploy replaces warm containers immediately.** Deploying a new version
  started new-version containers to replace the running one, so a "cold
  start" right after a deploy may already be in progress.
- **Stopping a container gets it replaced** while the app is inside its
  scale-down window (15 min here): the autoscaler keeps the capacity it last
  needed. A true request-to-ready cold start can only be measured after the
  window runs out. Container start → "Starting vLLM server" in the logs is a
  workable stand-in (it measured 281 s where the request-based number was
  290 s).

Ways to cut it further, by what they remove:
- Persist `~/.cache/huggingface`, `~/.cache/vllm`, and `~/.cache/flashinfer`
  on a Modal Volume: removes the download and most compilation on later cold
  starts (they still run once, to fill the cache).
- Turn off the FlashInfer sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`): removes
  the 108 s outright, at a sampling speed cost that should be negligible for
  one conversation at a time. Not yet measured.
- `--enforce-eager`: skips compilation and CUDA graphs, but makes generation
  slower every turn.
- Modal memory snapshots: could also cut the 54 s of imports.
- `min_containers=1`: no cold start at all, at the cost of an always-on GPU.

### Every session boots a new bot container (~8 s)
The bot class allows one session per container (`max_inputs=1`), so even a
"warm" connect waits ~8 s for a new CPU container to start. That's now the
floor for session start. Seen from the browser in test 2, a fully warm
connect took 17 s, almost all of it before the client even reached
"connected" (the `/offer` request waiting for the new bot container). A warm
pool of bot containers (`min_containers`/`buffer_containers`) would remove
it cheaply, since they're CPU only.

### Whisper also downloads its model on every cold start
The STT container logs "unauthenticated requests to the HF Hub" at every
start: faster-whisper downloads `large-v3-turbo` each time. In test 2 that was
about 10 s of Whisper's 14 s start. The same fix as the LLM's Step A applies:
download the model during the image build.

### CPU speed varies between containers, and in-process TTS feels it
The two sessions ran identical code on identical `cpu=4` containers, yet
Kokoro ran at **27–52 ms per character** in the first and **13–14 ms per
character** in the second (the benchmark gave 14). The slow container was
slow from its first turn; it didn't degrade. Modal's `cpu` setting reserves
CPU time but not a particular CPU type.

### The bot's audio breaks up, cause not yet known
The user heard the bot "breaking up" in **both** sessions. The second ran on
the fast container, so slow CPU is at most a contributing factor, not the
cause. The server logs show no warnings and each reply plays as one
continuous stretch, so the server side gives no clue.

Remaining candidates: network loss or jitter between the browser and Modal
(us-east, direct UDP via STUN, no TURN relay), or the pacing of audio sent
by the Python WebRTC stack (aiortc). Next step: capture
`chrome://webrtc-internals` during a session. Packet loss, jitter, and
concealed samples on the incoming audio separate a network problem from a
server one.

### Scale-to-zero returns 503 instead of waiting
A request to a `modal.server` app with no running container starts one, but
the request itself fails at once with `503 {'error': 'no upstreams
available'}`. vLLM then takes about a minute to become ready. Fix: at session
start, poll the LLM's `/v1/models` until it returns 200 before marking the bot
ready. Alternative: `min_containers=1`, which costs an always-on GPU.

### The default 300-second function timeout ends long sessions
The STT tunnel is a long-running Modal method that lives as long as the
conversation. Under Modal's default 300 s timeout it was cancelled, and the
STT connection dropped five minutes into every session. Fix: set the class
`timeout` to match the bot's 30-minute session limit.

### Slim images lack CUDA libraries
faster-whisper (through CTranslate2) loads cuBLAS and cuDNN for CUDA 12 at
runtime, and `debian_slim` doesn't include them. The model *loaded* fine on the
GPU; the failure (`libcublas.so.12 is not found`) only appeared on the first
transcription, so the container looked healthy until audio arrived. Fix:
install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` and set
`LD_LIBRARY_PATH`. Pin `ctranslate2` too, since its version decides which CUDA
libraries it expects.

### GPU models need warm-up before they're ready
Chatterbox's first generation took 23 s; later ones took 2.5–3 s. Running one
dummy generation during container startup moves that cost out of the first
user turn.

### A small CPU model in-process can beat a GPU service
Kokoro on the bot's 4 vCPUs was about 6× faster to first audio than Chatterbox
on an L4. It also removed a container, a network hop, and a cold start.
kokoro-onnx runs inference in a worker thread, so the event loop (audio input,
VAD) stalled at most 26 ms during a long sentence. Good English quality;
Swahili is a separate question.

### Smaller notes
- Cross-container signalling works with `modal.Dict` and `modal.Queue`.
  Object names are limited to 64 characters and some characters (Pipecat's
  `pc_id` contains `#`), so names are derived from a hash.
- `modal.Queue.from_name(..., create_if_missing=True)` is lazy; call
  `hydrate()` before handing the name to another container.
- The frontend endpoint uses blocking Modal calls in async code
  (`Dict.ephemeral()`, `.spawn()`), which Modal warns about. Not yet changed.

## Developer tools

- **Modal logs.** `modal app logs <app> --since 15m --tail 20000 --timestamps`,
  piped through `grep`. The bot logs an audio idle warning every 2 s, so
  filtering that out first is essential. `--tail` is capped at 20,000 lines.
- **Inspecting a live container.** `modal container list`, then
  `modal container exec <id> -- ...` with `pip install py-spy` and
  `py-spy dump`. This told a frozen event loop apart from an idle one. It
  showed the bot was idle, which pointed to the VAD setting.
- **Testing images on a real GPU.** A throwaway `modal run` script using the
  same image definition, to check the CUDA fix and the language-detection API
  before deploying. Catch: the container can't import local modules that
  aren't added to the image; define the image inline or guard imports with
  `modal.is_local()`.
- **Contract tests for custom services.** A fake websocket server that mimics
  the real model server (one binary reply per prompt, 2 s delay), driven
  through a real Pipecat pipeline. This checked the new TTS audio-context code
  without a GPU.
- **Benchmarking before switching.** Kokoro was benchmarked on the same CPU
  setup as the bot, including an event-loop stall check, before it replaced
  Chatterbox.
- **The installed library source is the best documentation.** Reading Pipecat
  1.12's own code in `.venv` gave the exact parameter names and behaviour
  (start timeout, audio contexts, renegotiation); a reference service of a
  similar shape (Resemble AI's) was the template for ours.
- **The client's event panel.** The voice-ui-kit console shows `botReady`,
  user started/stopped speaking, and errors (such as the LLM 503) with
  timestamps that line up with server logs, which is the quickest way to see
  which stage a failure is in.
- **Verifying pins.** `pip freeze` run inside each built image, to pin the
  versions actually running rather than guessing.
- **A per-turn latency table from logs.** `tools/turn_latency.py` reads the
  bot's log (`modal app logs ... --since 30m --tail 20000 > file`) and groups
  events into turns: VAD segments, transcripts, turn end, Pipecat TTFB
  metrics, and "Bot started speaking". It produced the latency tables above.
  Pipecat's metrics are per stage only, so end-to-end delay has to be
  rebuilt from timestamps like this. Catch: `modal app logs --since` reads an
  absolute time as *local* time; relative values (`30m`) are safer.
- **Comparing containers, not just turns.** Splitting TTS speed by session
  (ms per character) is what showed the CPU difference between containers.
  Averaged over the whole run it just looked like "TTS is sometimes slow".

## Next

1. Fixes found in the measured sessions, each re-measured with the same
   scripts:
   - Cold start, Step B: the remaining `torch.compile` (~55–60 s) and imports
     (~50 s); then a full end-to-end cold test after the LLM scales to zero.
   - Whisper: download the model during the image build.
   - Smart-turn fallback: lower `stop_secs` from 3 s and compare.
   - Long sentences: a longer Kokoro `stop_frame_timeout_s`, and shorter
     replies.
   - Barge-in: test with headphones, then a lower VAD `start_secs`.
   - Choppy audio: capture `chrome://webrtc-internals` during a session.
   - Put today's date in the system prompt; web search for recent facts
     later.
   - A warm pool of bot containers, to cut session start.
2. Build the same assistant with LiveKit Agents against the same STT and LLM
   servers, and compare it with this one on the measurements above.
3. Multilingual support and code-switching: a stronger LLM beyond English, a
   multilingual TTS, and language-switching rules.
