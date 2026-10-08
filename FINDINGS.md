# Findings

Running notes from building a real-time voice assistant on Modal, grouped
into runtime, infrastructure, and developer tools. Each finding gives the
symptom we saw, the cause, and what we did about it.

Last updated: 2026-10-07 (English baseline, Pipecat implementation).

## Current setup

| Part | Choice | Where it runs |
|---|---|---|
| Orchestration | Pipecat 1.12.0, SmallWebRTC transport | Modal CPU container (4 vCPU), one per session |
| Speech-to-text | faster-whisper `large-v3-turbo`, fp16, forced English | Modal L4 GPU, reached over a websocket tunnel |
| LLM | Qwen3-4B-Instruct-2507 on vLLM 0.21.0 | Modal L4 GPU, OpenAI-compatible endpoint |
| Text-to-speech | Kokoro-82M (`af_heart` voice) via kokoro-onnx | In-process, inside the bot container (CPU) |
| Turn-taking | Silero VAD (0.3 s stop) + Pipecat smart-turn v3 | In-process |
| Client | Pipecat voice-ui-kit console (React) | Static files served from Modal |

Not in use, kept for the Swahili work: Chatterbox multilingual TTS on an L4
GPU (`server/tts.py`, `modal_chatterbox_service.py`).

## Latency baseline

Per-stage time to first output, from one English session (Pipecat metrics):

| Stage | Average | Worst | Samples |
|---|---|---|---|
| STT (Whisper, one segment) | 0.90 s | 1.84 s | 6 |
| LLM (time to first token) | 1.09 s | 1.56 s | 4 |
| TTS (Kokoro, first audio) | 0.79 s | 1.07 s | 2 |

The TTS switch was the biggest single gain:

| | Chatterbox (L4 GPU, separate container) | Kokoro (4 vCPU, in-process) |
|---|---|---|
| First audio, short sentence | 2.5–3.2 s | 0.52 s |
| First audio, 74-character sentence | ~3 s | 1.05 s |
| Model load | 30–40 s | 1.7 s |
| First generation after load | 23 s (warm-up) | no warm-up seen |

End-to-end delay (user stops speaking → bot starts speaking) went from about
5.7 s with Chatterbox to a subjectively much shorter delay with Kokoro. It has
not yet been measured across many turns; that's the next measurement to make.

Not yet measured: cold-start time for a full session, and how latency changes
over a longer conversation.

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

### VAD pause length trades speed for transcript quality
With a segmented STT, each VAD segment is transcribed separately.
- At 0.2–0.3 s, sentences are split into fragments ("setting up my" /
  "Multilingo." / "Voici AI"), which Whisper mis-hears more often.
- At 0.6 s, transcripts are cleaner, but every reply starts later.

The smart-turn model still decides when the whole turn is over, so the LLM gets
the full turn either way; only transcription accuracy suffers. Currently 0.3 s.

### Known issues, not yet fixed
- Custom services log `TTSSettings/STTSettings ... NOT_GIVEN` errors at
  startup; 1.12 expects settings fields to be initialised. Harmless so far.
- Shutting down a session with our websocket services takes about 13 s and
  reports dangling tasks.

## Infrastructure (Modal)

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

## Next

1. Measure end-to-end latency over 10–20 turns, plus cold-start time.
2. Build the same assistant with LiveKit Agents against the same STT and LLM
   servers, and compare it with this one on the measurements above.
3. Swahili and code-switching: a stronger LLM for Swahili, a Swahili-capable
   TTS, and language-switching rules.
