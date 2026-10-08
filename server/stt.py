import asyncio
import base64
import json
import time
import threading

import modal


SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ("sw", "en")
# Set to None to auto-detect between SUPPORTED_LANGUAGES per segment.
FORCED_LANGUAGE = "en"
MINUTES = 60
UVICORN_PORT = 8000

app = modal.App("multilingual-stt")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Pinned to the versions verified working; ctranslate2 is
        # faster-whisper's engine and decides which CUDA libraries it needs.
        "faster-whisper==1.2.1",
        "ctranslate2==4.8.2",
        "fastapi==0.142.4",
        "numpy==1.26.4",
        "uvicorn[standard]==0.54.0",
        # CTranslate2 (used by faster-whisper) loads cuBLAS/cuDNN for
        # CUDA 12 at runtime; debian_slim doesn't ship them.
        "nvidia-cublas-cu12==12.9.2.10",
        "nvidia-cudnn-cu12==9.27.0.42",
    )
    .env(
        {
            "LD_LIBRARY_PATH": (
                "/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:"
                "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"
            ),
        }
    )
)


@app.cls(
    image=image,
    gpu="L4",
    # run_tunnel_client lives as long as a conversation; Modal's default
    # 300s timeout was killing the tunnel 5 minutes into every session.
    # Matches the bot's own 30-minute session limit.
    timeout=30 * MINUTES,
)
class WhisperSTT:
    @modal.enter()
    def load_model(self):
        from faster_whisper import WhisperModel

        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        self.model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )

    @modal.enter()
    def start_server(self):
        import uvicorn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from starlette.websockets import WebSocketState

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()

            async def recv_loop(ws, audio_queue):
                while True:
                    msg = await ws.receive_text()

                    print("Whisper received WebSocket message")

                    try:
                        json_data = json.loads(msg)

                        if json_data.get("type") != "audio":
                            continue

                        data = json_data["audio"]
                        data = base64.b64decode(
                            data.encode("utf-8")
                        )

                    except Exception as e:
                        print(f"Whisper audio decoding error: {e}")
                        continue

                    await audio_queue.put(data)

            async def inference_loop(
                audio_queue,
                transcription_queue,
            ):
                import numpy as np

                while True:
                    audio_bytes = await audio_queue.get()

                    # Raw 16-bit PCM -> normalized float32 waveform.
                    audio_data = np.frombuffer(
                        audio_bytes,
                        dtype=np.int16,
                    )

                    audio_data = (
                        audio_data.astype(np.float32) / 32768.0
                    )

                    start_time = time.perf_counter()

                    language = (
                        FORCED_LANGUAGE
                        or self.detect_language(audio_data)
                    )

                    segments, _ = self.model.transcribe(
                        audio_data,
                        language=language,
                    )

                    transcript = "".join(
                        segment.text for segment in segments
                    ).strip()

                    end_time = time.perf_counter()

                    stt_latency = end_time - start_time

                    print(
                        f"STT inference latency: "
                        f"{stt_latency:.3f} seconds"
                    )

                    await transcription_queue.put(
                        transcript
                    )

            async def send_loop(
                transcription_queue,
                ws,
            ):
                while True:
                    transcript = (
                        await transcription_queue.get()
                    )

                    print(
                        f"Sending transcription: "
                        f"{transcript}"
                    )

                    await ws.send_text(transcript)

            await ws.accept()

            tasks = []

            try:
                tasks = [
                    asyncio.create_task(
                        recv_loop(ws, audio_queue)
                    ),
                    asyncio.create_task(
                        inference_loop(
                            audio_queue,
                            transcription_queue,
                        )
                    ),
                    asyncio.create_task(
                        send_loop(
                            transcription_queue,
                            ws,
                        )
                    ),
                ]

                await asyncio.gather(*tasks)

            except WebSocketDisconnect:
                print("WebSocket disconnected")

            except Exception as e:
                print("WebSocket exception:", e)

            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()

                if tasks:
                    await asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    )

                if (
                    ws.application_state
                    is WebSocketState.CONNECTED
                ):
                    try:
                        await ws.close()
                    except (RuntimeError, WebSocketDisconnect):
                        pass

        def run_uvicorn():
            uvicorn.run(
                self.web_app,
                host="0.0.0.0",
                port=UVICORN_PORT,
            )

        self.server_thread = threading.Thread(
            target=run_uvicorn,
            daemon=True,
        )

        self.server_thread.start()

        self.tunnel_ctx = modal.forward(
            UVICORN_PORT
        )

        self.tunnel = self.tunnel_ctx.__enter__()

        self.websocket_url = (
            self.tunnel.url.replace(
                "https://",
                "wss://",
            )
            + "/ws"
        )

        print(
            f"WebSocket URL: "
            f"{self.websocket_url}"
        )

    def detect_language(self, audio_data) -> str:
        # Unrestricted detection on short clips often picks an unrelated
        # language (e.g. Swahili transcribed as Russian), so only choose
        # between the languages this assistant supports.
        _, _, all_probs = self.model.detect_language(audio_data)
        probs = dict(all_probs)
        language = max(
            SUPPORTED_LANGUAGES,
            key=lambda lang: probs.get(lang, 0.0),
        )
        print(
            "Detected language: "
            f"{language} "
            + ", ".join(
                f"{lang}={probs.get(lang, 0.0):.2f}"
                for lang in SUPPORTED_LANGUAGES
            )
        )
        return language

    @modal.method()
    def get_websocket_url(self):
        return self.websocket_url

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            print(f"Sending Whisper WebSocket URL: {self.websocket_url}")

            await d.put.aio("url", self.websocket_url)

            while await d.get.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Whisper tunnel session finished.")

        except Exception as e:
            print(
                f"Whisper tunnel session error: "
                f"{type(e).__name__}: {e}"
            )
            raise

    @modal.asgi_app()
    def webapp(self):
        return self.web_app

    @modal.exit()
    def exit(self):
        if self.tunnel_ctx:
            self.tunnel_ctx.__exit__(
                None,
                None,
                None,
            )

            self.tunnel_ctx = None
            self.tunnel = None
            self.websocket_url = None