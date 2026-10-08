import asyncio
import json
import threading
import time

import modal


LANGUAGE_ID = "sw"
UVICORN_PORT = 8000

app = modal.App("multilingual-tts")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "git+https://github.com/resemble-ai/chatterbox.git",
        "fastapi[standard]",
        "uvicorn[standard]",
    )
)


@app.cls(
    image=image,
    gpu="L4",
)
class ChatterboxTTS:
    @modal.enter()
    def load_model(self):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.tunnel_ctx = None
        self.tunnel = None
        self.websocket_url = None

        start_time = time.perf_counter()

        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device="cuda",
            t3_model="v3",
        )

        print(
            f"TTS model load time: "
            f"{time.perf_counter() - start_time:.3f} seconds"
        )

        # The first generation is ~10x slower than later ones (warm-up).
        # Pay that cost here, before the websocket URL is handed out, so
        # the first reply of a session isn't delayed past the bot's timeout.
        start_time = time.perf_counter()
        self.model.generate("Habari.", language_id=LANGUAGE_ID)
        print(
            f"TTS warm-up time: "
            f"{time.perf_counter() - start_time:.3f} seconds"
        )

    @modal.enter()
    def start_server(self):
        import numpy as np
        import uvicorn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from starlette.websockets import WebSocketState

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            prompt_queue = asyncio.Queue()
            audio_queue = asyncio.Queue()

            async def recv_loop(ws, prompt_queue):
                while True:
                    msg = await ws.receive_text()

                    try:
                        json_data = json.loads(msg)

                        if json_data.get("type") != "prompt":
                            continue

                        await prompt_queue.put(json_data)

                    except Exception:
                        continue

            async def inference_loop(prompt_queue, audio_queue):
                while True:
                    prompt_msg = await prompt_queue.get()

                    text = prompt_msg["text"]
                    language_id = prompt_msg.get(
                        "language_id",
                        LANGUAGE_ID,
                    )

                    start_time = time.perf_counter()

                    wav = self.model.generate(
                        text,
                        language_id=language_id,
                    )

                    tts_latency = time.perf_counter() - start_time

                    print(
                        f"TTS synthesis latency: "
                        f"{tts_latency:.3f} seconds"
                    )

                    audio = wav.squeeze().cpu().numpy()
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767).astype(np.int16)

                    await audio_queue.put(audio.tobytes())

            async def send_loop(audio_queue, ws):
                while True:
                    audio = await audio_queue.get()

                    await ws.send_bytes(audio)

                    print(
                        f"Sending audio data: "
                        f"{len(audio)} bytes"
                    )

            await ws.accept()
            tasks = []

            try:
                tasks = [
                    asyncio.create_task(
                        recv_loop(ws, prompt_queue)
                    ),
                    asyncio.create_task(
                        inference_loop(
                            prompt_queue,
                            audio_queue,
                        )
                    ),
                    asyncio.create_task(
                        send_loop(audio_queue, ws)
                    ),
                ]

                await asyncio.gather(*tasks)

            except WebSocketDisconnect:
                print("WebSocket disconnected")

            except Exception as e:
                print("WebSocket exception:", e)

            finally:
                if (
                    ws.application_state
                    is WebSocketState.CONNECTED
                ):
                    await ws.close(code=1011)

                for task in tasks:
                    if not task.done():
                        try:
                            task.cancel()
                            await task
                        except asyncio.CancelledError:
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

        self.tunnel_ctx = modal.forward(UVICORN_PORT)
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

    @modal.method()
    def get_websocket_url(self):
        return self.websocket_url

    @modal.method()
    async def run_tunnel_client(self, d: modal.Dict):
        try:
            print(f"Sending Chatterbox WebSocket URL: {self.websocket_url}")

            await d.put.aio("url", self.websocket_url)

            while await d.get.aio("is_running"):
                await asyncio.sleep(1.0)

            print("Chatterbox tunnel session finished.")

        except Exception as e:
            print(
                f"Chatterbox tunnel session error: "
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