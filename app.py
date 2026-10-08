import asyncio
import hashlib
import time
from pathlib import Path

import modal


APP_NAME = "multilingual-voice-ai"
MINUTES = 60
RENEGOTIATION_TIMEOUT_SECONDS = 30

app = modal.App(APP_NAME)


def signalling_queue_name(pc_id: str) -> str:
    # pc_id contains characters Modal object names don't allow ("#"),
    # so derive a stable, valid name from it.
    # Modal names are capped at 64 characters.
    digest = hashlib.sha1(pc_id.encode("utf-8")).hexdigest()[:32]
    return f"{APP_NAME}-sig-{digest}"


bot_image = (
    modal.Image.debian_slim(
        python_version="3.12",
    )
    .apt_install(
        "git",
        "ffmpeg",
    )
    .uv_pip_install(
        # Pinned: an unpinned rebuild pulled in breaking changes (VAD,
        # TTS and transport APIs) that failed silently.
        "pipecat-ai[webrtc,openai,silero,local-smart-turn,kokoro]==1.12.0",
        "websockets",
        "loguru",
        "fastapi[standard]",
    )
    # Bake the Kokoro model into the image instead of downloading it at
    # the start of every session.
    .run_commands(
        "python -c \"from pipecat.services.kokoro.tts import "
        "KOKORO_CACHE_DIR, _ensure_model_files; _ensure_model_files("
        "KOKORO_CACHE_DIR / 'kokoro-v1.0.onnx', "
        "KOKORO_CACHE_DIR / 'voices-v1.0.bin')\""
    )
    .add_local_dir(
        "server",
        remote_path="/root/server",
    )
)


with bot_image.imports():
    from loguru import logger

    from pipecat.transports.smallwebrtc.connection import (
        IceServer,
        SmallWebRTCConnection,
    )

    from server.bot.bot import run_bot


@app.cls(
    image=bot_image,
    timeout=30 * MINUTES,
    max_inputs=1,
    # Kokoro TTS and the VAD/turn models run on CPU in this container.
    cpu=4,
    memory=4096,
)
class ModalVoiceAssistant:

    @modal.method()
    async def run_bot(self, d: modal.Dict):
        try:
            start_time = time.time()

            offer = await d.get.aio("offer")
            ice_servers = await d.get.aio(
                "ice_servers"
            )

            ice_servers = [
                IceServer(**ice_server)
                for ice_server in ice_servers
            ]

            print(
                "Time taken to get ICE servers: "
                f"{time.time() - start_time:.3f} seconds"
            )

            webrtc_connection = (
                SmallWebRTCConnection(
                    ice_servers
                )
            )

            await webrtc_connection.initialize(
                sdp=offer["sdp"],
                type=offer["type"],
            )

            @webrtc_connection.event_handler(
                "closed"
            )
            async def handle_disconnected(
                webrtc_connection,
            ):
                logger.info(
                    "WebRTC connection to bot closed."
                )

            print(
                "Time taken to initialize WebRTC "
                "connection: "
                f"{time.time() - start_time:.3f} seconds"
            )

            # The client re-sends an offer with the same pc_id when it
            # renegotiates (e.g. when the bot asks it to). Those offers
            # must reach this connection, not spawn a new bot.
            queue_name = signalling_queue_name(
                webrtc_connection.pc_id
            )
            signalling_queue = modal.Queue.from_name(
                queue_name,
                create_if_missing=True,
            )
            await signalling_queue.hydrate.aio()

            renegotiation_task = asyncio.create_task(
                self._handle_renegotiations(
                    webrtc_connection,
                    signalling_queue,
                )
            )

            start_time = time.time()

            print("Starting bot process.")

            bot_task = asyncio.create_task(
                run_bot(
                    webrtc_connection,
                )
            )

            print(
                "Time taken to create bot task: "
                f"{time.time() - start_time:.3f} seconds"
            )

            start_time = time.time()

            answer = (
                webrtc_connection.get_answer()
            )

            await d.put.aio(
                "answer",
                answer,
            )

            print(
                "Time taken to put answer: "
                f"{time.time() - start_time:.3f} seconds"
            )

            try:
                await bot_task
            finally:
                renegotiation_task.cancel()
                await modal.Queue.objects.delete.aio(
                    queue_name
                )

        except Exception as e:
            raise RuntimeError(
                "Failed to start bot pipeline: "
                f"{e}"
            )

    async def _handle_renegotiations(
        self,
        webrtc_connection,
        signalling_queue: modal.Queue,
    ):
        while True:
            offer = await signalling_queue.get.aio(
                partition="offer"
            )

            print(
                "Renegotiating peer connection "
                f"{webrtc_connection.pc_id}"
            )

            try:
                await webrtc_connection.renegotiate(
                    sdp=offer["sdp"],
                    type=offer["type"],
                    restart_pc=offer.get(
                        "restart_pc",
                        False,
                    ),
                )
                answer = webrtc_connection.get_answer()
            except Exception as e:
                print(f"Renegotiation failed: {e}")
                answer = {"error": str(e)}

            await signalling_queue.put.aio(
                answer,
                partition="answer",
            )

    @modal.method()
    def ping(self):
        return "pong"


frontend_image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .pip_install(
        "fastapi==0.115.12"
    )
    .add_local_dir(
        Path(__file__).parent / "client/dist",
        remote_path="/frontend",
    )
)


with frontend_image.imports():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles


@app.function(
    image=frontend_image,
)
@modal.asgi_app()
@modal.concurrent(
    max_inputs=100,
)
def serve_frontend():
    web_app = FastAPI()
    static_files = StaticFiles(
        directory="/frontend"
    )

    web_app.mount(
        "/frontend",
        static_files,
    )

    @web_app.get("/")
    async def root():
        return FileResponse(
            "/frontend/index.html"
        )

    @web_app.patch("/offer")
    async def ice_candidates(request: dict):
        # The client waits for ICE gathering, so its candidates are
        # already in the SDP offer; trickled ones can be ignored.
        return {"status": "ok"}

    async def renegotiate(offer: dict):
        signalling_queue = modal.Queue.from_name(
            signalling_queue_name(offer["pc_id"])
        )

        try:
            await signalling_queue.put.aio(
                offer,
                partition="offer",
            )
        except modal.exception.NotFoundError:
            raise HTTPException(
                status_code=404,
                detail="No active bot for this peer connection.",
            )

        answer = await signalling_queue.get.aio(
            partition="answer",
            timeout=RENEGOTIATION_TIMEOUT_SECONDS,
        )

        if answer is None:
            raise HTTPException(
                status_code=504,
                detail="Timed out waiting for renegotiation answer.",
            )

        if "error" in answer:
            raise HTTPException(
                status_code=500,
                detail=answer["error"],
            )

        return answer

    @web_app.post("/offer")
    async def offer(offer: dict):
        if offer.get("pc_id"):
            return await renegotiate(offer)

        start_time = time.time()

        ice_servers = [
            {
                "urls": (
                    "stun:stun.l.google.com:19302"
                ),
            }
        ]

        with modal.Dict.ephemeral() as d:
            await d.put.aio(
                "ice_servers",
                ice_servers,
            )

            await d.put.aio(
                "offer",
                offer,
            )

            print(
                "Time taken to put offer: "
                f"{time.time() - start_time:.3f} seconds"
            )

            start_time = time.time()

            bot_func_call = (
                ModalVoiceAssistant()
                .run_bot
                .spawn(d)
            )

            print(
                "Time taken to spawn bot: "
                f"{time.time() - start_time:.3f} seconds"
            )

            start_time = time.time()

            try:
                while True:
                    answer = await d.get.aio(
                        "answer"
                    )

                    if answer:
                        print(
                            "Time taken to get answer: "
                            f"{time.time() - start_time:.3f} "
                            "seconds"
                        )

                        return answer

                    await asyncio.sleep(0.1)

            except Exception as e:
                print(
                    "Error signaling with bot: "
                    f"{type(e)}: {e}"
                )

                bot_func_call.cancel()
                raise

    return web_app