import base64
import json
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.utils.time import time_now_iso8601

from .modal_services import ModalWebsocketSegmentedSTTService


class ModalWhisperSegmentedSTTService(
    ModalWebsocketSegmentedSTTService
):
    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, str):
                await self.push_frame(
                    TranscriptionFrame(
                        message,
                        "",
                        time_now_iso8601(),
                    )
                )

                await self._handle_transcription(
                    message,
                    True,
                )

                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

                logger.info(
                    f"Received transcription: {message}"
                )

            else:
                logger.warning(
                    "Received non-string message: "
                    f"{type(message)}"
                )

    async def run_stt(
        self,
        audio: bytes,
    ) -> AsyncGenerator[Frame, None]:

        logger.info(f"STT received audio segment: {len(audio)} bytes")

        if not self._websocket:
            logger.error(
                "Not connected to Whisper STT."
            )

            yield ErrorFrame(
                "Not connected to Whisper STT.",
                fatal=True,
            )
            return

        await self.start_ttfb_metrics()

        try:
            audio_msg = {
                "type": "audio",
                "audio": base64.b64encode(
                    audio
                ).decode("utf-8"),
            }

            await self._websocket.send(
                json.dumps(audio_msg)
            )

        except Exception as e:
            logger.error(
                f"Failed to send audio to Whisper: {e}"
            )

            yield ErrorFrame(
                f"Failed to send audio to Whisper: {e}"
            )
            return

        yield None