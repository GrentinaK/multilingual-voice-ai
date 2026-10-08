import json
import traceback
from collections import deque
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from .modal_services import ModalWebsocketTTSService


class ModalChatterboxTTSService(
    ModalWebsocketTTSService
):
    """Chatterbox TTS over a Modal websocket.

    The server answers each prompt with exactly one binary message holding
    the whole utterance, in the order the prompts were sent. We track which
    audio context each in-flight prompt belongs to, so incoming audio can be
    routed to it (Pipecat 1.x audio contexts), and close a context once its
    turn has been flushed and all of its prompts have come back.
    """

    def __init__(
        self,
        language_id: str = "sw",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._language_id = language_id
        # context_id of each prompt sent but not yet answered, in send order.
        self._pending_prompts: deque[str] = deque()
        # Contexts whose turn has ended (no more text will be sent).
        self._flushed_contexts: set[str] = set()

    def can_generate_metrics(self) -> bool:
        return True

    async def flush_audio(self, context_id: str | None = None):
        if not context_id:
            return

        self._flushed_contexts.add(context_id)
        await self._maybe_finish_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        # Audio still in flight for this context will be dropped on arrival.
        self._flushed_contexts.discard(context_id)
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    async def _maybe_finish_context(self, context_id: str):
        if context_id not in self._flushed_contexts:
            return
        if context_id in self._pending_prompts:
            return

        self._flushed_contexts.discard(context_id)

        if self.audio_context_available(context_id):
            await self.append_to_audio_context(
                context_id,
                TTSStoppedFrame(context_id=context_id),
            )
            await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if not self._pending_prompts:
                logger.warning(
                    "Received Chatterbox audio with no pending prompt; "
                    "dropping it."
                )
                continue

            context_id = self._pending_prompts.popleft()

            try:
                logger.info(
                    "Received audio data of length "
                    f"{len(message)} bytes"
                )

                if self.audio_context_available(context_id):
                    await self.append_to_audio_context(
                        context_id,
                        TTSAudioRawFrame(
                            audio=message,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        ),
                    )

                await self._maybe_finish_context(context_id)

            except Exception as e:
                logger.error(
                    "Error handling audio: "
                    f"{e}:{traceback.format_exc()}"
                )

                await self.push_error(
                    ErrorFrame(
                        f"Error handling audio: {e}"
                    )
                )

    async def run_tts(
        self,
        text: str,
        context_id: str,
    ) -> AsyncGenerator[Frame | None, None]:

        if not self._websocket:
            logger.error(
                "Not connected to Chatterbox TTS."
            )

            yield ErrorFrame(
                "Not connected to Chatterbox TTS.",
                fatal=True,
            )
            return

        try:
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)
                await self.start_ttfb_metrics()
                yield TTSStartedFrame(context_id=context_id)

            tts_msg = {
                "type": "prompt",
                "text": text.strip(),
                "language_id": self._language_id,
            }

            self._pending_prompts.append(context_id)

            await self._websocket.send(
                json.dumps(tts_msg)
            )

        except Exception as e:
            logger.error(
                "Failed to send prompt to "
                f"Chatterbox TTS: {e}"
            )

            if self._pending_prompts and self._pending_prompts[-1] == context_id:
                self._pending_prompts.pop()

            yield ErrorFrame(
                "Failed to send prompt to "
                f"Chatterbox TTS: {e}"
            )
            yield TTSStoppedFrame(context_id=context_id)
            return

        yield None
