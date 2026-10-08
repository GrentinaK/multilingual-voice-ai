import asyncio
import uuid

import modal
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    StartFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt

class ModalTunnelManager:
    def __init__(self, app_name: str, cls_name: str):
        self.app_name = app_name
        self.cls_name = cls_name
        self.service = modal.Cls.from_name(app_name, cls_name)()
        self.session_dict = None
        self.session_dict_name = None
        self.function_call = None

    async def get_url(self):
        if self.function_call is None:
            session_id = str(uuid.uuid4())
            self.session_dict_name = f"{session_id}-url-dict"

            self.session_dict = modal.Dict.from_name(
                self.session_dict_name,
                create_if_missing=True,
            )

            await self.session_dict.put.aio("is_running", True)

            self.function_call = await self.service.run_tunnel_client.spawn.aio(
                self.session_dict
            )

        try:
            for _ in range(1200):
                if await self.session_dict.contains.aio("url"):
                    return await self.session_dict.get.aio("url")

                await asyncio.sleep(0.1)

            raise TimeoutError(
                f"Timed out waiting for {self.app_name}.{self.cls_name}"
            )
        except Exception:
            await self.close()
            raise

    async def close(self):
        try:
            if self.session_dict is not None:
                await self.session_dict.put.aio("is_running", False)

            if self.function_call is not None:
                try:
                    await asyncio.wait_for(
                        self.function_call.get.aio(),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, Exception):
                    await self.function_call.cancel.aio()

        finally:
            if self.session_dict_name is not None:
                await modal.Dict.objects.delete.aio(
                    self.session_dict_name
                )

            self.function_call = None
            self.session_dict = None
            self.session_dict_name = None

class ModalWebsocketService(WebsocketService):
    def __init__(
        self,
        websocket_url: str | None = None,
        modal_tunnel_manager: ModalTunnelManager | None = None,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        super().__init__(
            reconnect_on_error=reconnect_on_error,
            **kwargs,
        )

        if not websocket_url and not modal_tunnel_manager:
            raise ValueError(
                "Provide websocket_url or modal_tunnel_manager"
            )

        self.modal_tunnel_manager = modal_tunnel_manager
        self._websocket_url = websocket_url
        self._receive_task = None

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler(
            "on_connection_error",
            error.error,
        )
        await self.push_error(error)

    async def _connect(self):
        if self.modal_tunnel_manager:
            self._websocket_url = await self.modal_tunnel_manager.get_url()
        logger.info(
            f"Connecting to: {self._websocket_url}"
        )

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(
                    self._report_error
                )
            )

        if not self._websocket:
            raise ConnectionError(
                f"Failed to connect to {self._websocket_url}"
            )

        logger.info(f"Connected to: {self._websocket_url}")

    async def _disconnect(self):
        try:
            if self._receive_task:
                await self.cancel_task(
                    self._receive_task,
                    timeout=2.0,
                )
                self._receive_task = None

            await self._disconnect_websocket()

        except Exception as e:
            logger.error(
                f"Error during disconnect: {e}"
            )
        finally:
            if self.modal_tunnel_manager:
                await self.modal_tunnel_manager.close()

    async def _connect_websocket(self):
        try:
            if (
                self._websocket
                and self._websocket.state is State.OPEN
            ):
                return

            self._websocket = await websocket_connect(
                self._websocket_url
            )

            logger.debug(
                "Connected to Modal WebSocket"
            )

        except Exception as e:
            logger.error(
                f"{self} initialization error: {e}"
            )

            self._websocket = None

            await self._call_event_handler(
                "on_connection_error",
                f"{e}",
            )

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug(
                    "Disconnecting from Modal WebSocket"
                )
                await self._websocket.close()

        except Exception as e:
            logger.error(
                f"{self} error closing WebSocket: {e}"
            )

    def _get_websocket(self):
        if self._websocket:
            return self._websocket

        raise Exception("WebSocket not connected")


class ModalWebsocketTTSService(
    TTSService,
    ModalWebsocketService,
):
    def __init__(self, **kwargs):
        TTSService.__init__(
            self,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_text_frames=False,
            # Chatterbox sends nothing until a whole sentence is
            # synthesized; an audio context idle for longer than this is
            # closed and its late audio dropped.
            stop_frame_timeout_s=15.0,
            **kwargs,
        )

        ModalWebsocketService.__init__(
            self,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()


class ModalWebsocketSegmentedSTTService(
    SegmentedSTTService,
    ModalWebsocketService,
):
    def __init__(self, **kwargs):
        SegmentedSTTService.__init__(
            self,
            **kwargs,
        )

        ModalWebsocketService.__init__(
            self,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    @traced_stt
    async def _handle_transcription(
        self,
        transcript: str,
        is_final: bool,
        language: Language | None = None,
    ):
        pass