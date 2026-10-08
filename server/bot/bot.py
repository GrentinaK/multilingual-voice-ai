import asyncio

import httpx

from .services.modal_services import ModalTunnelManager

from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
    LocalSmartTurnAnalyzerV3,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)

from pipecat.processors.frameworks.rtvi import (
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import (
    SmallWebRTCConnection,
)
from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .services.modal_whisper_service import (
    ModalWhisperSegmentedSTTService,
)


AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_OUTPUT_SAMPLE_RATE = 24000

STT_APP_NAME = "multilingual-stt"
STT_CLASS_NAME = "WhisperSTT"

# Kokoro runs in-process on the bot's CPUs: ~0.5s to first audio per
# sentence, vs ~3s for Chatterbox on a separate GPU container.
TTS_VOICE = "af_heart"

LLM_BASE_URL = (
    "https://nazariusgrentina--multilingual-llm-qwenllm"
    ".us-east.modal.direct/v1"
)

LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

LLM_WARMUP_TIMEOUT_SECONDS = 10 * 60


async def wait_for_llm():
    # The vLLM server scales to zero. A request to a cold server starts a
    # container but immediately fails with 503, so poll until it's serving.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + LLM_WARMUP_TIMEOUT_SECONDS

    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                response = await client.get(f"{LLM_BASE_URL}/models")
                if response.status_code == 200:
                    logger.info("LLM is ready.")
                    return
                logger.info(
                    f"Waiting for LLM (status {response.status_code})..."
                )
            except httpx.HTTPError as e:
                logger.info(f"Waiting for LLM ({e})...")

            if loop.time() > deadline:
                raise TimeoutError("Timed out waiting for LLM to start")

            await asyncio.sleep(2)


async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
):
    stt_tunnel_manager = ModalTunnelManager(
        app_name=STT_APP_NAME,
        cls_name=STT_CLASS_NAME,
    )

    # Bring up STT and the LLM in parallel before the pipeline starts.
    # Cold GPU containers can take longer than the pipeline's start
    # timeout, which would otherwise tear the session down.
    try:
        await asyncio.gather(
            stt_tunnel_manager.get_url(),
            wait_for_llm(),
        )
    except Exception:
        await stt_tunnel_manager.close()
        raise

    # Pipecat 1.x ignores vad_analyzer/turn_analyzer on TransportParams;
    # they belong on the user aggregator (see LLMUserAggregatorParams).
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=AUDIO_INPUT_SAMPLE_RATE,
        audio_out_enabled=True,
        audio_out_sample_rate=AUDIO_OUTPUT_SAMPLE_RATE,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=transport_params,
    )

    stt = ModalWhisperSegmentedSTTService(
        modal_tunnel_manager=stt_tunnel_manager,
        sample_rate=AUDIO_INPUT_SAMPLE_RATE,
    )

    llm = OpenAILLMService(
        model=LLM_MODEL,
        api_key="not-needed",
        base_url=LLM_BASE_URL,
        params=OpenAILLMService.InputParams(
            extra={
                "stream": True,
            },
        ),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=TTS_VOICE,
            language=Language.EN,
        ),
        sample_rate=AUDIO_OUTPUT_SAMPLE_RATE,
    )

    rtvi = RTVIProcessor()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Your replies are "
                "spoken aloud, so answer in English in one or two short "
                "sentences of plain text: no emojis, markdown, or lists."
            ),
        }
    ]

    context = LLMContext(messages)

    user_aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    # Silence that ends a VAD segment. Each segment is one
                    # Whisper call; the smart-turn model decides when the
                    # turn as a whole is over. Lower = faster replies but
                    # more fragmented transcripts.
                    stop_secs=0.3,
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams()
                        ),
                    )
                ],
            ),
        ),
    )

    assistant_aggregator = LLMAssistantAggregator(
        context,
    )

    processors = [
        transport.input(),
        rtvi,
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]

    pipeline = Pipeline(
        processors=processors,
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        observers=[
            RTVIObserver(rtvi),
        ],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info(
            "Pipecat client ready."
        )
        await rtvi.set_bot_ready()

    @transport.event_handler(
        "on_client_disconnected"
    )
    async def on_client_disconnected(
        transport,
        client,
    ):
        logger.info(
            "Pipecat client disconnected."
        )

        await task.cancel()

    @transport.event_handler(
        "on_client_connected"
    )
    async def on_client_connected(
        transport,
        client,
    ):
        logger.info(
            "Pipecat client connected."
        )

    runner = PipelineRunner()

    try:
        await runner.run(task)
    finally:
        await stt_tunnel_manager.close()

    logger.info("Pipeline task finished.")