import subprocess

import modal


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# Pinned so the weights baked into the image and the ones vLLM loads match.
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App("multilingual-llm")


vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    # Download the weights (7.5 GB) at build time, so no container, not even
    # the first after a deploy, spends ~30 s downloading them at startup.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_NAME}', revision='{MODEL_REVISION}')\""
    )
    .env(
        {
            # Everything is in the image now; don't call Hugging Face at startup.
            "HF_HUB_OFFLINE": "1",
            # vLLM's FlashInfer sampler compiled a GPU kernel from source on
            # every cold start (~108 s). Use PyTorch's sampler instead.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
)


@app.server(
    image=vllm_image,
    gpu="L4",
    port=VLLM_PORT,
    startup_timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    target_concurrency=1,
    unauthenticated=True,
)
class QwenLLM:
    @modal.enter()
    def start_vllm(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--tokenizer-revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--enable-chunked-prefill",
            "--max-num-seqs",
            "1",
            "--max-model-len",
            "8192",
            "--enable-prefix-caching",
        ]

        print("Starting vLLM:")
        print(" ".join(cmd))

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop_vllm(self):
        self.process.terminate()