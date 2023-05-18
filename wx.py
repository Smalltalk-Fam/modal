import time
from modal import Secret, Stub, Image, SharedVolume, Mount
from pathlib import Path
import os
import json

CACHE_DIR = "/root/.cache"
URL_DOWNLOADS_DIR = Path(CACHE_DIR, "url_downloads")
RAW_AUDIO_DIR = Path("/tmp", "raw_audio")
PATTI = RAW_AUDIO_DIR / "patti.mp3"

LANGUAGE = "en"
DEVICE = "cuda"
BATCH_SIZE = 24


def download_models():
    t0 = time.time()
    import whisperx

    whisperx.load_model("large-v2", device=DEVICE, compute_type="float16")
    whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
    t1 = time.time()
    print(f"Downloaded model in {t1 - t0:.2f} seconds.")


app_image = (
    Image.debian_slim("3.10.0")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "dacite==1.8.0",
        "ffmpeg-python==0.2.0",
        "git+https://github.com/kousun12/whisperx.git",
        "git+https://github.com/yt-dlp/yt-dlp.git@master",
        "jupyter",
        "loguru==0.6.0",
        "pandas==1.5.3",
        "torch==2.0.1",
        "pytorch-lightning==2.0.2",
        "torchaudio==2.0.2",
    )
    .run_commands("curl https://sh.rustup.rs -sSf | bash -s -- -y")
    .run_commands(". $HOME/.cargo/env && cargo install bore-cli")
    .run_function(
        download_models,
        timeout=60 * 30,
        gpu="a10g",
    )
    .run_commands(
        "python -m pytorch_lightning.utilities.upgrade_checkpoint /root/.cache/torch/whisperx-vad-segmentation.bin",
        gpu="a10g",
    )
    # todo bake this converted version into the model
)

# python -m pytorch_lightning.utilities.upgrade_checkpoint --file .cache/torch/pyannote/models--pyannote--segmentation/snapshots/c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b/pytorch_model.bin

stub = Stub("whisper-x", image=app_image)
volume = SharedVolume().persist("whisper-x-tmp")


@stub.function(
    gpu="a10g",
    shared_volumes={"/sv": volume},
    mounts=[Mount.from_local_file("patti.mp3", remote_path=str(PATTI))],
)
def transcribe():
    t0 = time.time()
    import whisperx
    from whisperx.utils import WriteSRT

    model = whisperx.load_model(
        "large-v2", device=DEVICE, compute_type="float16", language=LANGUAGE
    )
    t1 = time.time()
    print(f"Loaded model in {t1 - t0:.2f} seconds.")

    t0 = time.time()
    audio = whisperx.load_audio(str(PATTI))
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    t1 = time.time()
    print(f"Transcribed in {t1 - t0:.2f} seconds.")

    t0 = time.time()
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )
    t1 = time.time()
    print(f"Aligned in {t1 - t0:.2f} seconds.")

    Path("/sv/results").mkdir(exist_ok=True, parents=True)
    with open("/sv/results/patti.json", "w") as f:
        json.dump(result, f)
    with open("/sv/results/patti-aligned.json", "w") as f:
        json.dump(aligned, f)

    writer = WriteSRT("/sv/results")
    writer(
        result,
        str(PATTI),
        {"highlight_words": True, "max_line_width": 48, "max_line_count": None},
    )


@stub.function(
    concurrency_limit=1,
    timeout=60 * 60,
    secrets=[Secret.from_name("api-secret-key")],
    mounts=[Mount.from_local_file("patti.mp3", remote_path=str(PATTI))],
    gpu="a10g",
)
def run_jupyter():
    import subprocess

    jupyter_process = subprocess.Popen(
        [
            "jupyter",
            "notebook",
            "--no-browser",
            "--allow-root",
            "--port=8888",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
        ],
        env={**os.environ, "JUPYTER_TOKEN": os.environ["API_SECRET_KEY"] or "1234"},
    )

    bore_process = subprocess.Popen(
        ["/root/.cargo/bin/bore", "local", "8888", "--to", "bore.pub"],
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        bore_process.kill()
        jupyter_process.kill()
