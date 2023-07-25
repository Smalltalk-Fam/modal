import time
from modal import Secret, Stub, Image, Dict, Mount, NetworkFileSystem
import json
from pathlib import Path
import re
from typing import NamedTuple
from logger import log
from network import save_file, download_vid_audio, notify_callback, put_s3
import os
import base64
from io import BytesIO

from transcribe_args import args, all_models, TranscribeConfig

CACHE_DIR = "/cache"
TRANSCRIPTIONS_DIR = Path(CACHE_DIR, "transcriptions")
URL_DOWNLOADS_DIR = Path(CACHE_DIR, "url_downloads")
MODEL_DIR = Path(CACHE_DIR, "model")
RAW_AUDIO_DIR = Path("/mounts", "raw_audio")


def download_models():
    t0 = time.time()
    import whisperx

    whisperx.load_model(MODEL, device=DEVICE, compute_type="float16")
    # TODO load more alignment models
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
        "jiwer==2.5.1",
        "jupyter",
        "loguru==0.6.0",
        "pandas==1.5.3",
        "torch==2.0.0",
        "pytorch-lightning==2.0.2",
        "safetensors==0.3.1",
        "torchaudio==2.0.1",
        "openai-whisper==20230314",
        "openai",
    )
    .run_function(
        download_models,
        timeout=60 * 30,
        gpu="a10g",
    )
    .pip_install("boto3==1.26.137")
)

stub = Stub("wx", image=app_image)
stub.running_jobs = Dict.new({})

volume = NetworkFileSystem.persisted("fan-transcribe-volume")
silence_end_re = re.compile(
    r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
)


class RunningJob(NamedTuple):
    model: str
    start_time: int
    source: str


def create_mounts():
    fname = args.filename
    if not fname:
        return []
    name = Path(fname).name if fname else ""
    return [Mount.from_local_file(fname, remote_path=str(RAW_AUDIO_DIR / name))]


if stub.is_inside():
    mounts = []
    gpu = None
else:
    mounts = create_mounts()
    gpu = args.gpu


def summarize_transcript(text: str):
    log.info("Summarizing transcript")
    import openai

    openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"]
    chunk_size = 31_000
    summaries = []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    is_multi = len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        log.info(f"Summary chunk {idx + 1}/{len(chunks)}")
        if not is_multi:
            msg = f"Summarize the following conversation:\n\n{chunk}"
        elif idx == 0:
            msg = f"This is the first part of a conversation. Summarize it. Begin with 'The conversation starts ':\n\n{chunk}"
        elif idx == len(chunks) - 1:
            msg = f"This is the last part of a conversation. Summarize it:\n\n{chunk}"
        else:
            msg = f"This is part {idx +1}/{len(chunks)} of a conversation. Continue your summary of the convo. Start your response with a variation of 'In the next part' or 'After that,', but don't use those words exactly.\n\nNext part:\n\n{chunk}"

        messages = [
            {
                "role": "system",
                "content": f"You are an AI that summarizes {'multi-part ' if is_multi else ''}conversations.",
            },
        ]
        if len(summaries) > 0:
            messages.extend([{"role": "assistant", "content": s} for s in summaries])
        messages.append({"role": "user", "content": msg})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.85,
                frequency_penalty=1,
                n=1,
            )
            summary = response["choices"][0]["message"]["content"].strip()
            summaries.append(summary)
        except Exception as e:
            log.info(f"Error: {e}")

    if len(summaries) and len(text) >= 1000 * 12:
        summary_text = "\n".join(summaries)
        messages = [
            {
                "role": "user",
                "content": f"Condense this conversation summary into bullet points:\n\n{summary_text}",
            },
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5,
                frequency_penalty=1.0,
                n=1,
            )
            bullet = response["choices"][0]["message"]["content"].strip()
            summaries.insert(
                0, f"##### Overview:\n\n{bullet}\n\n##### Extended summary:"
            )
        except Exception as e:
            log.info(f"Error: {e}")

    return "\n\n".join(summaries)


@stub.function(
    keep_warm=1,
    timeout=60,
    secrets=[
        Secret.from_name("openai-secret-key"),
        Secret.from_name("openai-org-id"),
    ],
)
def llm_respond(text: str):
    log.info("Running LLM response")
    import openai

    openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"]
    messages = [
        {
            "role": "system",
            "content": """Do not use "As an AI language model" in your responses. Be concise.""",
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.85,
            n=1,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "I don't know"


def make_title(from_text: str, what: str = "conversation transcription"):
    if len(from_text) == 0:
        return ""
    log.info("Summarizing transcript")
    import openai

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"{from_text}\n\n=======================\n\nCome up with a short memorable (non-clickbait) title that describes the above {what}",
            }
        ],
        temperature=0.8,
        frequency_penalty=1,
        n=1,
    )
    t = response["choices"][0]["message"]["content"].strip()
    if t.startswith('"') and t.endswith('"'):
        return t[1:-1]
    return t


LANGUAGE = "en"
DEVICE = "cuda"
BATCH_SIZE = 24
MODEL = "large-v2"


@stub.function(
    gpu="a10g",
    network_file_systems={CACHE_DIR: volume},
    timeout=60 * 10,
)
def transcribe_x(file_path: str, result_path: str, skip_align: bool = False):
    t0 = time.time()
    import whisperx

    model = whisperx.load_model(
        "large-v2",
        device=DEVICE,
        compute_type="float16",
        language=LANGUAGE,
        asr_options={
            "initial_prompt": "A conversation on Smalltalk",
            "compression_ratio_threshold": 2,
        },
    )
    t1 = time.time()
    print(f"Loaded model in {t1 - t0:.2f} seconds.")

    t0 = time.time()
    audio = whisperx.load_audio(file_path)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    t1 = time.time()
    print(f"Transcribed in {t1 - t0:.2f} seconds.")

    t2 = time.time()
    model_a, meta = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    if skip_align:
        aligned = result
    else:
        aligned = whisperx.align(
            result["segments"],
            model_a,
            meta,
            audio,
            DEVICE,
            return_char_alignments=False,
        )

    log.info(f"Aligned in {time.time() - t2:.2f}s")
    full_text = ""
    for r in aligned["segments"]:
        full_text += r["text"]
    full_text = full_text.strip()
    aligned["full_text"] = full_text
    aligned["model"] = MODEL
    try:
        del aligned["word_segments"]
    except KeyError:
        print("no word segments")

    with open(result_path, "w") as f:
        json.dump(aligned, f)

    return aligned


@stub.function(
    image=app_image,
    network_file_systems={CACHE_DIR: volume},
    timeout=60 * 12,
    secrets=[
        Secret.from_name("openai-secret-key"),
        Secret.from_name("openai-org-id"),
        Secret.from_name("aws-clip-manager"),
        Secret.from_name("api-secret-key"),
    ],
    keep_warm=1,
)
def start_transcribe(
    cfg: TranscribeConfig,
    notify=None,
    summarize=False,
    byte_string=None,
):
    import whisper
    from modal import container_app

    model_name = cfg.model
    force = cfg.force or False

    job_source, job_id = cfg.identifier()
    log.info(f"Starting job {job_id}, source: {job_source}, args: {cfg}")
    # cache the model in the shared volume
    model = all_models[model_name]

    # noinspection PyProtectedMember
    whisper._download(whisper._MODELS[model.name], str(MODEL_DIR), False)

    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    URL_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if byte_string:
        b = BytesIO(base64.b64decode(byte_string.encode("ISO-8859-1")))
        with open(URL_DOWNLOADS_DIR / cfg.filename, "wb") as file:
            file.write(b.getbuffer())
        log.info(f"Saved bytes to {URL_DOWNLOADS_DIR / cfg.filename}")

    log.info(f"Using model '{model.name}' with {model.params} parameters.")

    result_path = TRANSCRIPTIONS_DIR / f"{job_id}.json"
    use_llm = bool(byte_string)
    if result_path.exists() and not force:
        log.info(f"Transcription already exists for {job_id}, returning from cache.")
        with open(result_path, "r") as f:
            result = json.load(f)
            if notify:
                notify_webhook(result, notify)
            return result
    else:
        container_app.running_jobs[job_id] = RunningJob(
            model=model.name, start_time=int(time.time()), source=job_source
        )
        if cfg.url:
            save_file(cfg.url, URL_DOWNLOADS_DIR / job_id)
        elif cfg.video_url:
            download_vid_audio(cfg.video_url, URL_DOWNLOADS_DIR / job_id)
        try:
            file_dir = URL_DOWNLOADS_DIR if byte_string else RAW_AUDIO_DIR
            job_source, job_id = cfg.identifier()

            if cfg.url:
                filepath = URL_DOWNLOADS_DIR / job_id
            elif cfg.video_url:
                filepath = URL_DOWNLOADS_DIR / f"{job_id}.mp3"
            else:
                file = Path(cfg.filename)
                filepath = file_dir / file.name

            result = transcribe_x.call(
                file_path=filepath,
                result_path=result_path,
                skip_align=bool(byte_string),
            )
            recording_id = notify.get("metadata") and notify["metadata"].get(
                "recording_id"
            )
            if recording_id:
                put_s3(
                    "talk-clips",
                    f"transcriptions/{recording_id}/aligned.json",
                    json.dumps(result),
                )

            if summarize:
                try:
                    summary = summarize_transcript(result["full_text"])
                except Exception as e:
                    log.info("failed to summarize")
                    log.info(e)
                    summary = ""
                result["summary"] = summary
                if len(result["full_text"]) > 4000:
                    title_from = summary
                    title_type = "conversation summary"
                else:
                    title_from = result["full_text"]
                    title_type = "conversation transcription"
                try:
                    title = make_title(title_from, title_type)
                except Exception as e:
                    log.info("failed to make title")
                    log.info(e)
                    title = ""
                result["title"] = title
            if use_llm:
                res = result["full_text"].strip()
                if res:
                    llm_response = llm_respond.call(res)
                    result["llm_response"] = llm_response
                else:
                    result["llm_response"] = "Sorry, I couldn't understand that."
            if notify:
                notify_webhook(result, notify)
            return result
        except Exception as e:
            log.error(e)
        finally:
            del container_app.running_jobs[job_id]
            if byte_string:
                log.info(f"Cleaning up cache: {URL_DOWNLOADS_DIR / cfg.filename}")
                os.remove(URL_DOWNLOADS_DIR / cfg.filename)
            if cfg.url or cfg.video_url:
                filepath = URL_DOWNLOADS_DIR / (
                    f"{job_id}{'.mp3' if cfg.video_url else ''}"
                )
                log.info(f"Cleaning up cache: {filepath}")
                os.remove(filepath)


def notify_webhook(result, notify):
    meta = notify["metadata"] or {}
    notify_callback(notify["url"], {"data": result, "metadata": meta})


class FanTranscriber:
    @staticmethod
    def run(overrides: dict = None, byte_string: str = None):
        log.info(f"Starting fan-out transcriber with overrides: {overrides}")
        cfg = args.merge(overrides) if overrides else args
        if stub.is_inside():
            return start_transcribe.call(cfg=cfg, byte_string=byte_string)
        else:
            with stub.run():
                return start_transcribe.call(cfg=cfg, byte_string=byte_string)

    @staticmethod
    def queue(url: str, cfg: TranscribeConfig, metadata: dict = None, summarize=False):
        notify = {"url": url, "metadata": metadata or {}}
        if stub.is_inside():
            return start_transcribe.spawn(cfg=cfg, notify=notify, summarize=summarize)
        else:
            with stub.run():
                return start_transcribe.spawn(
                    cfg=cfg, notify=notify, summarize=summarize
                )


entities_image = (
    Image.debian_slim("3.10.0")
    .pip_install("spacy")
    .run_commands(["python -m spacy download en_core_web_md"])
)


@stub.function(image=entities_image)
def get_entity_bounds(text: str) -> list[tuple[int, int, str]]:
    import spacy

    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
