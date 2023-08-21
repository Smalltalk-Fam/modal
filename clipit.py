import dataclasses
import os

import modal
from typing import TypedDict, List, Optional
from modal import Stub, Secret, web_endpoint, Mount, NetworkFileSystem
import time
from pathlib import Path
import hashlib
from fastapi import Header

from network import get_s3_json, save_s3_file, save_file, put_s3, notify_callback
from logger import log


class Word(TypedDict):
    word: str
    start: Optional[float]
    end: Optional[float]
    score: Optional[float]


class Sentence(TypedDict):
    text: str
    words: List[Word]
    start: Optional[float]
    end: Optional[float]


Transcript = List[Sentence]

image = (
    modal.Image.debian_slim("3.10.0")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "Pillow==9.5.0",
        "boto3==1.26.137",
        "ffmpeg-python==0.2.0",
        "moviepy==1.0.3",
        "numpy==1.24.3",
        "opencv-python==4.7.0.72",
        "requests==2.31.0",
        "tqdm==4.65.0",
    )
    .run_commands("git clone https://github.com/google/fonts.git /gfonts")
)
web_img = modal.Image.debian_slim("3.10.0")

stub = Stub("st-clipit", image=image)
sv = NetworkFileSystem.persisted("clipit-tmp")

BUCKET = "talk-clips"
RECORDINGS_BUCKET = "talk-recordings"
dir_path = os.path.dirname(os.path.realpath(__file__))
TMP_AUDIO = Path("/tmp/audio")
TMP_IMG = Path("/tmp/img")
TMP_VID = Path("/tmp/vid")
MNT_STATIC = Path("/tmp/assets")
WIDTH, HEIGHT = 1024, 1024
FRAME_RATE = 24
SANS_FONT_PATH = str(MNT_STATIC / "GT-Flexa-Standard-Medium.ttf")
MONO_FONT_PATH = "/gfonts/apache/robotomono/RobotoMono[wght].ttf"
SERIF_FONT_PATH = "/gfonts/ofl/spectral/Spectral-Medium.ttf"
FONT_SIZE = 48
TEXT_COLOR = (10, 10, 10, 80)
BG_COLOR = (0, 0, 0, 0)
HIGHLIGHT_COLOR = (255, 190, 213, 255)
Y_PAD = 8


def resize(img, target_width: int = WIDTH, target_height: int = HEIGHT):
    """
    Resizes an image to fit within the target dimensions, cropping if necessary
    PIL.Image in, PIL.Image out.
    """
    from PIL import Image

    img_ratio = img.width / img.height
    target_ratio = target_width / target_height
    if img_ratio < target_ratio:
        rescale_size = (target_width, int(target_width / img_ratio))
    else:
        rescale_size = (int(target_height * img_ratio), target_height)
    img = img.resize(rescale_size, Image.ANTIALIAS)
    left_margin = (img.width - target_width) / 2
    top_margin = (img.height - target_height) / 2
    right_margin = left_margin + target_width
    bottom_margin = top_margin + target_height
    img = img.crop((left_margin, top_margin, right_margin, bottom_margin))
    return img


def image_for(sentence: Sentence, word_idx: int | None, img_path: str | None):
    """
    Returns a cv2 image with the sentence text, optionally overlaid on an image
    If a word_idx is provided, words up to that index will be highlighted.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    import numpy as np

    img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    lines = textwrap.wrap(sentence["text"], width=38)
    font = ImageFont.truetype(SANS_FONT_PATH, FONT_SIZE)
    y_pos = HEIGHT // 2 - (len(lines) // 2 * (FONT_SIZE + Y_PAD))

    wi = 0
    for li, line in enumerate(lines):
        words = line.strip().split(" ")
        x_pos = (WIDTH - sum(draw.textlength(f"{w} ", font) for w in words)) // 2
        for w in words:
            if word_idx is not None and wi <= word_idx:
                draw.text((x_pos, y_pos), w, font=font, fill=HIGHLIGHT_COLOR)
            else:
                draw.text((x_pos, y_pos), w, font=font, fill=TEXT_COLOR)
            x_pos += draw.textlength(f"{w} ", font)
            wi += 1
        y_pos += FONT_SIZE + Y_PAD

    # So far we are in RGBA, but when we make the video we need to be in RGB
    # Here, either composite with the provided bg image, or just convert to RGB
    if img_path:
        base_image = resize(Image.open(img_path).convert("RGBA"))
        final = Image.alpha_composite(base_image, img).convert("RGB")
    else:
        final = img.convert("RGB")

    # PIL is in RGB, cv2 is in BGR, convert to a np array in BGR
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)


def find_next_word(sentence: Sentence, cur_word: int | None):
    """
    Given a sentence and a current word index, returns the next word
    that has a time stamp. If no words after cur_word have a time stamp,
    returns the next word index.
    """
    w_count = len(sentence["words"])
    max_word_idx = w_count - 1
    next_word_idx = 0 if cur_word is None else min(cur_word + 1, max_word_idx)

    if not sentence["words"][next_word_idx].get("start"):
        for j in range(w_count - next_word_idx):
            maybe_next = min(j + next_word_idx, max_word_idx)
            if sentence["words"][maybe_next].get("start") is not None:
                return maybe_next
            if maybe_next == max_word_idx:
                break
    return next_word_idx


def cut_video(transcript: Transcript, media_path: str, write_path: str):
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from moviepy.video.VideoClip import ImageClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    t0 = time.time()
    start_time = transcript[0]["start"]
    end_time = transcript[-1]["end"]

    icon_size = 86
    icon_padding = 56
    icon = ImageClip(str(MNT_STATIC / "icon.png"))
    icon = icon.resize(width=icon_size)
    icon = icon.set_opacity(0.85)

    clip = VideoFileClip(media_path)
    clip = clip.subclip(start_time, end_time)

    aspect_ratio = clip.size[0] / clip.size[1]

    if aspect_ratio >= 1:
        clip = clip.resize(width=WIDTH)
    else:
        clip = clip.resize(height=HEIGHT)

    icon = icon.set_duration(clip.duration)
    icon = icon.set_position((WIDTH - icon_size - icon_padding, icon_padding))
    clip = CompositeVideoClip([clip, icon])

    last_frame_time = clip.duration - 0.1
    last_frame_image = clip.get_frame(last_frame_time)
    last_frame = ImageClip(last_frame_image, duration=1)
    last_frame = last_frame.set_audio(None)

    bumper = VideoFileClip(str(MNT_STATIC / "st-bumper.mp4"))
    bumper = bumper.set_audio(None)

    final_clip = concatenate_videoclips([clip, last_frame, bumper])
    final_clip.write_videofile(write_path)

    log.info(f"Cut video in {time.time() - t0:.2f} seconds")


def make_video(transcript: Transcript, video_path: str, base_img: str | None):
    """
    The main function. Will take in a series of sentence/phrases, and go
    frame-by-frame to generate a video with the text, optionally overlaying it
    on an image. Output video will be silent.
    """
    import cv2
    from tqdm import tqdm
    import numpy as np

    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (WIDTH, HEIGHT))

    start_time = transcript[0]["start"]
    end_time = transcript[-1]["end"]
    total_frames = int((end_time - start_time) * FRAME_RATE)

    cur_sentence = 0
    cur_word = None
    image_cache = {}

    for i in tqdm(range(total_frames)):
        current_time = start_time + (i / FRAME_RATE)
        sentence = transcript[cur_sentence]

        if current_time > sentence["end"]:
            cur_sentence += 1
            cur_word = None
            sentence = transcript[cur_sentence]

        next_word_idx = find_next_word(sentence, cur_word)
        next_word = sentence["words"][next_word_idx]
        if next_word.get("start") and current_time > next_word["start"]:
            cur_word = next_word_idx

        image_key = f"{cur_sentence}_{cur_word or 'None'}_ts"
        if image_key not in image_cache:
            image_cache[image_key] = image_for(
                transcript[cur_sentence], cur_word, base_img
            )
        video.write(np.array(image_cache[image_key]))

    video.release()
    log.info(f"generated video in {time.time() - t0:.2f} seconds")
    return video_path


def combine_audio_video(
    audio_path: str | Path, video_path: str, start_time: float, write_path: str
):
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from moviepy.video.VideoClip import ImageClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    t0 = time.time()

    icon_size = 86
    icon_padding = 56
    icon = ImageClip(str(MNT_STATIC / "icon.png"))
    icon = icon.resize(width=icon_size)
    icon = icon.set_opacity(0.85)

    video_clip = VideoFileClip(video_path)
    icon = icon.set_duration(video_clip.duration)
    icon = icon.set_position((WIDTH - icon_size - icon_padding, icon_padding))

    video_clip = CompositeVideoClip([video_clip, icon])

    audio = AudioFileClip(str(audio_path)).subclip(
        start_time, start_time + video_clip.duration
    )
    video_clip = video_clip.set_audio(audio)

    last_frame_time = video_clip.duration - 0.1
    last_frame_image = video_clip.get_frame(last_frame_time)
    last_frame = ImageClip(last_frame_image, duration=1)
    last_frame = last_frame.set_audio(None)

    bumper = VideoFileClip(str(MNT_STATIC / "st-bumper.mp4"))
    bumper = bumper.set_audio(None)
    final_clip = concatenate_videoclips([video_clip, last_frame, bumper])
    final_clip.write_videofile(write_path, audio_codec="aac", audio_bitrate="192k")

    log.info(f"combined audio and video in {time.time() - t0:.2f} seconds")


@stub.function(
    secrets=[
        Secret.from_name("aws-clip-manager"),
        Secret.from_name("api-secret-key"),
    ],
    mounts=[
        Mount.from_local_dir(
            local_path=os.path.join(dir_path, "assets"),
            remote_path=str(MNT_STATIC),
        ),
    ],
)
def gen_clip(
    clip_id: str,
    transcript_key: str,
    start_idx: int,
    end_idx: int,
    audio_key: str = None,
    video_key: str = None,
    start_word_offset: Optional[int] = None,
    end_word_offset: Optional[int] = None,
    base_image_url: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> str:
    t0 = time.time()
    obj = get_s3_json(BUCKET, transcript_key)
    transcript = obj["segments"]
    selection = transcript[start_idx : end_idx + 1]
    if start_word_offset and start_word_offset > 0:
        words = selection[0]["words"][start_word_offset:]
        # find first word with a start and set start
        start = next(w["start"] for w in words if w.get("start"))
        end = selection[0]["end"]
        text = " ".join([w["text"] for w in words])
        selection[0] = {"start": start, "end": end, "text": text, "words": words}
    if end_word_offset and end_word_offset > 0:
        words = selection[-1]["words"][:-end_word_offset]
        # find last word with an end and set end
        end = next(w["end"] for w in reversed(words) if w.get("end"))
        start = selection[-1]["start"]
        text = " ".join([w["text"] for w in words])
        selection[-1] = {"start": start, "end": end, "text": text, "words": words}

    log.info(f"\n\nstart sample: {selection[0]['text']}\n\n")
    media_key = audio_key or video_key
    media_hash = hashlib.md5(
        f"{RECORDINGS_BUCKET}/{media_key}".encode("utf-8")
    ).hexdigest()

    TMP_AUDIO.mkdir(parents=True, exist_ok=True)
    media_path = TMP_AUDIO / media_hash

    if not os.path.exists(media_path):
        save_s3_file(RECORDINGS_BUCKET, media_key, media_path)

    img_path = str(MNT_STATIC / "default_bg.png")
    if base_image_url:
        TMP_IMG.mkdir(parents=True, exist_ok=True)
        img_hash = hashlib.md5(base_image_url.encode("utf-8")).hexdigest()
        pth = TMP_IMG / img_hash
        save_file(base_image_url, pth)
        img_path = str(pth)

    TMP_VID.mkdir(parents=True, exist_ok=True)
    final_path = str(TMP_VID / f"{clip_id}.mp4")

    if audio_key:
        silent_vid_path = make_video(selection, f"/tmp/{clip_id}_silent.mp4", img_path)
        start_time = selection[0]["start"]
        combine_audio_video(media_path, silent_vid_path, start_time, final_path)
    elif video_key:
        cut_video(selection, str(media_path), final_path)

    with open(final_path, "rb") as file:
        data = file.read()
        prefix = os.path.dirname(transcript_key).replace("transcriptions/", "clips/")
        out_file_key = f"{prefix}/{clip_id}.mp4"
        put_s3(BUCKET, out_file_key, data)
    log.info(f"Total time generating gram: {time.time() - t0:.2f} seconds")
    if callback_url:
        body = {"file_key": out_file_key}
        notify_callback(callback_url, body)
    return out_file_key


@dataclasses.dataclass
class APIArgs:
    clip_id: str = None
    transcript_key: str = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    start_word_offset: int | None = None
    end_word_offset: int | None = None

    audio_key: Optional[str] = None
    video_key: Optional[str] = None
    base_image_url: Optional[str] = None
    callback_url: Optional[str] = None
    sync: Optional[bool] = False


@stub.function(secret=Secret.from_name("api-secret-key"), image=web_img)
@web_endpoint(method="POST", label="clipit")
def web(api_args: APIArgs, x_modal_secret: str = Header(default=None)):
    log.info(f"Start clip {api_args.transcript_key}")
    log.info(f"Boundary: {api_args.start_idx} - {api_args.end_idx}")
    secret = os.environ["API_SECRET_KEY"]
    if secret and x_modal_secret != secret:
        return {"error": "Not authorized"}
    has_media = api_args.audio_key or api_args.video_key
    if api_args.clip_id and api_args.transcript_key and has_media:
        if api_args.sync:
            args = dataclasses.asdict(api_args)
            del args["sync"]
            key = gen_clip.call(**args)
            log.info(f"Generated clip: {key}")
            return {"file_key": key}
        else:
            args = dataclasses.asdict(api_args)
            del args["sync"]
            call = gen_clip.spawn(**args)
            return {"call_id": call.object_id}
    return {"error": "Invalid resource"}
