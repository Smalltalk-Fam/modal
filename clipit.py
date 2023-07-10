import os
import modal
from modal import Stub, SharedVolume
import time
from pathlib import Path
import hashlib

from from_url import cache_file
from logger import log

from typing import TypedDict, List, Optional


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
        "boto3==1.26.137",
        "Pillow==9.5.0",
        "tqdm==4.65.0",
        "numpy==1.24.3",
        "ffmpeg-python==0.2.0",
        "opencv-python==4.7.0.72",
        "moviepy==1.0.3",
    )
    .run_commands("git clone https://github.com/google/fonts.git /gfonts")
)

stub = Stub("ss-audiogram", image=image)
sv = SharedVolume().persist("audiogram-tmp")

TMP_FILES = Path("/tmp/audio")
WIDTH, HEIGHT = 1024, 1024
FRAME_RATE = 24
SANS_FONT_PATH = "/gfonts/apache/roboto/static/Roboto-Medium.ttf"
FONT_PATH = "/gfonts/ofl/merriweather/Merriweather-Regular.ttf"
FONT_SIZE = 48
TEXT_COLOR = (255, 255, 255, 225)
BG_COLOR = (0, 0, 0, 180)
HIGHLIGHT_COLOR = (240, 215, 10, 255)
Y_PAD = 8

# Cache of cv2 images. Most of the time, we'll be re-using the same frame
# and there generally won't be that many distinct frames in a video.
image_cache = {}


def resize(img, target_width: int = WIDTH, target_height: int = HEIGHT):
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
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    import numpy as np

    img = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    lines = textwrap.wrap(sentence["text"], width=38)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
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


def make_video(transcript: Transcript, video_path: str, base_img: str | None):
    """
    The main function. Will take in a series of sentence/phrases, and go frame-by-frame
    to generate a video with the text, optionally overlaying it on an image. Output video
    will be silent.
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
    """
    Given the audio file that the transcript came from, combine that subclip
    with the generated audiogram.
    """
    from moviepy.editor import VideoFileClip, AudioFileClip

    t0 = time.time()
    video_clip = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path).subclip(
        start_time, start_time + video_clip.duration
    )
    video_clip = video_clip.set_audio(audio)
    video_clip.write_videofile(write_path)
    log.info(f"combined audio and video in {time.time() - t0:.2f} seconds")


@stub.function()
def audiogram(
    transcript: Transcript,
    audio_url: str,
    write_url: str,
    start_idx: int,
    end_idx: int,
    base_image: str | None = None,
):
    t0 = time.time()
    selection = transcript[start_idx:end_idx]
    audio_hash = hashlib.md5(f"{audio_url}".encode("utf-8")).hexdigest()

    TMP_FILES.mkdir(parents=True, exist_ok=True)
    audio_path = TMP_FILES / audio_hash
    if not os.path.exists(audio_path):
        cache_file(audio_url, audio_path)

    gram_hash = hashlib.md5(
        f"{audio_url}[{start_idx}:{end_idx}]".encode("utf-8")
    ).hexdigest()

    vid_path = make_video(selection, f"/tmp/{gram_hash}_silent.mp4", base_image)
    start_time = selection[0]["start"]
    final_path = f"/tmp/{gram_hash}.mp4"
    combine_audio_video(audio_path, vid_path, start_time, final_path)

    # save_s3_file(bucket, gram_key, final_path)
    log.info(f"Total time generating gram: {time.time() - t0:.2f} seconds")
