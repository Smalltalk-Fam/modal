import os
import hashlib
import hmac
import time
from typing import NamedTuple
from pathlib import Path
import urllib.request
import json

from logger import log


class DownloadResult(NamedTuple):
    data: bytes
    content_type: str


FAKE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"


def download_file(url: str) -> DownloadResult:
    req = urllib.request.Request(url, data=None, headers={"User-Agent": FAKE_UA})
    with urllib.request.urlopen(req) as response:
        return DownloadResult(
            data=response.read(), content_type=response.headers["content-type"]
        )


def pretty_size(num: float | int, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def save_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists():
        if overwrite:
            log.info(f"Overwriting file at {destination}")
        else:
            log.info(f"Using cached file at {destination}")
            return

    result = download_file(url=url)
    size = pretty_size(num=len(result.data))
    log.info(f"Downloaded {size} from {url}")
    with open(destination, "wb") as f:
        f.write(result.data)
    log.info(f"Stored file at {destination}")


def save_s3_file(bucket: str, key: str, write_path: str | Path) -> None:
    import boto3

    t0 = time.time()
    s3 = boto3.client("s3")
    file = s3.get_object(Bucket=bucket, Key=key)
    with open(write_path, "wb") as f:
        f.write(file["Body"].read())
    size = pretty_size(file["ContentLength"])
    log.info(f"Downloaded media ({size}) in {time.time() - t0:.2f}s")


def put_s3(bucket: str, key: str, body: bytes | str) -> None:
    import boto3

    t0 = time.time()
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    size = pretty_size(len(body))
    log.info(f"Uploaded {size} to s3://{bucket}/{key} in {time.time() - t0:.2f}s")


def get_s3_json(bucket: str, key: str) -> dict | None:
    import boto3

    s3 = boto3.client("s3")
    try:
        data = s3.get_object(Bucket=bucket, Key=key)
        content = data["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        log.error(f"Error downloading json from S3. Error: {str(e)}")
        return None


def download_vid_audio(
    url: str,
    destination_path: Path,
) -> None:
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "outtmpl": f"{destination_path}.%(ext)s",
    }
    log.info(f"Download video from {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
    log.info(f"Saved audio from {url} to {destination_path}")


def sign_webhook(secret_key: str, message: dict) -> str:
    """
    We sign the post body in python with the secret key so that we can trust that we sent the message
    back to the webhook. In order to verify the signature, we need to replicate the process in javascript
    with the same secret. NB since we are signing the whole body object, we need a standard stringification
    method. Here we choose the most compact version, without key or item separators.

    See models/Transcription.ts for the other half
    """
    return hmac.new(
        secret_key.encode(),
        json.dumps(message, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()


DEV_CALLBACK = "http://"


def notify_callback(callback_url: str, body: dict) -> None:
    import requests

    if not callback_url.startswith(DEV_CALLBACK):
        try:
            signature = sign_webhook(os.environ["API_SECRET_KEY"], body)
            res = requests.post(
                callback_url,
                headers={"x-transcribe-sig": signature},
                json=body,
                verify=False,
            )
            if res.status_code == 200:
                log.info(f"Notified webhook @ {callback_url}")
            else:
                log.error(
                    f"Failed to notify webhook @ {callback_url}: {res.status_code} {res.text}"
                )
        except Exception as e:
            log.exception(f"Failed to notify webhook @ {callback_url}: {e}")
    else:
        log.info(f"Skipping dev webhook @ {callback_url}")
