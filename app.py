#!/usr/bin/env python3
"""
CLI Application for Transcribing Audio and YouTube Videos using OpenAI Whisper.
"""

import argparse
import os
import sys
import tempfile
import time
from typing import Optional, Tuple

import torch
import yt_dlp as youtube_dl
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


# Constants
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files
AUDIO_RECORD_DURATION = 30  # seconds
AUDIO_SAMPLE_RATE = 16000  # Hz


def get_device() -> int:
    """Determine the appropriate device (CPU or GPU)."""
    return 0 if torch.cuda.is_available() else -1  # -1 for CPU in sounddevice


pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=get_device(),
)


def transcribe_audio(
    audio_path: str, task: str = "transcribe"
) -> str:
    """
    Transcribe or translate an audio file.

    Args:
        audio_path (str): Path to the audio file.
        task (str, optional): Task to perform. Defaults to "transcribe".

    Returns:
        str: Transcribed or translated text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the task is not recognized.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' does not exist.")

    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Invalid task '{task}'. Choose 'transcribe' or 'translate'.")

    try:
        inputs = ffmpeg_read(audio_path, pipe.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

        result = pipe(
            inputs,
            batch_size=BATCH_SIZE,
            generate_kwargs={"task": task},
            return_timestamps=False
        )
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e


def download_youtube_audio(
    yt_url: str, output_dir: str
) -> Tuple[str, str]:
    """
    Download audio from a YouTube video.

    Args:
        yt_url (str): YouTube video URL.
        output_dir (str): Directory to save the downloaded audio.

    Returns:
        Tuple[str, str]: Path to the downloaded audio file and video title.

    Raises:
        youtube_dl.DownloadError: If downloading fails.
        ValueError: If the video length exceeds the limit.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(yt_url, download=True)
            duration = info.get("duration", 0)
            title = info.get("title", "Unknown Title")
        except youtube_dl.DownloadError as e:
            raise RuntimeError(f"Failed to download YouTube video: {e}") from e

    if duration > YT_LENGTH_LIMIT_S:
        limit_hms = time.strftime("%H:%M:%S", time.gmtime(YT_LENGTH_LIMIT_S))
        video_hms = time.strftime("%H:%M:%S", time.gmtime(duration))
        raise ValueError(
            f"Maximum YouTube video length is {limit_hms}, but the video is {video_hms} long."
        )

    audio_extensions = ["wav", "m4a", "mp3"]
    audio_path = None
    for ext in audio_extensions:
        potential_path = os.path.join(output_dir, f"audio.{ext}")
        if os.path.isfile(potential_path):
            audio_path = potential_path
            break

    if not audio_path:
        raise FileNotFoundError("Downloaded audio file not found.")

    return audio_path, title


def record_audio(duration: int = AUDIO_RECORD_DURATION, sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
    """
    Record audio from the microphone.

    Args:
        duration (int, optional): Duration to record in seconds. Defaults to AUDIO_RECORD_DURATION.
        sample_rate (int, optional): Sampling rate. Defaults to AUDIO_SAMPLE_RATE.

    Returns:
        str: Path to the recorded audio file.

    Raises:
        RuntimeError: If audio recording fails.
    """
    print(f"Recording audio for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, sample_rate, recording)
        print(f"Audio recorded and saved to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        raise RuntimeError(f"Audio recording failed: {e}") from e


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files or YouTube videos using OpenAI Whisper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for audio transcription
    audio_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file or record from microphone."
    )
    audio_group = audio_parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--file",
        type=str,
        help="Path to the audio file to transcribe."
    )
    audio_group.add_argument(
        "--record",
        action="store_true",
        help="Record audio from the microphone."
    )
    audio_parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: transcribe or translate the audio."
    )

    # Subparser for YouTube transcription
    yt_parser = subparsers.add_parser(
        "yt-transcribe",
        help="Transcribe audio from a YouTube video."
    )
    yt_parser.add_argument(
        "url",
        type=str,
        help="YouTube video URL."
    )
    yt_parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: transcribe or translate the audio."
    )

    return parser.parse_args()


def main() -> None:
    """Main function to execute the CLI application."""
    args = parse_arguments()

    if args.command == "transcribe":
        if args.file:
            audio_path = args.file
            print(f"Transcribing audio file: {audio_path}")
        elif args.record:
            try:
                audio_path = record_audio()
            except RuntimeError as e:
                print(e, file=sys.stderr)
                sys.exit(1)
        else:
            print("Either --file or --record must be provided.", file=sys.stderr)
            sys.exit(1)

        try:
            transcription = transcribe_audio(audio_path, task=args.task)
            print("\n--- Transcription ---")
            print(transcription)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        finally:
            if args.record:
                os.unlink(audio_path)  # Clean up the temporary recorded file

    elif args.command == "yt-transcribe":
        yt_url = args.url
        print(f"Downloading and transcribing YouTube video: {yt_url}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                audio_path, title = download_youtube_audio(yt_url, tmpdirname)
                print(f"Downloaded audio: {audio_path} (Title: {title})")
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                print(e, file=sys.stderr)
                sys.exit(1)

            try:
                transcription = transcribe_audio(audio_path, task=args.task)
                print(f"\n--- Transcription for '{title}' ---")
                print(transcription)
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(e, file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()