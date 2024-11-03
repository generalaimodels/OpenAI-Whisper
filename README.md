

# Audio and YouTube Transcription CLI App

This CLI application allows users to transcribe audio files and YouTube videos using OpenAI's Whisper model. It can also record audio directly from a microphone for transcription.

## Features

- **Transcribe audio files**: Input an audio file to get a transcription.
- **Transcribe YouTube videos**: Provide a YouTube URL, and the audio will be downloaded and transcribed.
- **Record audio**: Record audio directly from your microphone for live transcription.


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/generalaimodels/OpenAI-Whisper.git
   cd OpenAI-Whisper
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Transcribe an Audio File

To transcribe an audio file:

```bash
python app.py transcribe --file path/to/your-audio-file.wav
```

### Record and Transcribe

To record audio and transcribe it:

```bash
python app.py transcribe --record
```

### Transcribe a YouTube Video

To transcribe audio from a YouTube video:

```bash
python app.py yt-transcribe "https://youtube.com/your-video-url"
```

### Additional Options

- **`--task`**: Specify the task, either `transcribe` (default) or `translate` (transcribes and translates into English).

## CLI Options

```bash
usage: app.py [-h] {transcribe,yt-transcribe} ...
```

- `transcribe`: Transcribe an audio file or record from microphone.
  - `--file`: Path to an audio file.
  - `--record`: Record audio from the microphone.
  - `--task`: Task to perform: `transcribe` or `translate`.

- `yt-transcribe`: Transcribe audio from a YouTube video.
  - `url`: YouTube video URL.
  - `--task`: Task to perform: `transcribe` or `translate`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
Note: Thanks to openai [whisper](https://github.com/openai/whisper) 
```

This README provides clear instructions on installation, usage, and features, helping users quickly set up and use your application.