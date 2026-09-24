from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".deps"
sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(1, str(ROOT))

from faster_whisper import WhisperModel  # noqa: E402


def srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: Path, rows: list[dict], bilingual: bool = False) -> None:
    blocks = []
    for index, row in enumerate(rows, 1):
        text = row["zh"] + "\n" + row["en"] if bilingual else row["text"]
        blocks.append(
            f"{index}\n{srt_time(row['start'])} --> {srt_time(row['end'])}\n"
            f"{text.strip()}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def wait_for_server(server_url: str, server: subprocess.Popen, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(f"llama-server exited during startup (exit code {exit_code})")
        try:
            with urllib.request.urlopen(server_url.rstrip("/") + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError("llama-server did not become ready within the timeout")


def translate(server_url: str, text: str) -> str:
    prompt = (
        "Translate the following English subtitle into Simplified Chinese. "
        "Only output the translated subtitle, without explanations, quotation marks, "
        "or language labels. Preserve names, numbers, and technical terms accurately.\n\n"
        + text
    )
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "max_tokens": 256,
        "stream": False,
    }
    request = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Whisper + Hy-MT2 bilingual subtitles")
    parser.add_argument("--input", required=True, help="Input MP4 file")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to input/subtitles")
    parser.add_argument("--whisper-model", default=str(ROOT / "whisper" / "medium"))
    parser.add_argument("--translation-model", default=str(ROOT / "models" / "Hy-MT2-1.8B-GGUF" / "Hy-MT2-1.8B-Q4_K_M.gguf"))
    parser.add_argument("--llama-server", default=str(ROOT / "llama_cpp" / "llama-server.exe"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--threads", type=int, default=2, help="Hy-MT2 translation CPU threads")
    parser.add_argument("--whisper-threads", type=int, default=2, help="Whisper CPU threads")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent / "subtitles"
    whisper_model = Path(args.whisper_model).resolve()
    translation_model = Path(args.translation_model).resolve()
    llama_server = Path(args.llama_server).resolve()
    for path, label in [
        (input_path, "Input video"),
        (whisper_model, "Whisper model"),
        (translation_model, "Translation model"),
        (llama_server, "llama-server"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Whisper has exclusive use of CPU resources.
    print("Transcribing English with Whisper...", flush=True)
    whisper = WhisperModel(
        str(whisper_model),
        device="cpu",
        compute_type="int8",
        cpu_threads=args.whisper_threads,
    )
    segments, info = whisper.transcribe(
        str(input_path),
        language=args.language,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    rows = [
        {"start": float(s.start), "end": float(s.end), "en": s.text.strip()}
        for s in segments if s.text.strip()
    ]
    write_srt(
        output_dir / f"{input_path.stem}.en.srt",
        [{**row, "text": row["en"]} for row in rows],
    )
    print(f"Whisper produced {len(rows)} segments; language={info.language}", flush=True)

    # Phase 2: free Whisper's compute work before loading Hy-MT2 for translation.
    del whisper
    print("Loading Hy-MT2 Q4_K_M...", flush=True)
    server = subprocess.Popen(
        [
            str(llama_server), "--model", str(translation_model),
            "--host", "127.0.0.1", "--port", "8765",
            "--ctx-size", "4096", "--threads", str(args.threads),
            "--n-gpu-layers", "0",
        ],
        cwd=str(llama_server.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    try:
        wait_for_server(args.server_url, server)
        translated = []
        for index, row in enumerate(rows, 1):
            for attempt in range(3):
                try:
                    chinese = translate(args.server_url, row["en"])
                    if chinese:
                        break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            translated.append({"start": row["start"], "end": row["end"], "en": row["en"], "zh": chinese})
            print(f"Translated {index}/{len(rows)}", flush=True)

        base = output_dir / input_path.stem
        write_srt(output_dir / f"{input_path.stem}.zh.srt", [{**row, "text": row["zh"]} for row in translated])
        write_srt(output_dir / f"{input_path.stem}.bilingual.srt", translated, bilingual=True)
        (output_dir / f"{input_path.stem}.segments.json").write_text(
            json.dumps({"input": str(input_path), "source_language": "English", "target_language": "Simplified Chinese", "segments": translated}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Chinese SRT: {base}.zh.srt", flush=True)
        print(f"Bilingual SRT: {base}.bilingual.srt", flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
