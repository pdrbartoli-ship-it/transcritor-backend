import os
import io
import math
import asyncio
import httpx
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Transcritor Isis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MAX_CHUNK_BYTES = 23 * 1024 * 1024  # 23MB por chunk (margem de segurança)


class TranscriptionResult(BaseModel):
    transcript: str
    summary: str
    chunks_used: int
    duration_estimate: str


def split_audio_ffmpeg(audio_bytes: bytes, filename: str) -> list[bytes]:
    """Divide áudio em chunks usando ffmpeg — corta nos momentos certos."""
    if len(audio_bytes) <= MAX_CHUNK_BYTES:
        return [audio_bytes]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Salva o arquivo original
        input_path = os.path.join(tmpdir, filename)
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Descobre a duração total
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", input_path],
            capture_output=True, text=True
        )

        import json
        info = json.loads(probe.stdout)
        total_seconds = float(info["format"]["duration"])

        # Calcula quantos chunks precisamos
        num_chunks = math.ceil(len(audio_bytes) / MAX_CHUNK_BYTES)
        chunk_duration = total_seconds / num_chunks

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration
            output_path = os.path.join(tmpdir, f"chunk_{i}.m4a")

            subprocess.run([
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-c", "copy",
                output_path
            ], capture_output=True)

            with open(output_path, "rb") as f:
                chunks.append(f.read())

        return chunks


async def transcribe_chunk(client: httpx.AsyncClient, audio_bytes: bytes, chunk_index: int) -> str:
    """Transcreve um chunk usando Groq Whisper."""
    files = {
        "file": (f"chunk_{chunk_index}.m4a", io.BytesIO(audio_bytes), "audio/m4a"),
        "model": (None, "whisper-large-v3-turbo"),
        "response_format": (None, "text"),
        "language": (None, "pt"),
    }

    response = await client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files=files,
        timeout=180.0,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro na transcrição (parte {chunk_index + 1}): {response.text}"
        )

    return response.text.strip()


async def summarize(client: httpx.AsyncClient, transcript: str) -> str:
    """Gera resumo usando Claude Haiku."""
    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": f"""Você recebeu a transcrição de um áudio gravado no iPhone. Crie um resumo estruturado em português com:

**🎯 Tema principal** — uma frase resumindo o assunto

**📌 Pontos principais** — lista dos tópicos mais importantes discutidos

**✅ Conclusões / Ações** — o que foi decidido, combinado ou concluído (se houver)

**💬 Observações** — contexto ou detalhes relevantes (se houver)

Seja direto e use linguagem natural.

Transcrição:
{transcript}"""
            }]
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao gerar resumo: {response.text}"
        )

    return response.json()["content"][0]["text"]


@app.get("/")
async def health():
    return {"status": "ok", "service": "Transcritor Isis"}


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = File(...)):
    if not GROQ_API_KEY or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Chaves de API não configuradas.")

    audio_bytes = await file.read()
    file_size_mb = len(audio_bytes) / 1024 / 1024

    # Estima duração
    estimated_minutes = file_size_mb / 0.75
    if estimated_minutes < 1:
        duration_str = "menos de 1 minuto"
    elif estimated_minutes < 60:
        duration_str = f"~{int(estimated_minutes)} minutos"
    else:
        hours = int(estimated_minutes // 60)
        mins = int(estimated_minutes % 60)
        duration_str = f"~{hours}h{mins:02d}min"

    # Divide em chunks com ffmpeg (corte correto no áudio)
    try:
        chunks = split_audio_ffmpeg(audio_bytes, file.filename or "audio.m4a")
    except Exception as e:
        # Fallback: divide por bytes se ffmpeg falhar
        chunks = []
        for i in range(0, len(audio_bytes), MAX_CHUNK_BYTES):
            chunks.append(audio_bytes[i:i + MAX_CHUNK_BYTES])

    async with httpx.AsyncClient() as client:
        # Transcreve em paralelo
        tasks = [transcribe_chunk(client, chunk, i) for i, chunk in enumerate(chunks)]
        transcripts = await asyncio.gather(*tasks)
        full_transcript = " ".join(transcripts)

        # Gera resumo
        summary = await summarize(client, full_transcript)

    return TranscriptionResult(
        transcript=full_transcript,
        summary=summary,
        chunks_used=len(chunks),
        duration_estimate=duration_str,
    )
