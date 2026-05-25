import os
import io
import math
import asyncio
import httpx
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

MAX_CHUNK_BYTES = 24 * 1024 * 1024  # 24MB por chunk


class TranscriptionResult(BaseModel):
    transcript: str
    summary: str
    chunks_used: int
    duration_estimate: str


async def transcribe_chunk(client: httpx.AsyncClient, audio_bytes: bytes, filename: str, chunk_index: int) -> str:
    """Transcreve um chunk de áudio usando Groq Whisper."""
    files = {
        "file": (filename, io.BytesIO(audio_bytes), "audio/m4a"),
        "model": (None, "whisper-large-v3-turbo"),
        "response_format": (None, "text"),
        "language": (None, "pt"),
    }

    response = await client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files=files,
        timeout=120.0,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro na transcrição (chunk {chunk_index}): {response.text}"
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
            "messages": [
                {
                    "role": "user",
                    "content": f"""Você recebeu a transcrição de um áudio gravado no iPhone. Crie um resumo estruturado em português com:

**🎯 Tema principal** — uma frase resumindo o assunto

**📌 Pontos principais** — lista dos tópicos mais importantes discutidos

**✅ Conclusões / Ações** — o que foi decidido, combinado ou concluído (se houver)

**💬 Observações** — contexto ou detalhes relevantes (se houver)

Seja direto e use linguagem natural. Se for uma nota pessoal, reunião, aula ou outro tipo, adapte o resumo ao contexto.

Transcrição:
{transcript}""",
                }
            ],
        },
        timeout=60.0,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao gerar resumo: {response.text}"
        )

    data = response.json()
    return data["content"][0]["text"]


def split_audio_bytes(audio_bytes: bytes, max_chunk_size: int) -> list[bytes]:
    """Divide o áudio em chunks de tamanho máximo."""
    if len(audio_bytes) <= max_chunk_size:
        return [audio_bytes]

    num_chunks = math.ceil(len(audio_bytes) / max_chunk_size)
    chunk_size = len(audio_bytes) // num_chunks

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(audio_bytes)
        chunks.append(audio_bytes[start:end])

    return chunks


@app.get("/")
async def health():
    return {"status": "ok", "service": "Transcritor Isis"}


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = File(...)):
    if not GROQ_API_KEY or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Chaves de API não configuradas no servidor.")

    audio_bytes = await file.read()
    file_size_mb = len(audio_bytes) / 1024 / 1024

    # Estima duração (M4A ~0.5-1MB/min)
    estimated_minutes = file_size_mb / 0.75
    if estimated_minutes < 1:
        duration_str = "menos de 1 minuto"
    elif estimated_minutes < 60:
        duration_str = f"~{int(estimated_minutes)} minutos"
    else:
        hours = int(estimated_minutes // 60)
        mins = int(estimated_minutes % 60)
        duration_str = f"~{hours}h{mins:02d}min"

    # Divide em chunks se necessário
    chunks = split_audio_bytes(audio_bytes, MAX_CHUNK_BYTES)
    filename = file.filename or "audio.m4a"

    async with httpx.AsyncClient() as client:
        # Transcreve todos os chunks em paralelo
        tasks = [
            transcribe_chunk(client, chunk, filename, i)
            for i, chunk in enumerate(chunks)
        ]
        transcripts = await asyncio.gather(*tasks)

        # Junta as transcrições
        full_transcript = " ".join(transcripts)

        # Gera resumo
        summary = await summarize(client, full_transcript)

    return TranscriptionResult(
        transcript=full_transcript,
        summary=summary,
        chunks_used=len(chunks),
        duration_estimate=duration_str,
    )
