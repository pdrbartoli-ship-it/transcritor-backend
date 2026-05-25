# Transcritor Isis — Backend

Backend para transcrição e resumo de áudios via Groq Whisper + Claude.

## Deploy no Render

1. Faça fork ou upload deste repositório no GitHub
2. Acesse [render.com](https://render.com) e crie uma conta
3. Clique em **New → Web Service**
4. Conecte o repositório GitHub
5. Configure as variáveis de ambiente:
   - `GROQ_API_KEY` → sua chave da Groq
   - `ANTHROPIC_API_KEY` → sua chave da Anthropic
6. Clique em **Deploy**

## Endpoint

`POST /transcribe`
- Body: `multipart/form-data` com campo `file` (áudio)
- Retorna: `{ transcript, summary, chunks_used, duration_estimate }`
