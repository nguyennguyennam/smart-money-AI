# smart-money-AI
This repository implements the first stage of AI process: extracting both image and sound input to raw text file.

## Running

### 1) Install dependencies

Create and activate a virtualenv, then install from the pinned list:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r app\libs.txt
```

### 2) Configure environment

Create a `.env` file in the repo root.

- Start from [.env.example](.env.example)
- Set at least:
	- `REDIS_STREAM_URL` (jobs stream Redis)
	- `REDIS_RESULT_URL` (result Redis)

If you are using Redis Cloud, you usually need TLS:

```text
rediss://default:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT/0
```

### 3) Run the API server (optional)

```bash
uvicorn app.main:app --reload
```

### 4) Run the Redis stream worker

The worker consumes 1 stream entry at a time from `REDIS_STREAM_KEY` (default `stream:jobs:0`), performs OCR/ASR based on `duty`, classifies extracted text, and stores results to the result Redis as `job:{jobId}` JSON with a TTL.

```bash
python -m app.services.worker
```

## Quick smoke test

Add a job to the stream (fields must include `jobId`, `userId`, `fileUrl`, `duty`, `createdAt`):

```bash
redis-cli -u "$REDIS_STREAM_URL" XADD stream:jobs:0 \* jobId 1 userId u1 fileUrl "https://..." duty ocr createdAt "2026-03-30T00:00:00Z"
```

Then check the result key in result Redis:

```bash
redis-cli -u "$REDIS_RESULT_URL" GET job:1
```
