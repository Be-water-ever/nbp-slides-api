# NBP Slides API

Python backend for NBP Slides - handles image generation and enlargement using Gemini API.

## Deployment on Railway

### 1. Create a new project on Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect this repository

### 2. Configure Environment Variables

In Railway dashboard, add these environment variables:

```
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key_id
R2_SECRET_ACCESS_KEY=your_secret_access_key
R2_BUCKET_NAME=nbp-slides-assets
R2_PUBLIC_URL=https://your-bucket.r2.dev

FRONTEND_URL=https://your-vercel-app.vercel.app
```

### 3. Deploy

Railway will automatically deploy when you push to GitHub.

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET_NAME=...

# Run server
uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Health check
- `POST /generate` - Generate a slide image
- `POST /enlarge` - Enlarge/upscale an image

## API Request Examples

### Generate Slide

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your_gemini_api_key",
    "prompt": "A beautiful presentation slide about AI",
    "aspect_ratio": "16:9"
  }'
```

### Enlarge Image

```bash
curl -X POST http://localhost:8000/enlarge \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your_gemini_api_key",
    "image_url": "https://example.com/image.jpg"
  }'
```

