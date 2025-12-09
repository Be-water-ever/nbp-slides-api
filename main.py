"""
NBP Slides API - FastAPI Backend for Slide Generation
Deployed on Railway, called by Vercel frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

from generate import generate_slide, enlarge_slide

app = FastAPI(
    title="NBP Slides API",
    description="AI-powered slide generation with Gemini",
    version="1.0.0"
)

# CORS configuration - allow all origins for public API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    api_key: str
    prompt: str
    aspect_ratio: str = "16:9"
    asset_urls: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


class EnlargeRequest(BaseModel):
    api_key: str
    image_url: str


class EnlargeResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"status": "ok", "message": "NBP Slides API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """
    Generate a slide image using Gemini API.
    The user's API key is used directly, not stored.
    """
    try:
        result = await generate_slide(
            api_key=request.api_key,
            prompt=request.prompt,
            aspect_ratio=request.aspect_ratio,
            asset_urls=request.asset_urls,
        )
        
        if result["success"]:
            return GenerateResponse(
                success=True,
                image_url=result["image_url"]
            )
        else:
            return GenerateResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        return GenerateResponse(
            success=False,
            error=str(e)
        )


@app.post("/enlarge", response_model=EnlargeResponse)
async def enlarge_endpoint(request: EnlargeRequest):
    """
    Enlarge/upscale an image using Gemini API.
    """
    try:
        result = await enlarge_slide(
            api_key=request.api_key,
            image_url=request.image_url,
        )
        
        if result["success"]:
            return EnlargeResponse(
                success=True,
                image_url=result["image_url"]
            )
        else:
            return EnlargeResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        return EnlargeResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

