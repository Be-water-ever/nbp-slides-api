"""
NBP Slides API - FastAPI Backend for Slide Generation
Deployed on Railway, called by Vercel frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
import os

from generate import generate_slide, enlarge_slide, extract_text_from_slide, remove_text_from_slide

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
    image_size: Literal["1K", "2K", "4K"] = "1K"
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


class TextBlock(BaseModel):
    content: str
    x_percent: float
    y_percent: float
    width_percent: float = 80
    size: str = "medium"
    align: str = "center"
    color: str = "#333333"


class OCRRequest(BaseModel):
    api_key: str
    image_url: str


class OCRResponse(BaseModel):
    success: bool
    text_blocks: Optional[List[TextBlock]] = None
    error: Optional[str] = None


class RemoveTextRequest(BaseModel):
    api_key: str
    image_url: str


class RemoveTextResponse(BaseModel):
    success: bool
    clean_image_url: Optional[str] = None
    text_blocks: Optional[List[TextBlock]] = None
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
            image_size=request.image_size,
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


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(request: OCRRequest):
    """
    Extract text and positions from a slide image using Gemini Vision.
    """
    try:
        result = await extract_text_from_slide(
            api_key=request.api_key,
            image_url=request.image_url,
        )
        
        if result["success"]:
            # Convert to TextBlock models
            text_blocks = [
                TextBlock(**block) for block in result.get("text_blocks", [])
            ]
            return OCRResponse(
                success=True,
                text_blocks=text_blocks
            )
        else:
            return OCRResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        return OCRResponse(
            success=False,
            error=str(e)
        )


@app.post("/remove-text", response_model=RemoveTextResponse)
async def remove_text_endpoint(request: RemoveTextRequest):
    """
    Remove text from a slide image and return clean background + text positions.
    This allows rendering editable HTML text over a clean background.
    """
    try:
        result = await remove_text_from_slide(
            api_key=request.api_key,
            image_url=request.image_url,
        )
        
        if result["success"]:
            # Convert to TextBlock models
            text_blocks = [
                TextBlock(**block) for block in result.get("text_blocks", [])
            ]
            return RemoveTextResponse(
                success=True,
                clean_image_url=result["clean_image_url"],
                text_blocks=text_blocks
            )
        else:
            return RemoveTextResponse(
                success=False,
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        return RemoveTextResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
