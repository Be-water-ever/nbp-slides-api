"""
Slide generation and enlargement using Gemini API
"""

import os
import uuid
import httpx
import mimetypes
import json
import re
import base64
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")

# YunWu (proxy) configuration
YUNWU_BASE_URL = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1beta").rstrip("/")
YUNWU_OCR_MODEL = os.environ.get("YUNWU_OCR_MODEL", "gemini-2.0-flash")
YUNWU_IMAGE_MODEL = os.environ.get("YUNWU_IMAGE_MODEL", "gemini-3-pro-image-preview")


def _detect_provider(api_key: str) -> str:
    if api_key.startswith("AIza"):
        return "google"
    if api_key.startswith("sk-"):
        return "yunwu"
    return "unknown"


def _yunwu_endpoint(model: str, api_key: str) -> str:
    model_path = model if model.startswith("models/") else f"models/{model}"
    return f"{YUNWU_BASE_URL}/{model_path}:generateContent?key={api_key}"


def _b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_b64(b64_data: str) -> bytes:
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    missing = len(b64_data) % 4
    if missing:
        b64_data += "=" * (4 - missing)
    return base64.b64decode(b64_data)


def _yunwu_extract_text(resp_json: Dict[str, Any]) -> str:
    texts: List[str] = []
    for cand in resp_json.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _yunwu_extract_first_image(resp_json: Dict[str, Any]) -> tuple[bytes, str]:
    for cand in resp_json.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            inline_data = part.get("inline_data") or part.get("inlineData")
            if not inline_data:
                continue
            b64 = inline_data.get("data")
            if not b64:
                continue
            mime_type = inline_data.get("mime_type") or inline_data.get("mimeType") or "image/jpeg"
            return _decode_b64(b64), mime_type
    raise ValueError("No image data in response")


async def _yunwu_generate_content(
    *,
    api_key: str,
    model: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    url = _yunwu_endpoint(model, api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": generation_config,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 300:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"YunWu API error ({resp.status_code}): {detail}")
        return resp.json()


async def download_image(url: str) -> tuple[bytes, str]:
    """Download an image from URL and return bytes + mime type"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "image/jpeg")
        return response.content, content_type


async def upload_to_r2(image_data: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """Upload image to Cloudflare R2 and return public URL"""
    import boto3
    from botocore.config import Config
    
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        raise ValueError("R2 credentials not configured")
    
    # Create S3 client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    
    # Upload to R2
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=filename,
        Body=image_data,
        ContentType=content_type,
    )
    
    # Return public URL
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/{filename}"
    else:
        # Construct default R2 public URL
        return f"https://{R2_BUCKET_NAME}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{filename}"


async def generate_slide(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    asset_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a slide image using Gemini API.
    
    Args:
        api_key: User's Gemini API key
        prompt: The prompt for image generation
        aspect_ratio: Aspect ratio (default 16:9)
        asset_urls: Optional list of asset image URLs to include
        
    Returns:
        Dict with success, image_url or error
    """
    try:
        provider = _detect_provider(api_key)
        image_data = None
        mime_type = "image/jpeg"

        if provider == "yunwu":
            parts: List[Dict[str, Any]] = [{"text": prompt}]
            if asset_urls:
                for url in asset_urls:
                    try:
                        image_bytes, asset_mime = await download_image(url)
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": asset_mime,
                                    "data": _b64encode_bytes(image_bytes),
                                }
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Failed to download asset {url}: {e}")

            resp_json = await _yunwu_generate_content(
                api_key=api_key,
                model=YUNWU_IMAGE_MODEL,
                parts=parts,
                generation_config={
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": aspect_ratio},
                },
                timeout_s=120.0,
            )
            image_data, mime_type = _yunwu_extract_first_image(resp_json)
        else:
            client = genai.Client(api_key=api_key)
            
            # Build content parts
            parts = [types.Part.from_text(text=prompt)]
            
            # Download and add asset images if provided
            if asset_urls:
                for url in asset_urls:
                    try:
                        image_bytes, mime_type = await download_image(url)
                        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                    except Exception as e:
                        print(f"Warning: Failed to download asset {url}: {e}")
            
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                )
            ]
            
            # Configure image generation
            image_config_dict = {"aspect_ratio": aspect_ratio}
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(**image_config_dict),
            )
            
            # Generate image
            for chunk in client.models.generate_content_stream(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type or "image/jpeg"
                        break
                
                if image_data:
                    break
            
            if not image_data:
                return {"success": False, "error": "No image generated"}
        
        # Upload to R2
        ext = mimetypes.guess_extension(mime_type) or ".jpg"
        filename = f"slides/{uuid.uuid4()}{ext}"
        image_url = await upload_to_r2(image_data, filename, mime_type)
        
        return {"success": True, "image_url": image_url}
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "API quota exceeded. Please wait and try again."
        elif "401" in error_msg or "UNAUTHENTICATED" in error_msg:
            error_msg = "Invalid API key."
        
        return {"success": False, "error": error_msg}


async def extract_text_from_slide(
    api_key: str,
    image_url: str,
) -> Dict[str, Any]:
    """
    Extract text and positions from a slide image using Gemini Vision.
    
    Args:
        api_key: User's Gemini API key
        image_url: URL of the slide image
        
    Returns:
        Dict with success, text_blocks (list of {content, x_percent, y_percent, size, color}) or error
    """
    try:
        # Download the image
        image_bytes, mime_type = await download_image(image_url)
        provider = _detect_provider(api_key)
        
        # OCR prompt - ask Gemini to extract text with positioning
        prompt = """分析这张幻灯片图片，识别所有可见的文字及其大致位置。

请返回严格的 JSON 格式（不要包含其他文字）：
{
  "text_blocks": [
    {
      "content": "文字内容",
      "x_percent": 50,
      "y_percent": 30,
      "width_percent": 80,
      "size": "large",
      "align": "center",
      "color": "#333333"
    }
  ]
}

说明：
- x_percent, y_percent: 文字中心点相对于图片宽高的百分比 (0-100)
- width_percent: 文字块宽度占图片宽度的百分比
- size: "large" (标题), "medium" (副标题), "small" (正文), "tiny" (注释)
- align: "left", "center", "right"
- color: 文字颜色的近似 hex 值

如果没有检测到文字，返回: {"text_blocks": []}"""

        if provider == "yunwu":
            resp_json = await _yunwu_generate_content(
                api_key=api_key,
                model=YUNWU_OCR_MODEL,
                parts=[
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": _b64encode_bytes(image_bytes),
                        }
                    },
                    {"text": prompt},
                ],
                generation_config={
                    "responseModalities": ["TEXT"],
                },
                timeout_s=120.0,
            )
            response_text = _yunwu_extract_text(resp_json)
        else:
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ]
            
            # Use standard text generation (not image generation)
            response = client.models.generate_content(
                model="gemini-2.0-flash",  # Use flash for faster OCR
                contents=contents,
            )
            
            # Parse the response
            response_text = response.text.strip()
        
        # Try to extract JSON from the response
        # Find JSON in the response (might be wrapped in markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            text_blocks = result.get("text_blocks", [])
            return {"success": True, "text_blocks": text_blocks}
        else:
            return {"success": True, "text_blocks": []}
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}, response: {response_text[:500]}")
        return {"success": True, "text_blocks": []}
    except Exception as e:
        error_msg = str(e)
        print(f"OCR error: {error_msg}")
        return {"success": False, "error": error_msg}


async def remove_text_from_slide(
    api_key: str,
    image_url: str,
) -> Dict[str, Any]:
    """
    Remove text from a slide image using Gemini, returning a clean background
    and extracted text positions for HTML overlay.
    
    Args:
        api_key: User's Gemini API key
        image_url: URL of the slide image
        
    Returns:
        Dict with success, clean_image_url, text_blocks or error
    """
    try:
        # Step 1: Extract text positions using OCR
        ocr_result = await extract_text_from_slide(api_key, image_url)
        if not ocr_result["success"]:
            return {"success": False, "error": f"OCR failed: {ocr_result.get('error', 'Unknown error')}"}
        
        text_blocks = ocr_result.get("text_blocks", [])
        
        # If no text found, return original image
        if not text_blocks:
            return {
                "success": True,
                "clean_image_url": image_url,
                "text_blocks": [],
            }
        
        # Step 2: Download the original image
        image_bytes, mime_type = await download_image(image_url)
        provider = _detect_provider(api_key)
        
        prompt = """Remove ALL text from this slide image.
Keep all visual elements, backgrounds, graphics, and design elements intact.
Fill the areas where text was with the surrounding background naturally (content-aware fill).
The result should look like the original slide but with no readable text.
DO NOT add any new elements. Just remove the text cleanly."""

        if provider == "yunwu":
            resp_json = await _yunwu_generate_content(
                api_key=api_key,
                model=YUNWU_IMAGE_MODEL,
                parts=[
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": _b64encode_bytes(image_bytes),
                        }
                    },
                    {"text": prompt},
                ],
                generation_config={
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": "16:9"},
                },
                timeout_s=180.0,
            )
            clean_image_bytes, output_mime_type = _yunwu_extract_first_image(resp_json)
            clean_image_data = clean_image_bytes
        else:
            # Step 3: Use Gemini to remove text from the image
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ]
            
            # Configure for image generation
            image_config_dict = {"aspect_ratio": "16:9"}
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_config_dict),
            )
            
            # Generate clean image
            clean_image_data = None
            output_mime_type = "image/jpeg"
            
            for chunk in client.models.generate_content_stream(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        clean_image_data = part.inline_data.data
                        output_mime_type = part.inline_data.mime_type or "image/jpeg"
                        break
                
                if clean_image_data:
                    break
        
        if not clean_image_data:
            return {"success": False, "error": "Failed to generate clean image"}
        
        # Step 4: Upload clean image to R2
        ext = mimetypes.guess_extension(output_mime_type) or ".jpg"
        filename = f"slides/clean_{uuid.uuid4()}{ext}"
        clean_image_url = await upload_to_r2(clean_image_data, filename, output_mime_type)
        
        return {
            "success": True,
            "clean_image_url": clean_image_url,
            "text_blocks": text_blocks,
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "API quota exceeded. Please wait and try again."
        
        return {"success": False, "error": error_msg}


async def enlarge_slide(
    api_key: str,
    image_url: str,
) -> Dict[str, Any]:
    """
    Enlarge/upscale an image using Gemini API.
    
    Args:
        api_key: User's Gemini API key
        image_url: URL of the image to enlarge
        
    Returns:
        Dict with success, image_url or error
    """
    try:
        # Download the original image
        image_bytes, orig_mime_type = await download_image(image_url)
        provider = _detect_provider(api_key)
        
        # Upscale prompt
        prompt = "Upscale this image to higher resolution. Maintain all details, text, and structure exactly. Do not add or remove elements. Just increase the resolution and sharpness."
        
        if provider == "yunwu":
            resp_json = await _yunwu_generate_content(
                api_key=api_key,
                model=YUNWU_IMAGE_MODEL,
                parts=[
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": orig_mime_type,
                            "data": _b64encode_bytes(image_bytes),
                        }
                    },
                ],
                generation_config={
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"aspectRatio": "16:9"},
                },
                timeout_s=180.0,
            )
            image_data, mime_type = _yunwu_extract_first_image(resp_json)
        else:
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_bytes, mime_type=orig_mime_type)
                    ],
                )
            ]
            
            # Configure for upscaling
            image_config_dict = {"aspect_ratio": "16:9"}
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_config_dict),
            )
            
            # Generate upscaled image
            image_data = None
            mime_type = "image/jpeg"
            
            for chunk in client.models.generate_content_stream(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        image_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type or "image/jpeg"
                        break
                
                if image_data:
                    break
        
            if not image_data:
                return {"success": False, "error": "No image generated"}
        
        # Upload to R2
        ext = mimetypes.guess_extension(mime_type) or ".jpg"
        filename = f"slides/enlarged_{uuid.uuid4()}{ext}"
        new_image_url = await upload_to_r2(image_data, filename, mime_type)
        
        return {"success": True, "image_url": new_image_url}
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "API quota exceeded. Please wait and try again."
        
        return {"success": False, "error": error_msg}
