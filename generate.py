"""
Slide generation and enlargement using Gemini API
"""

import os
import uuid
import httpx
import mimetypes
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")


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
        image_data = None
        mime_type = "image/jpeg"
        
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-preview-image-generation",
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
        client = genai.Client(api_key=api_key)
        
        # Download the original image
        image_bytes, orig_mime_type = await download_image(image_url)
        
        # Upscale prompt
        prompt = "Upscale this image to higher resolution. Maintain all details, text, and structure exactly. Do not add or remove elements. Just increase the resolution and sharpness."
        
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
            model="gemini-2.0-flash-preview-image-generation",
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

