#!/usr/bin/env python3
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Optional
from http import HTTPStatus
import base64
import io
import torch
import click
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline

MODEL_NAME = "Qwen/Qwen-Image"
DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

def scan_for_max_image_number(directory: str = "images") -> int:
    """Scan directory for existing images and return the highest number found"""
    if not os.path.exists(directory):
        return 0
    
    existing_files = [f for f in os.listdir(directory) if f.endswith(".png")]
    max_num = 0
    for filename in existing_files:
        match = re.match(r'img(\d+)\.png', filename)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load pipeline and initialize file counter
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"
    
    print(f"Loading model on {device}...")
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        cache_dir="./model_cache"
    )
    pipeline = pipeline.to(device)
    print("Model loaded successfully and staying in memory!")
    
    # Initialize file counter
    os.makedirs("images", exist_ok=True)
    max_num = scan_for_max_image_number("images")
    
    # Store in app state
    app.state.pipeline = pipeline
    app.state.device = device
    app.state.next_image_num = max_num + 1
    
    print(f"Next image will be: img{app.state.next_image_num:04d}.png")
    
    yield
    
    # Shutdown: cleanup if needed
    app.state.pipeline = None

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = " "
    aspect_ratio: str = "16:9"
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int = 42
    filename: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    ready: bool
    model_loaded: bool
    next_image_num: int

class GenerationResponse(BaseModel):
    success: bool
    message: str
    filename: str
    image_data: Optional[str] = None

# FastAPI app with lifespan
app = FastAPI(title="Qwen Image Generator", version="0.1.0", lifespan=lifespan)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status"""
    return StatusResponse(
        status="running",
        ready=hasattr(app.state, 'pipeline') and app.state.pipeline is not None,
        model_loaded=hasattr(app.state, 'pipeline') and app.state.pipeline is not None,
        next_image_num=getattr(app.state, 'next_image_num', 0)
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate image from prompt"""
    if not hasattr(app.state, 'pipeline') or app.state.pipeline is None:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    try:
        # Determine filename and increment counter immediately
        if request.filename:
            filename = request.filename
        else:
            current_num = app.state.next_image_num
            app.state.next_image_num += 1  # Increment immediately to avoid race conditions
            filename = f"images/img{current_num:04d}.png"
        
        # Aspect ratio mapping
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472)
        }
        
        if request.aspect_ratio not in aspect_ratios:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, 
                              detail=f"Unsupported aspect ratio: {request.aspect_ratio}")
        
        width, height = aspect_ratios[request.aspect_ratio]
        
        positive_magic = "Ultra HD, 4K, cinematic composition."
        full_prompt = f"{request.prompt} {positive_magic}"
        
        generator = torch.Generator(device=app.state.device).manual_seed(request.seed)
        
        image = app.state.pipeline(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.true_cfg_scale,
            generator=generator
        ).images[0]
        
        # Convert image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        return GenerationResponse(
            success=True,
            message="Image generated successfully",
            filename=filename,
            image_data=img_base64
        )
        
    except Exception as e:
        return GenerationResponse(
            success=False,
            message=f"Generation failed: {str(e)}",
            filename=""
        )

@click.group()
def cli():
    """Qwen Image Generator CLI"""
    pass

@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Host to bind server to')
@click.option('--port', default=DEFAULT_PORT, help='Port to bind server to')
def server(host: str, port: int):
    """Run the image generation server"""
    print(f"Starting server on {host}:{port}")
    print("Model will be loaded and kept in memory for fast generation!")
    uvicorn.run(app, host=host, port=port)

@cli.command()
@click.argument('prompt')
@click.option('--host', default=DEFAULT_HOST, help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
@click.option('--filename', help='Output filename (auto-generated if not provided)')
@click.option('--aspect-ratio', default='16:9', help='Image aspect ratio')
@click.option('--steps', default=50, help='Number of inference steps')
@click.option('--cfg-scale', default=4.0, help='CFG scale')
@click.option('--seed', default=42, help='Random seed')
def generate(prompt: str, host: str, port: int, filename: Optional[str], aspect_ratio: str, 
            steps: int, cfg_scale: float, seed: int):
    """Generate image using the server"""
    server_url = f"http://{host}:{port}"
    
    try:
        with httpx.Client() as client:
            # Check server status
            status_response = client.get(f"{server_url}/status")
            if status_response.status_code != HTTPStatus.OK:
                click.echo(f"Server not responding at {server_url}")
                sys.exit(1)
            
            status_data = status_response.json()
            if not status_data.get("ready"):
                click.echo("Server is not ready (model not loaded)")
                sys.exit(1)
            
            if not filename:
                click.echo(f"Server is ready. Next image will be: img{status_data['next_image_num']:04d}.png")
            click.echo("Generating image...")
            
            # Generate image
            gen_request = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": steps,
                "true_cfg_scale": cfg_scale,
                "seed": seed,
                "filename": filename
            }
            
            gen_response = client.post(f"{server_url}/generate", json=gen_request)
            if gen_response.status_code != HTTPStatus.OK:
                click.echo(f"Generation failed: {gen_response.text}")
                sys.exit(1)
            
            result = gen_response.json()
            if not result.get("success"):
                click.echo(f"Generation failed: {result.get('message')}")
                sys.exit(1)
            
            # Save image
            output_filename = result["filename"]
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            with open(output_filename, 'wb') as f:
                image_data = base64.b64decode(result["image_data"])
                f.write(image_data)
            
            click.echo(f"Image saved as {output_filename}")
            
    except httpx.ConnectError:
        click.echo(f"Cannot connect to server at {server_url}")
        click.echo("Make sure the server is running with: python -m qwenimage server")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()