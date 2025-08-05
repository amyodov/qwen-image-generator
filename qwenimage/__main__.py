#!/usr/bin/env python3
import os
import re
import sys
import uuid
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from http import HTTPStatus
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

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(BaseModel):
    id: str
    prompt: str
    negative_prompt: str = " "
    aspect_ratio: str = "16:9"
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int = 42
    filename: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_filename: Optional[str] = None

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
    queue_length: int
    current_job: Optional[Dict[str, Any]] = None
    recent_completed: list[Dict[str, Any]] = []

class GenerationResponse(BaseModel):
    success: bool
    job_id: str
    message: str

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

async def process_job_queue(app: FastAPI):
    """Background task to process the job queue"""
    while True:
        try:
            if not hasattr(app.state, 'job_queue') or len(app.state.job_queue) == 0:
                await asyncio.sleep(1)
                continue
            
            if not hasattr(app.state, 'pipeline') or app.state.pipeline is None:
                await asyncio.sleep(1)
                continue
            
            # Get next job
            job = app.state.job_queue.popleft()
            app.state.current_job = job
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now()
            
            print(f"Processing job {job.id}: {job.prompt[:50]}...")
            
            try:
                # Determine filename and increment counter
                if job.filename:
                    filename = job.filename
                else:
                    current_num = app.state.next_image_num
                    app.state.next_image_num += 1
                    filename = f"images/img{current_num:04d}.png"
                
                # Aspect ratio mapping
                aspect_ratios = {
                    "1:1": (1328, 1328),
                    "16:9": (1664, 928),
                    "9:16": (928, 1664),
                    "4:3": (1472, 1140),
                    "3:4": (1140, 1472)
                }
                
                if job.aspect_ratio not in aspect_ratios:
                    raise ValueError(f"Unsupported aspect ratio: {job.aspect_ratio}")
                
                width, height = aspect_ratios[job.aspect_ratio]
                
                positive_magic = "Ultra HD, 4K, cinematic composition."
                full_prompt = f"{job.prompt} {positive_magic}"
                
                generator = torch.Generator(device=app.state.device).manual_seed(job.seed)
                
                image = app.state.pipeline(
                    prompt=full_prompt,
                    negative_prompt=job.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=job.num_inference_steps,
                    true_cfg_scale=job.true_cfg_scale,
                    generator=generator
                ).images[0]
                
                # Save image to disk
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)
                
                # Mark job as completed
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.output_filename = filename
                
                print(f"Job {job.id} completed: {filename}")
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                print(f"Job {job.id} failed: {e}")
            
            # Move to completed jobs (keep last 10)
            if not hasattr(app.state, 'completed_jobs'):
                app.state.completed_jobs = deque(maxlen=10)
            app.state.completed_jobs.append(job)
            app.state.current_job = None
            
        except Exception as e:
            print(f"Error in job processing: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load pipeline and initialize state
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
    app.state.job_queue = deque()
    app.state.current_job = None
    app.state.completed_jobs = deque(maxlen=10)
    
    print(f"Next image will be: img{app.state.next_image_num:04d}.png")
    
    # Start background job processor
    job_processor_task = asyncio.create_task(process_job_queue(app))
    
    yield
    
    # Shutdown: cleanup
    job_processor_task.cancel()
    try:
        await job_processor_task
    except asyncio.CancelledError:
        pass
    app.state.pipeline = None

# FastAPI app with lifespan
app = FastAPI(title="Qwen Image Generator", version="0.1.0", lifespan=lifespan)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status including queue information"""
    current_job_info = None
    if hasattr(app.state, 'current_job') and app.state.current_job:
        job = app.state.current_job
        current_job_info = {
            "id": job.id,
            "prompt": job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "status": job.status
        }
    
    recent_completed = []
    if hasattr(app.state, 'completed_jobs'):
        for job in list(app.state.completed_jobs)[-5:]:  # Last 5 completed jobs
            recent_completed.append({
                "id": job.id,
                "prompt": job.prompt[:50] + "..." if len(job.prompt) > 50 else job.prompt,
                "status": job.status,
                "output_filename": job.output_filename,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            })
    
    return StatusResponse(
        status="running",
        ready=hasattr(app.state, 'pipeline') and app.state.pipeline is not None,
        model_loaded=hasattr(app.state, 'pipeline') and app.state.pipeline is not None,
        next_image_num=getattr(app.state, 'next_image_num', 0),
        queue_length=len(getattr(app.state, 'job_queue', [])),
        current_job=current_job_info,
        recent_completed=recent_completed
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Queue image generation job"""
    if not hasattr(app.state, 'pipeline') or app.state.pipeline is None:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    # Create job
    job = Job(
        id=str(uuid.uuid4()),
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        aspect_ratio=request.aspect_ratio,
        num_inference_steps=request.num_inference_steps,
        true_cfg_scale=request.true_cfg_scale,
        seed=request.seed,
        filename=request.filename,
        created_at=datetime.now()
    )
    
    # Add to queue
    app.state.job_queue.append(job)
    
    print(f"Queued job {job.id}: {job.prompt[:50]}... (Queue length: {len(app.state.job_queue)})")
    
    return GenerationResponse(
        success=True,
        job_id=job.id,
        message=f"Job queued successfully. Queue position: {len(app.state.job_queue)}"
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
    print("Jobs will be processed in the background, even if clients disconnect.")
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
            
            # Submit job
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
                click.echo(f"Job submission failed: {gen_response.text}")
                sys.exit(1)
            
            result = gen_response.json()
            if not result.get("success"):
                click.echo(f"Job submission failed: {result.get('message')}")
                sys.exit(1)
            
            job_id = result["job_id"]
            click.echo(f"Job submitted: {job_id}")
            click.echo(result["message"])
            
    except httpx.ConnectError:
        click.echo(f"Cannot connect to server at {server_url}")
        click.echo("Make sure the server is running with: python -m qwenimage server")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--host', default=DEFAULT_HOST, help='Server host')
@click.option('--port', default=DEFAULT_PORT, help='Server port')
def status(host: str, port: int):
    """Check server status and queue information"""
    server_url = f"http://{host}:{port}"
    
    try:
        with httpx.Client() as client:
            response = client.get(f"{server_url}/status")
            if response.status_code != HTTPStatus.OK:
                click.echo(f"Server not responding at {server_url}")
                sys.exit(1)
            
            data = response.json()
            
            click.echo(f"Server Status: {data['status']}")
            click.echo(f"Model Loaded: {data['model_loaded']}")
            click.echo(f"Ready: {data['ready']}")
            click.echo(f"Next Image Number: {data['next_image_num']}")
            click.echo(f"Queue Length: {data['queue_length']}")
            
            current_job = data.get('current_job')
            if current_job:
                click.echo(f"\nCurrent Job:")
                click.echo(f"  ID: {current_job['id']}")
                click.echo(f"  Prompt: {current_job['prompt']}")
                click.echo(f"  Started: {current_job['started_at']}")
            
            recent = data.get('recent_completed', [])
            if recent:
                click.echo(f"\nRecent Completed Jobs:")
                for job in recent:
                    status_symbol = "[OK]" if job['status'] == 'completed' else "[FAIL]"
                    click.echo(f"  {status_symbol} {job['id']}: {job['prompt']}")
                    if job['status'] == 'completed':
                        click.echo(f"    -> {job['output_filename']}")
                    elif job['error_message']:
                        click.echo(f"    -> Error: {job['error_message']}")
                        
    except httpx.ConnectError:
        click.echo(f"Cannot connect to server at {server_url}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()