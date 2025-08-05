#!/usr/bin/env python3
import os
import re
import sys
import uuid
import asyncio
import threading
import logging
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor
import torch
import click
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

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
    number: int = 1

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
    jobs_created: int
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

def _cuda_worker_thread(job_queue: deque, completed_jobs: deque, app_state: Dict[str, Any]):
    """CUDA worker thread - owns all GPU resources and operations"""
    try:
        # Load pipeline in this thread
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"
        
        logger.info("Loading model on %s in worker thread...", device)
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            cache_dir="./model_cache"
        )
        pipeline = pipeline.to(device)
        logger.info("Model loaded successfully on %s in worker thread!", device)
        
        # Signal main thread that we're ready
        app_state["cuda_ready"] = True
        app_state["device"] = device
        
        # Process jobs forever
        while not app_state.get("shutdown", False):
            if len(job_queue) == 0:
                threading.Event().wait(0.1)  # Small sleep
                continue
            
            # Get next job (thread-safe pop)
            try:
                job = job_queue.popleft()
            except IndexError:
                continue
            
            # Update job status
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now()
            app_state["current_job"] = job
            
            logger.info("Processing job %s: %s...", job.id, job.prompt[:50])
            
            try:
                # Determine filename and increment counter atomically
                if job.filename:
                    filename = job.filename
                else:
                    current_num = app_state["next_image_num"]
                    app_state["next_image_num"] += 1
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
                
                generator = torch.Generator(device=device).manual_seed(job.seed)
                
                # Generate image
                image = pipeline(
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
                
                logger.info("Job %s completed: %s", job.id, filename)
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                logger.exception("Job %s failed", job.id)
            
            # Move to completed jobs (keep last 10)
            completed_jobs.append(job)
            app_state["current_job"] = None
            
    except Exception as e:
        logger.exception("Worker thread failed")
        app_state["cuda_ready"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared state
    os.makedirs("images", exist_ok=True)
    max_num = scan_for_max_image_number("images")
    
    # Thread-safe collections and shared state
    job_queue = deque()
    completed_jobs = deque(maxlen=10)
    shared_state = {
        "next_image_num": max_num + 1,
        "cuda_ready": False,
        "device": None,
        "current_job": None,
        "shutdown": False
    }
    
    logger.info("Next image will be: img%04d.png", shared_state['next_image_num'])
    
    # Store in app state for FastAPI endpoints
    app.state.job_queue = job_queue
    app.state.completed_jobs = completed_jobs
    app.state.shared_state = shared_state
    
    # Start CUDA worker thread
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda-worker")
    cuda_future = executor.submit(_cuda_worker_thread, job_queue, completed_jobs, shared_state)
    
    # Wait for CUDA thread to signal readiness
    logger.info("Waiting for worker thread to load model...")
    while not shared_state.get("cuda_ready", False):
        await asyncio.sleep(0.1)
    
    logger.info("Worker thread ready!")
    
    yield
    
    # Shutdown: stop CUDA worker thread
    logger.info("Shutting down worker thread...")
    shared_state["shutdown"] = True
    executor.shutdown(wait=True)

# FastAPI app with lifespan
app = FastAPI(title="Qwen Image Generator", version="0.1.0", lifespan=lifespan)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status including queue information"""
    shared_state = app.state.shared_state
    
    current_job_info = None
    current_job = shared_state.get("current_job")
    if current_job:
        current_job_info = {
            "id": current_job.id,
            "prompt": current_job.prompt[:100] + "..." if len(current_job.prompt) > 100 else current_job.prompt,
            "started_at": current_job.started_at.isoformat() if current_job.started_at else None,
            "status": current_job.status
        }
    
    recent_completed = []
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
        ready=shared_state.get("cuda_ready", False),
        model_loaded=shared_state.get("cuda_ready", False),
        next_image_num=shared_state.get("next_image_num", 0),
        queue_length=len(app.state.job_queue),
        current_job=current_job_info,
        recent_completed=recent_completed
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Queue image generation job(s)"""
    if not app.state.shared_state.get("cuda_ready", False):
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    # Create N jobs
    jobs_created = 0
    for i in range(request.number):
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
        jobs_created += 1
    
    logger.debug("Queued %d job%s: %s... (Queue length: %d)", 
                 jobs_created, 's' if jobs_created > 1 else '', 
                 request.prompt[:50], len(app.state.job_queue))
    
    return GenerationResponse(
        success=True,
        jobs_created=jobs_created,
        message=f"{jobs_created} job{'s' if jobs_created > 1 else ''} queued successfully. Queue length: {len(app.state.job_queue)}"
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
    # Set up logging for server mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    click.echo(f"Starting server on {host}:{port}")
    click.echo("Model will be loaded and kept in memory for fast generation!")
    click.echo("Jobs will be processed in the background, even if clients disconnect.")
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
@click.option('-n', '--number', default=1, help='Number of images to generate')
def generate(prompt: str, host: str, port: int, filename: Optional[str], aspect_ratio: str, 
            steps: int, cfg_scale: float, seed: int, number: int):
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
            
            # Submit job(s)
            gen_request = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": steps,
                "true_cfg_scale": cfg_scale,
                "seed": seed,
                "filename": filename,
                "number": number
            }
            
            gen_response = client.post(f"{server_url}/generate", json=gen_request)
            if gen_response.status_code != HTTPStatus.OK:
                click.echo(f"Job submission failed: {gen_response.text}")
                sys.exit(1)
            
            result = gen_response.json()
            if not result.get("success"):
                click.echo(f"Job submission failed: {result.get('message')}")
                sys.exit(1)
            
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