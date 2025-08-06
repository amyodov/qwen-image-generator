# Qwen Image Generator

This is a simple but enhanced image generator for Qwen-Image model, released on Huggingface at [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image).

The original model contains the suggested Python code to startup, setup the pipeline and invoke the model. This project is just a slightly enhanced version of the very same code, with bare minimum of improvements to make its usage more comfortable. This is not indended to be all-encompassing Diffusion models launcher; more like, "make it more convenient to generate different images, with little clutter".

## Features

* Client-server architecture:
  * The server loads the model and pipeline to keep them hot, receives requests, and generates the images in a common directory (auto-incrementing the number of the image), one by one.
  * The client allows you to quickly add a job to generate an image (or multiple images with the same prompt and different seeds).
* Job-based architecture:
  * Single (preloaded, prewarmed) worker for sequential jobs; multiple jobs don't slow down each other;
  * The CUDA (or CPU if feasible) based generation occurs in a separate thread, so the server API is still accessible.
* Better Python architecture:
  * [Uv](https://docs.astral.sh/uv/)-based project, for easier maintenance of dependencies/Python version.
  * Tested with Python 3.12.
* Reasonable defaults:
  * The image directory is automatically generated for you, if missing.
  * The file names are auto-incremented, and don't block each other.
  * The communication IP port is has the same defaults both for client and server (so you don't need to mention it explicitly); you can still configure it if needed.

## Installation

Install [Uv](https://docs.astral.sh/uv/), as recommended by their official documentation.

Clone the repository.

(If you are going to use CUDA-based Torch libraries) Install the Torch libraries with CUDA support:
```sh
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install/sync the rest of the packages:
```sh
uv sync
```

## Usage

Run the server:
```sh
uv run python qwenimage server
```

Add a job to generate an image, with some prompt:
```sh
uv run python qwenimage generate "A cute fluffy white kitten looks at you with blue eyes, while holding a huge cleaver in its paws. Comicbook style, pastel colors."
```

... or a job generating multiple images with the same prompt:
```sh
uv run python qwenimage generate -n 10 "A cute fluffy white kitten looks at you with blue eyes, while holding a huge cleaver in its paws. Comicbook style, pastel colors."```
```

Note: you can always run the commands with `--help` argument, to see their options:

```sh
uv run python qwenimage server --help
uv run python qwenimage generate --help
```

## Contacts

Author: Alex Myodov
