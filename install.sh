#!/bin/bash

# Cài đặt các thư viện cơ bản
pip install transformers sentencepiece tensorboard -U -q
pip install datasets transformers[sentencepiece] -U -q
pip install accelerate bitsandbytes peft -U -q
pip install scipy protobuf==3.20 numpy==1.23 scikit-learn pillow opencv-python opencv-contrib-python -U -q
pip install pydantic fastapi fake_headers json_repair uvicorn redis redisvl beautifulsoup4 asyncio -U -q
pip install "fastapi[standard]" fastapi_limiter -U -q

# vLLM QWen2-VL-7B nếu bạn chạy vLLM
pip uninstall transformers -y
pip uninstall vllm -y
pip install vllm -U
pip install qwen-vl-utils

# Cài đặt phiên bản transformers từ GitHub (commit cụ thể)
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830

# Cài đặt flash-attn và các thư viện liên quan
pip install flash-attn --no-build-isolation
pip install transformers accelerate bitsandbytes optimum
