#!/bin/bash

# Kiểm tra nếu thư mục model đã tồn tại
if [ ! -d "./models/<model-folder>" ]; then
  echo "Đang tải model từ Hugging Face..."
  
  # Cài đặt Git Large File Storage (Git LFS) nếu chưa có
  git lfs install

  # Clone model từ Hugging Face về thư mục models/
  git clone https://huggingface.co/THP2903/Erax_llm_ocr

  echo "Model đã được tải về thư mục models/Erax_llm_ocr."
else
  echo "Model đã tồn tại trong thư mục models/Erax_llm_ocr."
fi
