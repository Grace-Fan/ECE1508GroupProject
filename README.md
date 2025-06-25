# Flickr8k CLIP Fine-Tuning

This repository contains Python scripts to download the Flickr8k dataset and fine-tune the OpenAI CLIP model on it for image-caption alignment. The fine-tuned model can later be used for caption generation or related tasks.

---

## Files

- `scripts/download_flickr8k.py`  
  Downloads and extracts the Flickr8k dataset (images and captions) into `dataset/Flickr8k/`.

  Flickr8k.token.txt contains image and caption pairing. It contains 8092 images × 5 captions

- `scripts/train_clip_flickr8k.py`
  Condition image features extracted from pretrained CLIP model (`openai/clip-vit-base-patch32`). Maps CLIP image embeddings into GPT-2's embedding space, then prepends these embeddings as a prefix to the token embeddings of captions. Trained on Flickr8k dataset, which contains images paired with captions. After training, the fine-tuned GPT-2 model and projection layer are saved for later use in caption generation. (needs more work)

- `scripts/caption_generator.py`
  Test script to generate captions from downloaded images (needs more work)

- `requirements.txt`  
  Python package dependencies for running both scripts in a Windows virtual environment.

---

## Setup

1. **Create and activate a virtual environment (Windows):**

   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Download dataset**
   ```
   python scripts\download_flickr8k.py
   ```
3. **Train CLIP**
   ```
   python scripts\train_clip_flickr8k.py 
   ```
