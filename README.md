# Flickr8k CLIP Fine-Tuning

This repository contains Python scripts to download the Flickr8k dataset and fine-tune the OpenAI CLIP model on it for image-caption alignment. The fine-tuned model can later be used for caption generation or related tasks.

---

## Files

- `download_flickr8k.py`  
  Downloads and extracts the Flickr8k dataset (images and captions) into a local folder `Flickr8k/`.

  Flickr8k.token.txt contains image and caption pairing. It contains 8092 images × 5 captions

- `train_clip_flickr8k.py`  
  Fine-tunes the pretrained CLIP model (`openai/clip-vit-base-patch32`) on the Flickr8k dataset captions and images, assess generalization with retrieval metrics after each epoch, then saves the trained model locally.

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
   python download_flickr8k.py
   ```
3. **Train CLIP**
   ```
   python train_clip_flickr8k.py 
   ```
