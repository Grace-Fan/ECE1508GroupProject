# scripts
- `scripts/download_flickr8k.py`  
  Downloads and extracts the Flickr8k dataset (images and captions) into `dataset/Flickr8k/`.

  Flickr8k.token.txt contains image and caption pairing. It contains 8092 images × 5 captions

- `scripts/train_clip_flickr8k.py`
  Condition image features extracted from pretrained CLIP model (`openai/clip-vit-base-patch32`). Maps CLIP image embeddings into GPT-2's embedding space, then prepends these embeddings as a prefix to the token embeddings of captions. Trained on Flickr8k dataset, which contains images paired with captions. After training, the fine-tuned GPT-2 model and projection layer are saved for later use in caption generation. (needs more work)

- `scripts/caption_generator.py`
  Test script to generate captions from downloaded images (needs more work)
