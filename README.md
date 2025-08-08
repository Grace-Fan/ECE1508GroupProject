# ECE1508GroupProject

Contains:

- dataset folder
- model folder
- scripts folder
- requirements.txt

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
3. **Train ViT-GPT2**

   ```
   python scripts\train_ViT_GPT2.py
   ```

4. **Run Text Generator**
   ```
   python scripts\caption_generator.py
   ```
