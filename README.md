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
3. **Train CLIP**
   ```
   python scripts\train_Vit_flickr8k.py 
   ```

4. **Run Test Generator**
    ```
    python scripts\caption_generator.py
    ```