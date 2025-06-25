# === Import necessary libraries ===
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from tqdm import tqdm

# === Configuration ===
IMAGE_DIR = "Flickr8k/Images/Flicker8k_Dataset"  # Directory containing Flickr8k images
CAPTION_FILE = "Flickr8k/Captions/Flickr8k.token.txt"  # File with image captions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
BATCH_SIZE = 32  # Number of samples per batch
EPOCHS = 5  # Number of training epochs

# === Load pre-trained CLIP model and processor ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Custom dataset for Flickr8k ===
class Flickr8kDataset(Dataset):
    def __init__(self, caption_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.data = []
        with open(caption_file, 'r') as f:
            for line in f:
                # Each line has format: 'image_name#caption_number<TAB>caption'
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0]  # Remove the #caption_number suffix
                self.data.append((img_name, caption))

        # Image preprocessing pipeline
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
        ])  # Do not convert to tensor or normalize

    def __len__(self):
        return len(self.data)  # Number of (image, caption) pairs
    
    def __getitem__(self, idx):
        # Get image path and caption
        img_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, img_name)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

# === Load dataset and create dataloader ===
dataset = Flickr8kDataset(CAPTION_FILE, IMAGE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# === Set up optimizer ===
optimizer = AdamW(model.parameters(), lr=5e-6)

max_samples = 1000 # samples per epoch
sample_count = 0

# === Training loop ===
model.train()  # Set model to training mode
for epoch in range(EPOCHS):
    total_loss = 0
    total_batches = (max_samples + BATCH_SIZE - 1) // BATCH_SIZE
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", total=total_batches, leave=True)
    for images, captions in loop:
        # Prepare inputs for CLIP
        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(DEVICE)

        # Forward pass through the model
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()  # Accumulate loss
        loop.set_postfix(loss=total_loss / (loop.n if loop.n else 1))

        sample_count += len(images)
        if sample_count >= max_samples:
            break
    sample_count = 0  # Reset sample count for next epoch
    # Print average loss for the epoch
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader)}")

# === Save the trained model and processor ===
SAVE_PATH = "clip_flickr8k"

model.save_pretrained(SAVE_PATH)       # Save model weights and config
processor.save_pretrained(SAVE_PATH)   # Save tokenizer + image processor config

print(f"Model saved to: {SAVE_PATH}")