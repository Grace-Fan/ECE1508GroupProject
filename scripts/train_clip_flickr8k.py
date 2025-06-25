import os
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPModel, CLIPProcessor

# === Paths ===
current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir.parent / 'dataset'
model_dir = current_dir.parent / 'model'
os.makedirs(model_dir, exist_ok=True)

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
MAX_LEN = 30
MAX_SAMPLES = 500  # per epoch

# === Load models ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Important for padding
gpt2.resize_token_embeddings(len(tokenizer))
gpt2 = gpt2.to(DEVICE)

# === Linear projection from CLIP to GPT2 ===
clip_embed_dim = clip_model.config.projection_dim  # 512
gpt2_embed_dim = gpt2.config.n_embd  # 768
clip_to_gpt_proj = nn.Linear(clip_embed_dim, gpt2_embed_dim).to(DEVICE)

# === Dataset ===
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.data = []
        with open(caption_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0]
                self.data.append((img_name, caption))
        if max_samples:
            self.data = self.data[:max_samples]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        image = self.transform(image)
        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

# === Prepare DataLoader ===
image_dir = dataset_dir / "Flickr8k/Images/Flicker8k_Dataset"
caption_file = dataset_dir / "Flickr8k/Captions/Flickr8k.token.txt"
dataset = ImageCaptionDataset(image_dir, caption_file, max_samples=MAX_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# === Optimizer ===
params = list(gpt2.parameters()) + list(clip_to_gpt_proj.parameters())
optimizer = torch.optim.AdamW(params, lr=5e-5)

# === Training ===
gpt2.train()
clip_to_gpt_proj.train()

for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0

    for images, captions in loop:
        # Get image features from CLIP
        inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        image_embeds = clip_to_gpt_proj(image_features).unsqueeze(1)  # (B, 1, 768)

        # Tokenize captions
        tokenized = tokenizer(list(captions), padding="max_length", max_length=MAX_LEN, return_tensors="pt", truncation=True)
        input_ids = tokenized.input_ids.to(DEVICE)
        attention_mask = tokenized.attention_mask.to(DEVICE)

        # Prepare labels
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        # Pad labels for image prefix token
        labels = torch.cat([
            torch.full((labels.size(0), 1), -100).to(DEVICE),
            labels
        ], dim=1)

        # Prepare image embeddings and inputs_embeds
        image_embeds = clip_to_gpt_proj(image_features).unsqueeze(1)  # (B, 1, 768)
        inputs_embeds = gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)

        # Pad attention mask
        attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1)).to(DEVICE),
            attention_mask
        ], dim=1)

        # Forward pass
        outputs = gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n if loop.n else 1))

# === Save model ===
save_path = model_dir / "clip_caption_gpt2"
os.makedirs(save_path, exist_ok=True)
torch.save({
    "gpt2": gpt2.state_dict(),
    "proj": clip_to_gpt_proj.state_dict(),
    "tokenizer": tokenizer.name_or_path
}, save_path / "model.pt")

print(f"Model saved to: {save_path}")
