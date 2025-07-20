import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, ViTModel, ViTImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import evaluate  # using the 'evaluate' library for BLEU

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / 'dataset' / 'Flickr8k'
MODEL_DIR = BASE_DIR / "model"
IMAGE_DIR = DATASET_DIR / "Images" / "Flicker8k_Dataset"
CAPTION_FILE = DATASET_DIR / "Captions" / "Flickr8k.token.txt"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
MAX_LEN = 30
NUM_SAMPLES = 5000
VAL_SPLIT = 0.1
LR = 3e-5

# Load BLEU metric
bleu_metric = evaluate.load("bleu")

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.data = []
        with open(caption_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0].strip()
                full_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(full_path):
                    self.data.append((img_name, caption))
                else:
                    print(f"[Warning] Skipping missing image: {img_name}")
        if max_samples:
            self.data = self.data[:max_samples]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        image = self.transform(image)
        return image, caption

def split_dataset(dataset, val_split=0.1):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

vit = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").to(DEVICE)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# Freeze all ViT params except last 4 layers and pooler
for name, param in vit.named_parameters():
    param.requires_grad = False
for name, param in vit.encoder.layer[-4:].named_parameters():
    param.requires_grad = True
for name, param in vit.pooler.named_parameters():
    param.requires_grad = True

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained('gpt2')
config.add_cross_attention = True  # enable cross-attention layers
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(DEVICE)
gpt2.resize_token_embeddings(len(tokenizer))

vit_to_gpt2_proj = nn.Linear(vit.config.hidden_size, gpt2.config.n_embd).to(DEVICE)

dataset = ImageCaptionDataset(IMAGE_DIR, CAPTION_FILE, max_samples=NUM_SAMPLES)
train_set, val_set = split_dataset(dataset, val_split=VAL_SPLIT)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)

params = list(gpt2.parameters()) + list(vit_to_gpt2_proj.parameters()) + [p for p in vit.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def encode_images(images):
    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
    inputs = vit_processor(images=pil_images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = vit(**inputs).last_hidden_state  # (B, N, D)
    projected_feats = vit_to_gpt2_proj(feats)  # (B, N, gpt2_dim)
    return projected_feats

@torch.no_grad()
def generate_caption(image_embed, max_len=MAX_LEN):
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)
    encoder_attention_mask = torch.ones(image_embed.shape[:2], dtype=torch.long).to(DEVICE)
    # output_ids = gpt2.generate(
    #     input_ids=input_ids,
    #     encoder_hidden_states=image_embed,
    #     encoder_attention_mask=encoder_attention_mask,
    #     max_length=max_len,
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.95,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    # )
    output_ids = gpt2.generate(
        input_ids=input_ids,
        encoder_hidden_states=image_embed,
        encoder_attention_mask=encoder_attention_mask,
        max_length=max_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False
    )
    return tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)

def evaluate_bleu(references, hypotheses):
    # references: list of ground truth captions (strings)
    # hypotheses: list of generated captions (strings)
    # BLEU expects list of references per prediction, so wrap each ref in list
    results = bleu_metric.compute(predictions=hypotheses, references=[[ref] for ref in references])
    return results

for epoch in range(EPOCHS):
    gpt2.train()
    vit.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, captions in loop:
        optimizer.zero_grad()
        image_embeds = encode_images(images)
        encoder_attention_mask = torch.ones(image_embeds.shape[:2], dtype=torch.long).to(DEVICE)

        captions = [tokenizer.bos_token + " " + cap.strip().lower() + " " + tokenizer.eos_token for cap in captions]
        tokenized = tokenizer(
            captions,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
            truncation=True
        )
    
        input_ids = tokenized.input_ids.to(DEVICE)
        attention_mask = tokenized.attention_mask.to(DEVICE)

        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        # input_embeds = gpt2.transformer.wte(input_ids)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = gpt2(input_ids=input_ids, encoder_hidden_states=image_embeds, encoder_attention_mask=encoder_attention_mask,
                           attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n or 1))

    scheduler.step()

    # Validation BLEU evaluation
    gpt2.eval()
    vit.eval()
    references, hypotheses = [], []
    for images, captions in val_loader:
        image_embed = encode_images(images)
        gen = generate_caption(image_embed)
        print(f"Generated caption: {gen}")
        print(f"Reference caption: {captions[0]}")
        references.append(captions[0])
        hypotheses.append(gen)

#FIXME: used to test BLEU
    # # Normalize and print samples to debug
    # def normalize(text):
    #     return text.lower().strip()

    # references_norm = [normalize(r) for r in references]
    # hypotheses_norm = [normalize(h) for h in hypotheses]

    # print("Sample references:", references_norm[:3])
    # print("Sample hypotheses:", hypotheses_norm[:3])

    # Wrap references for BLEU
    references_wrapped = [[ref] for ref in references]

    # Compute BLEU using evaluate
    bleu_metric = evaluate.load("bleu")
    scores = bleu_metric.compute(predictions=hypotheses, references=references_wrapped)

    print(f"[Validation BLEU] Epoch {epoch+1}: {scores['bleu']:.4f}")

# Save model
torch.save({
    "gpt2": gpt2.state_dict(),
    "vit": vit.state_dict(),
    "proj": vit_to_gpt2_proj.state_dict(),
    "tokenizer": tokenizer.name_or_path
}, MODEL_DIR / "model.pt")

print("Model saved.")
