import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer, ViTModel, ViTImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import evaluate  # using the 'evaluate' library for BLEU
from transformers.modeling_outputs import CausalLMOutput
from peft import LoraConfig, get_peft_model

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

lora_config_vit = LoraConfig(
    r=8,  # rank for low-rank approximation
    lora_alpha=32,  # scaling factor for LoRA weights
    target_modules=["query", "key", "value"],  # Apply LoRA to these modules in the attention layers
    lora_dropout=0.1,  # Dropout for low-rank matrices
    bias="none"  # No bias for LoRA adaptation
)

lora_config_gpt2 = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attn.c_attn", "mlp.c_fc"],  # Apply LoRA to GPT-2 attention and feed-forward layers
    lora_dropout=0.1,
    bias="none"
)

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
vit_lora = get_peft_model(vit, lora_config_vit)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttentionLayer, self).__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=12)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_embeds, image_embeds):
        """
        text_embeds: (batch_size, seq_len, hidden_size)
        image_embeds: (batch_size, num_image_tokens, hidden_size)
        attention_mask: (batch_size, seq_len + num_image_tokens)
        """
        # Linear projections for cross-attention (text queries, image keys/values)
        query = self.query_projection(text_embeds)  # (batch_size, seq_len, hidden_size)
        key = self.key_projection(image_embeds)  # (batch_size, num_image_tokens, hidden_size)
        value = self.value_projection(image_embeds)  # (batch_size, num_image_tokens, hidden_size)

        # Transpose for multihead attention (batch_size, hidden_size, seq_len)
        query = query.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        key = key.transpose(0, 1)  # (num_image_tokens, batch_size, hidden_size)
        value = value.transpose(0, 1)  # (num_image_tokens, batch_size, hidden_size)

        # Apply cross-attention
        attn_output, _ = self.attention(query, key, value)

        # Project back to hidden size
        output = self.output_projection(attn_output.transpose(0, 1))  # (batch_size, seq_len, hidden_size)
        return output

class GPT2WithCrossAttention(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2WithCrossAttention, self).__init__(config)
        self.cross_attention = CrossAttentionLayer(config.n_embd)  # Add cross-attention layer
        self.transformer = self.transformer  # GPT-2's transformer blocks

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, image_embeds=None):
        # Encode text input
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        hidden_states = outputs[0]

        # Apply cross-attention to the GPT-2 hidden states
        if image_embeds is not None:
            hidden_states = self.cross_attention(hidden_states, image_embeds)

        # Continue with the rest of GPT-2's forward pass
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)

gpt2 = GPT2WithCrossAttention.from_pretrained("gpt2").to(DEVICE)
gpt2.resize_token_embeddings(len(tokenizer))
# Freeze the pre-trained layers of GPT-2 (except the cross-attention layer)
for name, param in gpt2.named_parameters():
    if "cross_attention" not in name:
        param.requires_grad = False
for idx in [9, 10, 11]:  # Unfreeze layers 9, 10, 11
    for param in gpt2.transformer.h[idx].parameters():
        param.requires_grad = True
gpt2_lora = get_peft_model(gpt2, lora_config_gpt2)

vit_to_gpt2_proj = nn.Linear(vit.config.hidden_size, gpt2.config.n_embd).to(DEVICE)

dataset = ImageCaptionDataset(IMAGE_DIR, CAPTION_FILE, max_samples=NUM_SAMPLES)
train_set, val_set = split_dataset(dataset, val_split=VAL_SPLIT)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)

params = [p for p in gpt2_lora.parameters() if p.requires_grad] + list(vit_to_gpt2_proj.parameters()) + [p for p in vit_lora.parameters() if p.requires_grad]
optimizer = AdamW(params, lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def encode_images(images):
    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
    inputs = vit_processor(images=pil_images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = vit_lora(**inputs).pooler_output
    projected_feats = vit_to_gpt2_proj(feats)
    return projected_feats.unsqueeze(1)

def generate_caption(image_embed, max_len=MAX_LEN):
    generated = image_embed
    input_ids = None
    for _ in range(max_len):
        out = gpt2_lora(inputs_embeds=generated, return_dict=True)
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        if input_ids is None:
            input_ids = next_token
        else:
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        next_embed = gpt2_lora.transformer.wte(next_token)
        generated = torch.cat([generated, next_embed], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

def evaluate_bleu(references, hypotheses):
    # references: list of ground truth captions (strings)
    # hypotheses: list of generated captions (strings)
    # BLEU expects list of references per prediction, so wrap each ref in list
    results = bleu_metric.compute(predictions=hypotheses, references=[[ref] for ref in references])
    return results

for epoch in range(EPOCHS):
    gpt2_lora.train()
    vit_lora.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, captions in loop:
        optimizer.zero_grad()
        image_embeds = encode_images(images)

        tokenized = tokenizer(list(captions), padding="max_length", max_length=MAX_LEN, return_tensors="pt", truncation=True)
        input_ids = tokenized.input_ids.to(DEVICE)
        attention_mask = tokenized.attention_mask.to(DEVICE)

        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100
        labels = torch.cat([torch.full((labels.size(0), 1), -100).to(DEVICE), labels], dim=1)

        input_embeds = gpt2_lora.transformer.wte(input_ids)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = gpt2_lora(inputs_embeds=input_embeds, image_embeds=image_embeds, attention_mask=attention_mask, labels=labels)
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
    gpt2_lora.eval()
    vit_lora.eval()
    references, hypotheses = [], []
    for images, captions in val_loader:
        image_embed = encode_images(images)
        gen = generate_caption(image_embed)
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
    "gpt2": gpt2_lora.state_dict(),
    "vit": vit_lora.state_dict(),
    "proj": vit_to_gpt2_proj.state_dict(),
    "tokenizer": tokenizer.name_or_path
}, MODEL_DIR / "model_lora.pt")

print("Model saved.")
