import torch
from pathlib import Path
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPProcessor, CLIPModel
from torch import nn

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir.parent / 'dataset'
model_dir = current_dir.parent / 'model' / "clip_caption_gpt2"
image_path = dataset_dir / 'test' / 'test_image.jpg' 

# === Load tokenizer and models ===
checkpoint = torch.load(model_dir / "model.pt", map_location=DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint["tokenizer"])
tokenizer.pad_token = tokenizer.eos_token

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2.load_state_dict(checkpoint["gpt2"])
gpt2 = gpt2.to(DEVICE).eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

proj = nn.Linear(clip_model.config.projection_dim, gpt2.config.n_embd)
proj.load_state_dict(checkpoint["proj"])
proj = proj.to(DEVICE).eval()

# === Preprocess image ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    prefix_embed = proj(image_features).unsqueeze(1)  # (1, 1, 768)
    return prefix_embed

# === Generate caption ===
def generate_caption(image_path, max_length=30):
    prefix_embed = preprocess_image(image_path)

    generated = prefix_embed  # Initial prefix for GPT2
    input_ids = None

    for _ in range(max_length):
        outputs = gpt2(inputs_embeds=generated, return_dict=True)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

        if input_ids is None:
            input_ids = next_token
        else:
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        next_embed = gpt2.transformer.wte(next_token)
        generated = torch.cat([generated, next_embed], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    caption = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return caption.strip()

# === Run example ===
caption = generate_caption(image_path)
print(f"Generated caption: {caption}")
