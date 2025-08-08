import torch
from pathlib import Path
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, ViTModel, ViTImageProcessor
from torch import nn

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(__file__).resolve().parent
dataset_dir = current_dir.parent / 'dataset'
model_dir = current_dir.parent / 'model'
image_path = dataset_dir / 'test' / 'test_image.jpg' 

# Load tokenizer and models
checkpoint = torch.load(model_dir / "model.pt", map_location=DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint["tokenizer"])

vit = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").to(DEVICE)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
vit.load_state_dict(checkpoint["vit"])
vit = vit.to(DEVICE).eval()

config = GPT2Config.from_pretrained('gpt2')
config.add_cross_attention = True
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
gpt2.load_state_dict(checkpoint["gpt2"])
gpt2 = gpt2.to(DEVICE).eval()

proj = nn.Linear(vit.config.hidden_size, gpt2.config.n_embd)
proj.load_state_dict(checkpoint["proj"])
proj = proj.to(DEVICE).eval()

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = vit(**inputs).last_hidden_state
    projected_feats = proj(feats)
    return projected_feats

# Generate caption
def generate_caption(image_path, max_length=30):
    image_embed = encode_image(image_path)

    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=DEVICE)
    encoder_attention_mask = torch.ones(image_embed.shape[:2], dtype=torch.long).to(DEVICE)
    output_ids = gpt2.generate(
        input_ids=input_ids,
        encoder_hidden_states=image_embed,
        encoder_attention_mask=encoder_attention_mask,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False
    )
    return tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)

# Run example
caption = generate_caption(image_path)
print(f"Generated caption: {caption}")
