import argparse
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

# Load the trained model checkpoint
checkpoint = torch.load(model_dir / "model.pt", map_location=DEVICE)

# Load the tokenizer for GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint["tokenizer"])

# Load the Vision Transformer (ViT) model and processor
vit = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").to(DEVICE)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
vit.load_state_dict(checkpoint["vit"])
vit = vit.to(DEVICE).eval()

# Configure GPT-2 with cross-attention and load its weights
config = GPT2Config.from_pretrained('gpt2')
config.add_cross_attention = True
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
gpt2.load_state_dict(checkpoint["gpt2"])
gpt2 = gpt2.to(DEVICE).eval()

# Load the projection layer to align ViT and GPT-2 embeddings
proj = nn.Linear(vit.config.hidden_size, gpt2.config.n_embd)
proj.load_state_dict(checkpoint["proj"])
proj = proj.to(DEVICE).eval()

def encode_image(image_path):
    """
    Encodes an image into feature embeddings using the Vision Transformer (ViT).

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Projected feature embeddings of the image.
    """
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = vit(**inputs).last_hidden_state
    projected_feats = proj(feats)  # Project features to match GPT-2 embedding size
    return projected_feats

def generate_caption(image_path, max_length=30):
    """
    Generates a caption for the given image.

    Args:
        image_path (str): Path to the input image.
        max_length (int): Maximum length of the generated caption.

    Returns:
        str: Generated caption.
    """
    image_embed = encode_image(image_path)  # Encode the image into embeddings

    # Initialize input IDs with the BOS (beginning of sentence) token
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Generate and print the caption
    caption = generate_caption(args.image_path)
    print(f"Generated caption: {caption}")
