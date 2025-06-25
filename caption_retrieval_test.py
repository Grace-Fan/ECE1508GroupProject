import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load saved model and processor
model = CLIPModel.from_pretrained('./clip_flickr8k')
processor = CLIPProcessor.from_pretrained('./clip_flickr8k')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def retrieve_caption(image_path, candidate_captions):
    image = Image.open(image_path).convert('RGB')

    # Prepare inputs
    inputs = processor(text=candidate_captions, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'], input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        image_embed = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity between image and each caption
        similarities = (image_embed @ text_embeds.t()).squeeze(0)  # shape: (num_captions,)

    # Print similarity score for all captions
    for caption, score in zip(candidate_captions, similarities.tolist()):
        print(f"Caption: '{caption}' | Similarity score: {score:.4f}")

    # Find best matching caption
    best_idx = similarities.argmax().item()
    best_caption = candidate_captions[best_idx]
    similarity_score = similarities[best_idx].item()

    return best_caption, similarity_score

# Example usage:
image_path = 'download.jpg'
candidate_captions = [
    "A child is playing in the park.",
    "A group of people standing near a building.",
    "A dog running through a field.",
    "A girl raising her hand.",
    "A girl reading her book."
]

caption, score = retrieve_caption(image_path, candidate_captions)
print(f"Best caption: '{caption}' with similarity score {score:.4f}")
