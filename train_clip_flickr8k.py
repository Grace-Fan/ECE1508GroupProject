import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

class Flickr8kDataset(Dataset):
    """
    Class to load and process Flickr8k dataset images and captions as pairs.
    """
    def __init__(self, images_folder, captions_file, image_list=None, max_samples=None):
        self.images_folder = images_folder
        self.data = []

        # Group captions by image (5 captions per image)
        caption_dict = defaultdict(list)

        # Read captions file, format: image_name#img_key \t caption
        # Eg: 1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Ensure each image has a caption
                if len(parts) != 2:
                    continue
                # Group captions by image name (ignoring the #img_key part)
                img_key = parts[0].split('#')[0]
                caption = parts[1]
                # Collect all captions under that image name
                caption_dict[img_key].append(caption)

        # iterate image and list of captions
        for img_name, captions in caption_dict.items():
            # Only load if in image lists (if provided)
            if image_list and img_name not in image_list:
                continue

            img_path = os.path.join(images_folder, img_name)
            if not os.path.exists(img_path):
                continue

            # Add each image for caption to training sample
            for caption in captions:
                self.data.append((img_path, caption))
                if max_samples and len(self.data) >= max_samples:
                    break
            if max_samples and len(self.data) >= max_samples:
                break

    def __len__(self):
        """
        Returns the total number of text-image pairs in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns individual text-image tensor
        """
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        return image, caption

def split_images_by_caption_group(captions_file, test_size=0.2, random_state=42):
    """
    Split image filenames into train and eval sets, ensuring that all captions (5 of them)
    for each image stay in the same set.
    """
    image_names = set()
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            img_key = parts[0].split('#')[0]
            image_names.add(img_key)
    
    image_names = sorted(list(image_names))  # Ensure consistent order
    train_imgs, eval_imgs = train_test_split(image_names, test_size=test_size, random_state=random_state)
    return train_imgs, eval_imgs

def evaluate(model, device, dataloader):
    """
    Evaluate the fine-tuned CLIP model on image-text retrieval task.

    Args:
        model: The CLIP model to evaluate.
        device: Torch device (CPU or CUDA).
        dataloader: DataLoader for the evaluation dataset.

    Prints:
        Recall@1, Recall@5, Recall@10 for image-to-text and text-to-image retrieval.
    """
    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    # No gradient computation needed during evaluation for efficiency
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding evaluation data"):
            # Move inputs to device (GPU/CPU)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass through the model to get image and text embeddings
            outputs = model(pixel_values=batch['pixel_values'], input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings to unit length for cosine similarity comparison
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Collect embeddings from all batches
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)

    # Concatenate all batch embeddings into one tensor
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    # 2D similarity matrix how similar image i is to caption j
    sim_matrix = all_image_embeds @ all_text_embeds.t()
    n = sim_matrix.size(0)

    def recall_at_k(sim_matrix, k):
        """
        Compute Recall@k for image-to-text retrieval.
        Check how often correct caption is among the top-k most similar captions for each image (row)
        """
        # How many matches fall within top k for each image
        correct = 0
        # Loop image-caption pair
        for i in range(n):
            # select top k captions with highest similarity
            topk = sim_matrix[i].topk(k).indices
            # Check if correct caption at i is in top-k
            if i in topk:
                correct += 1
        return correct / n

    def recall_at_k_text_to_image(sim_matrix, k):
        """
        Compute Recall@k for text-to-image retrieval.
        Check how often correct image is among the top-k most similar images for each caption (row)
        """
        correct = 0
        for i in range(n):
            topk = sim_matrix[:, i].topk(k).indices
            if i in topk:
                correct += 1
        return correct / n

    # Calculate recall metrics for image-to-text retrieval
    r1 = recall_at_k(sim_matrix, 1)
    r5 = recall_at_k(sim_matrix, 5)
    r10 = recall_at_k(sim_matrix, 10)

    # Calculate recall metrics for text-to-image retrieval
    r1_t2i = recall_at_k_text_to_image(sim_matrix, 1)
    r5_t2i = recall_at_k_text_to_image(sim_matrix, 5)
    r10_t2i = recall_at_k_text_to_image(sim_matrix, 10)

    # Print recall results
    print(f"\nEvaluation Results:")
    print(f"Image-to-Text Retrieval: R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}")
    print(f"Text-to-Image Retrieval: R@1={r1_t2i:.3f}, R@5={r5_t2i:.3f}, R@10={r10_t2i:.3f}\n")

    # Labels are the diagonal (i-th image matches i-th text)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Compute loss: symmetric cross entropy
    loss_i2t = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_t2i = torch.nn.CrossEntropyLoss()(sim_matrix.t(), labels)
    loss = (loss_i2t + loss_t2i) / 2
    print(f"Evaluation Loss: {loss.item():.4f}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained CLIP model and processor from HuggingFace
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Custom collate function with padding for images and captions
    def collate_fn(batch):
        images, captions = zip(*batch)
        inputs = processor(text=list(captions), images=list(images), return_tensors="pt", padding=True)
        return inputs
    
    model = model.to(device)

    images_folder = os.path.join('Flickr8k', 'Images', 'Flicker8k_Dataset')
    captions_file = os.path.join('Flickr8k', 'Captions', 'Flickr8k.token.txt')

    # Split full dataset by image name
    train_imgs, eval_imgs = split_images_by_caption_group(captions_file)

    # Logging train/eval split sizes
    print(f"Total images: {len(train_imgs) + len(eval_imgs)}")
    print(f"Training images: {len(train_imgs)}")
    print(f"Evaluation images: {len(eval_imgs)}")

    print("\nFirst training image:")
    print(f"  {train_imgs[0]}")
    print("\nFirst evaluation image:")
    print(f"  {eval_imgs[0]}")

    # Setup training dataset and loader
    train_dataset = Flickr8kDataset(images_folder, captions_file, image_list=train_imgs, max_samples=3000)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Setup evaluation dataset and loader
    eval_dataset = Flickr8kDataset(images_folder, captions_file, image_list=eval_imgs, max_samples=1000)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=5e-6)
    epochs = 3

    model.train()  # Set model to training mode

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader)
        for batch in loop:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass to compute CLIP model outputs: image and text embeddings
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings to unit length for cosine similarity
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Cosine similarity logits between all image-text pairs in batch
            logits_per_image = image_embeds @ text_embeds.t()
            logits_per_text = text_embeds @ image_embeds.t()

            # Target labels: diagonal (matching pairs)
            labels = torch.arange(len(image_embeds), device=device)

            # Calculate symmetric cross-entropy loss for image-to-text and text-to-image
            loss_img = torch.nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = torch.nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar with loss
            loop.set_postfix(loss=loss.item())
        
        # Evaluate after each epoch
        evaluate(model, device, eval_loader)
        model.train()  # set back to training mode

    # Save the fine-tuned model for later use
    save_path = './clip_flickr8k_finetuned'
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

if __name__ == '__main__':
    train()
