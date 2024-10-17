import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Custom dataset class for loading images
class BrainImageDataset:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return image, self.image_files[idx]

# Main function to classify images using CLIP
def main():
    # Set the relative path to the images directory
    image_dir = "research-murtylab/paper_replication/data_murty185/data_murty185/images_185"
    
    # Output directory (where this script is located)
    output_dir = os.path.dirname(__file__)

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()

    # Prepare candidate labels for classification (according to the brain areas in the study)
    candidate_labels = [
        "face", "hand", "foot", "leg", "arm", "torso", "head", "body part",
        "animal", "car", "building", "landscape"
    ]

    # Create a dataset and process images
    dataset = BrainImageDataset(image_dir=image_dir)
    results = []  # Store results here
    for i in range(len(dataset)):
        image, img_name = dataset[i]

        # Prepare the inputs for CLIP
        inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to(device)
        
        # Pass inputs through the CLIP model
        outputs = model(**inputs)
        
        # Extract the similarity scores (logits) between the image and each candidate label
        logits_per_image = outputs.logits_per_image  # Shape: [1, num_labels]

        # Apply softmax to get probabilities
        probs = torch.softmax(logits_per_image, dim=1)

        # Get the most likely label (highest probability) for the image
        max_prob_idx = probs.argmax(dim=1).item()
        predicted_label = candidate_labels[max_prob_idx]

        # Append result (image name and predicted label) to the results list
        results.append((img_name, predicted_label))

    # Save the results in the same directory as the script
    with open(os.path.join(output_dir, 'clip_classification_results.txt'), 'w') as f:
        for img_name, label in results:
            f.write(f"{img_name}: {label}\n")

    print("Classification completed and results saved.")

if __name__ == "__main__":
    main()
