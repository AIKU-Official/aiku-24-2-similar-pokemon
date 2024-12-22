from typing import Tuple
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import faiss
import torch


class CLIP:
    def load_index(faiss_index_path='faiss_index.bin') -> faiss.IndexFlatL2:
        index = faiss.read_index(faiss_index_path)
        print(f"FAISS index loaded from {faiss_index_path}")
        return index
    
    def __init__(self, index_path):
        faiss_index = self.load_index(index_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.faiss_index = faiss_index

    def __call__(self, query_img: Image, k: int = 3) -> Tuple:
        with torch.no_grad():
            pixel_values = self.clip_processor(images=query_img, return_tensors = "pt")['pixel_values'].to(self.device)
            query_embedding = self.clip_model.get_image_features(pixel_values)[0].cpu().detach().numpy().reshape(1, -1)
            faiss.normalize_L2(query_embedding)
        
        distances, indices = self.faiss_index.search(query_embedding, k)
        return (distances[0], indices[0])
