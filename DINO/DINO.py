from typing import Tuple

import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class DINO:
	def __init__(self, index_path):
		dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
		faiss_index = faiss.read_index(index_path)
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		transform = T.Compose([
			T.Resize((244, 244)),
			T.CenterCrop(224),
			T.Lambda(lambda img: img if isinstance(img, Image.Image) else Image.fromarray(np.array(img))),
			T.ToTensor(),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])

		self.device = device
		self.transform_image = transform
		self.dino_model = dinov2_vitb14.to(self.device)
		self.faiss_index: faiss.IndexFlatL2 = faiss_index


	def __call__(self, query_img: Image, k: int = 3) -> Tuple:
		# 검색할 유명인 이미지
		with torch.no_grad():
			image = self.transform_image(query_img).unsqueeze(0).to(self.device)
			embedding = self.dino_model(image)
			query_embedding = np.array(embedding[0].cpu()).reshape(1, -1)
			faiss.normalize_L2(query_embedding)
		distances, indices = self.faiss_index.search(query_embedding, k)
		return (distances[0], indices[0])
