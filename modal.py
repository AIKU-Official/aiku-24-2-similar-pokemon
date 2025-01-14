
import io
import pickle
from typing import List

import cv2
import numpy as np
import pandas as pd
from CLIP import CLIP
from DINO import DINO
from fastapi import File, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from utils import retrieve_pokemon_images

import modal


class Item(BaseModel):
    name: str
    image: str

class Items(BaseModel):
    images: List[Item]


# https://modal.com/docs/guide/lifecycle-functions#build
# https://stackoverflow.com/a/69125651
image = modal.Image.debian_slim(python_version="3.9.18").pip_install(
	"torch", "torchvision", "transformers", "numpy", "pandas", "Pillow", "opencv-python-headless", "faiss-cpu", "fastapi[standard]", 'xformers', "pydantic"
)

volume = modal.Volume.from_name("pokemons")
# modal volume put pokemons ./pokemons
# modal volume put pokemons ./faiss_indexes
# modal volume put pokemons ./pokemon_list.pkl

app = modal.App(name = "pokemon-similarity", image=image)



@app.cls(image = image, volumes={"/vol": volume}, gpu="T4", timeout = 3600)
class Model:
	@modal.build()
	@modal.enter()
	def prepare(self):
		self.DINO = DINO(index_path="/vol/faiss_indexes/DINO_faiss_index.bin")
		self.CLIP = CLIP(index_path="/vol/faiss_indexes/CLIP_faiss_index.bin")
		with open("/vol/pokemon_list.pkl", "rb") as file:
			self.pokemons = pickle.load(file)

	def examine_similarity_test(self, image_file):
		DINO_scores, DINO_indexes = self.DINO(image_file, k=len(self.pokemons))
		CLIP_scores, CLIP_indexes = self.CLIP(image_file, k=len(self.pokemons))

		dino_df = pd.DataFrame({"index": DINO_indexes, "score": DINO_scores})
		clip_df = pd.DataFrame({"index": CLIP_indexes, "score": CLIP_scores})
		merged_df = pd.merge(dino_df, clip_df, on="index", suffixes=("_dino", "_clip"))

		dino_weight = 1.0
		clip_weight = 1.5
		merged_df["avg_score"] =  (dino_weight * merged_df["score_dino"] + clip_weight * merged_df["score_clip"]) / (dino_weight + clip_weight)
		merged_df["path"] = merged_df["index"].apply(lambda idx: self.pokemons[idx])

		sorted_df = merged_df.sort_values(by="avg_score", ascending=True)
		print("Sorted Scores")
		print(sorted_df.head(10))

		top_3_paths = sorted_df.iloc[:3]["path"].tolist()
		print(f"TOP_3_PATHS: {top_3_paths}")
		return top_3_paths


	@modal.web_endpoint(method="POST")
	async def infer(self, image: UploadFile = File(...)):
			# Read file data into memory
			file_bytes = await image.read()  # File bytes from the uploaded file
			np_array = np.frombuffer(file_bytes, np.uint8)
			query_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
			query_image = Image.fromarray(query_image)

			# Perform your inference logic
			top_3_paths = self.examine_similarity_test(query_image)
			top_pokemons = retrieve_pokemon_images(top_3_paths)

			return Items(images=top_pokemons)

if __name__ == "__main__":
	app.serve()
