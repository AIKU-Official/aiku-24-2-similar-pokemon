import cv2
from pathlib import Path
import base64

def retrieve_pokemon_images(paths: list):
	pokemon_data = []
	for img_path in paths:
		try:
			print(f"Processing image: {img_path}")
			pokemon_name = Path(img_path).parts[-2]

			img = cv2.imread(img_path)
			if img is None:
				raise ValueError(f"Image at {img_path} could not be loaded")

			# Base64 인코딩
			_, buffer = cv2.imencode(".jpg", img)
			img_base64 = base64.b64encode(buffer).decode("utf-8")

			pokemon_data.append({"name": pokemon_name, "image": img_base64})
		except Exception as e:
			print(f"Error processing path {img_path}: {e}")
			pokemon_data.append({"name": "N/A", "image": None})
	# print(f"POKEMON_DATA: {pokemon_data}")
	return pokemon_data
