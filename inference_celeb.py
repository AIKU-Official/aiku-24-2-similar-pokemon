
import io
import pickle
from typing import List
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from CLIP import CLIP
from DINO import DINO
from PIL import Image
from utils import retrieve_pokemon_images
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--query_img', type=str, default="query.jpg")
    parser.add_argument('--k', type=int, default="3")

    return parser.parse_args()


def prepare(self):
		self.DINO = DINO(index_path="similar_pokemon/faiss_indexes/DINO_faiss_index.bin")
		self.CLIP = CLIP(index_path="similar_pokemon/faiss_indexes/CLIP_faiss_index.bin")
		with open("pokemon_list.pkl", "rb") as file:
			self.pokemons = pickle.load(file)

def examine_similarity(self, image_file):
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

def retrieve_images(paths: List[str]):
    images = []
    for img_path in paths:
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image at {img_path} could not be loaded. Skipping.")
            continue
        img = cv2.resize(img, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def save_results_to_canvas(query_img, results, save_path):

    w, h = query_img.size
    canvas = Image.new("RGB", (4 * w, h), color='white')
    canvas.paste(query_img, (0, 0))
    canvas.paste(results[0], (w, 0))
    canvas.paste(results[1], (2 * w, 0))
    canvas.paste(results[2], (3 * w, 0))
    canvas.save(save_path)
    print(f"Result saved to {save_path}")
    
def main():
    args = parse_args()
    query_img_fp = Path(args.query_img)
    # assert query_img_fp.is_file(), f"File does not exist: {query_img_fp}"

    # with open('/home/aikusrv04/pokemon/similar-pokemon/pnp/embeddings.json', 'r') as f:
    #     embeddings = json.load(f)
    embeddings = {key.replace("../", ""): value for key, value in embeddings.items()}
    image_paths = list(embeddings.keys())

    query_img = cv2.cvtColor(cv2.resize(cv2.imread(str(query_img_fp)), (416, 416)), cv2.COLOR_BGR2RGB)
    query_img = Image.fromarray(query_img)

    modes = [("clip", "clip_result3.png"), ("dino", "dino_result3.png"), ("average", "average_result3.png")]

    for mode, save_path in modes:
        top_3_paths = examine_similarity(query_img_fp, image_paths=image_paths, mode=mode)
        results = retrieve_images(top_3_paths)

        results = [Image.fromarray(img) for img in results]

        save_results_to_canvas(query_img, results, save_path)

if __name__ == "__main__":
	prepare()
	examine_similarity()
