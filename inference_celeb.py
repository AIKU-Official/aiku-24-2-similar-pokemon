
import pickle
from typing import List
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from CLIP import CLIP
from DINO import DINO
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--query_fp', type=str, default="query.jpg")
    parser.add_argument('--k', type=int, default="3")

    return parser.parse_args()


DINO = DINO(index_path="/home/aikusrv04/pokemon/similar_pokemon/faiss_indexes/DINO_faiss_index.bin")
CLIP = CLIP(index_path="/home/aikusrv04/pokemon/similar_pokemon/faiss_indexes/CLIP_faiss_index.bin")

def examine_similarity(image_file):

    with open("/home/aikusrv04/pokemon/similar_pokemon/pokemon_list.pkl", "rb") as file:
        pokemons = pickle.load(file)

    DINO_scores, DINO_indexes = DINO(image_file, k=len(pokemons))
    CLIP_scores, CLIP_indexes = CLIP(image_file, k=len(pokemons))
    dino_df = pd.DataFrame({"index": DINO_indexes, "score": DINO_scores})
    clip_df = pd.DataFrame({"index": CLIP_indexes, "score": CLIP_scores})
    merged_df = pd.merge(dino_df, clip_df, on="index", suffixes=("_dino", "_clip"))

    dino_weight = 1.0
    clip_weight = 1.5
    merged_df["avg_score"] =  (dino_weight * merged_df["score_dino"] + clip_weight * merged_df["score_clip"]) / (dino_weight + clip_weight)
    merged_df["path"] = merged_df["index"].apply(lambda idx: pokemons[idx])

    sorted_df = merged_df.sort_values(by="avg_score", ascending=True)
    print("Sorted Scores")
    print(sorted_df.head(10))

    top_3_paths_temp = sorted_df.iloc[:3]["path"].tolist()
    top_3_paths=[]
    for each_path in top_3_paths_temp:
        new_path = each_path.replace("/vol/", "/home/aikusrv04/pokemon/similar_pokemon/dataset/")
        top_3_paths.append(new_path)
    print(f"TOP_3_PATHS: {top_3_paths}")
    return top_3_paths

def retrieve_images(paths: List[str]):
    images = []
    for img_path in paths:
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
    query_img_fp = Path(args.query_fp)
    assert query_img_fp.is_file(), f"File does not exist: {query_img_fp}"

    query_img = cv2.cvtColor(cv2.resize(cv2.imread(str(query_img_fp)), (416, 416)), cv2.COLOR_BGR2RGB)
    query_img = Image.fromarray(query_img)

    top_3_paths = examine_similarity(query_img)
    results = retrieve_images(top_3_paths)
    results = [Image.fromarray(img) for img in results]
    save_results_to_canvas(query_img, results, "output/retrieval_result.png")

if __name__ == "__main__":
    main()