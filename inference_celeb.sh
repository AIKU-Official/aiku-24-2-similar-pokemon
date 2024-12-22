export CUDA_VISIBLE_DEVICES=0
python inference_celeb.py \
  --query_fp "similar_pokemon/dataset/celebs" \
  --k 3
