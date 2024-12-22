export CUDA_VISIBLE_DEVICES=0
python inference_celeb.py \
  --query_fp "/home/aikusrv04/pokemon/similar_pokemon/dataset/celebs/Beyonce.png" \
  --k 3
