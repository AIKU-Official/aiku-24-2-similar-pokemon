export CUDA_VISIBLE_DEVICES=0
python pnp/preprocess.py \
    --data_path /home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg \
    --save_dir /home/aikusrv04/pokemon/similar_pokemon/pnp/latents \
    --start_index 0

python pnp/pnp.py \
    --data_path "/home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg" \
    --save_dir "/home/aikusrv04/pokemon/similar_pokemon/dataset/pnp_images"\
    --start_index 0

python inference_user.py \
    --query_fp "/home/aikusrv04/pokemon/similar_pokemon/dataset/images/seungryong_kim.jpg" \
    --k 3
