DATASET_PATH=data/iharmony4
CACHED_PATH=data/iharmony4/vae
torchrun --nproc_per_node=8 --nnodes=1 --standalone \
har_cache.py \
--img_size 256  \
--vae_path checkpoints/pretrained_models/vae/kl16.ckpt \
--vae_embed_dim 16 \
--batch_size 64 \
--data_path ${DATASET_PATH} \
--cached_path ${CACHED_PATH} 