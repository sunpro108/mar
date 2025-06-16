RESUME_DIR=output/mar_large_kl16_0613
OUTPUT_DIR=output/mar_large_kl16_0613
IMAGENET_PATH=data/imagenets/vae
torchrun --nproc_per_node=8 --nnodes=1 --standalone \
main_mar.py \
--img_size 256  \
--vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_large --diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 32 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${RESUME_DIR} \
--cached_path ${IMAGENET_PATH} \
--use_cached