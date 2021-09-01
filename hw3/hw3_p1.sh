##################
### Submission ###
##################
bash get_VAE_model.sh
python3 src/inference_VAE.py --generate \
    -model_path  "ckpt/VAE/VAE_lr0.001_kldw0.1_MSE0.0134_KLD53.4119.pth" \
    -batch_size 32 \
    -dest_path $1


# Pb. 1.3.4
# python src/inference_VAE.py \
#     -model_path  "ckpt/VAE/VAE_lr0.001_kldw0.1_MSE0.0134_KLD53.4119.pth" \
#     -dest_path "results/VAE_randomly_generated.png" \
#     -batch_size 32 \
#     --generate



# Pb. 1.3.5
# python src/inference_VAE.py \
#     -model_path  "ckpt/VAE/VAE_lr0.001_kldw0.1_MSE0.0134_KLD53.4119.pth" \
#     -dest_path "results/VAE_tSNE_KLD53.4119" \
#     -batch_size 32 \
#     -kld_weight 0.1\
#     -num_samples 32 \
#     --tsne


###################
### Traininging ###
###################

# python src/train_VAE.py \
#     -model_dir "results" \
#     -batch_size 128 \
#     -epoch 50 \
#     -lr 1e-3 \
#     -latent_size 2048 \
#     -kld_weight 0.0001
