##################
### Submission ###
##################
python3 src/inference_GAN.py \
    -model_path  "ckpt/GAN/G_latent500_epoch149.pth" \
    -batch_size 32 \
    -latent_size 500 \
    -num_samples 32 \
    -dest_path $1


# Pb 3.2.2
# python src/inference_GAN.py \
#     -model_path  "ckpt/GAN/G_latent500_epoch149.pth" \
#     -dest_path "results/GAN_randomly_generated.png" \
#     -batch_size 32 \
#     -latent_size 500 \
#     -num_samples 32 

# python src/train_GAN.py \
#     -latent_size 500 \
#     -epoch 150