CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "<set your pretrained hiera path here>" \
--train_image_path "<set your training image dir here>" \
--train_mask_path "<set your training mask dir here>" \
--save_path "<set your checkpoint saving dir here>" \
--epoch 20 \
--lr 0.001 \
--batch_size 12