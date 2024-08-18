CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "<set your checkpoint path here>" \
--test_image_path "<set your testing image dir here>" \
--test_gt_path "<set your testing mask dir here>" \
--save_path "<set your prediction results dir here>"