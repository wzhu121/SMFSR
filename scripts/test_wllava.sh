CUDA_VISIBLE_DEVICES=0,1 python test/test_wllava.py \
--pretrained_model_name_or_path SD3-Medium_PATH \
--transformer_model_name_or_path present/smfsr_q \
--image_path present/benchmark_realsr/test_LR \
--output_dir present/benchmark_realsr/output \
--num_inference_steps 1
