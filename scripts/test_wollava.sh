CUDA_VISIBLE_DEVICES=0 python test/test_wollava.py \
--pretrained_model_name_or_path SD3-Medium_PATH \
--transformer_model_name_or_path present/smfsr_q \
--image_path present/benchmark_realsr/test_LR \
--output_dir present/benchmark_realsr/output \
--prompt_path present/prompt/realsr_txt \
--num_inference_steps 1