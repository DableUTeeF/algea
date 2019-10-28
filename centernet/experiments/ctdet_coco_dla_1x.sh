cd src
# train
python main.py ctdet --exp_id coco_dla_1x --batch_size 16 --master_batch 9 --lr 5e-4 --gpus 0 --num_workers 2
# test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
