# preprocess video
python data_utils/process.py data/chuanhu/chuanhu.mp4
# train head
python main.py data/chuanhu/ --workspace trial_chuanhu/ -O --iters 200000

# train (finetune lips for another 250000 steps, run after the above command!)
python main.py data/chuanhu/ --workspace trial_chuanhu/ -O --iters 250000 --finetune_lips

# train (torso)
# <head>.pth should be the latest checkpoint in trial_obama
python main.py data/chuanhu/ --workspace trial_chuanhu_torso/ -O --torso --head_ckpt trial_chuanhu/checkpoints/ngp.pth --iters 200000

# test with our own audio
python main.py data/chuanhu/ --workspace trial_chuanhu_torso/ -O --torso --test --aud data/female_eo.npy
