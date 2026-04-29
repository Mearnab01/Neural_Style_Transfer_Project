To train:
!python train.py \
 --batch_size 4 \
 --epochs 10 \
 --experiment exp2 \
 --content_dir /content/nst_data/content \
 --style_dir /content/nst_data/style

To Resume:
!python train.py \
 --batch_size 4 \
 --epochs 10 \
 --experiment exp2 \
 --content_dir /content/nst_data/content \
 --style_dir /content/nst_data/style \
 --resume \
 --decoder_path experiments/exp2/checkpoint.pth
