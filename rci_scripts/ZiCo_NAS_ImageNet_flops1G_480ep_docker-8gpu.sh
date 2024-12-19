#!/bin/bash
#SBATCH --job-name=ZiCo1G
#SBATCH --time=196:00:00
#SBATCH --partition=amdgpuextralong
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=128gb
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --output=/home/neumalu1/data/zico/zico_%j.out
# -------------------------------



save_dir=./save_dir/ZiCo_NAS_ImageNet_flops1G_480ep_docker_8gpu
mkdir -p ${save_dir}


resolution=224
epochs=480


singularity exec --nv --bind /mnt:/mnt docker://luken86/aznas horovodrun -np 8 python ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
  --dist_mode single --workers_per_gpu 8 \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZiCo/ZiCo_imagenet1k_flops1G_res224.txt \
  --teacher_arch geffnet_tf_efficientnet_b3_ns \
  --teacher_pretrained \
  --teacher_input_image_size 320 \
  --teacher_feature_weight 1.0 \
  --teacher_logit_weight 1.0 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --use_se \
  --target_downsample_ratio 16 \
  --batch_size_per_gpu 64 --save_dir ${save_dir} \
  --world-size 1 \
  --dist_mode horovod
  



