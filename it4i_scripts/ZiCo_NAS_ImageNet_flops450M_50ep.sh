#!/bin/bash
#SBATCH --job-name ZiCo450M
#SBATCH --account OPEN-30-33
#SBATCH --partition qdgx
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 128
#SBATCH --time 24:00:00
#SBATCH --output=/home/neumalu/jobs/zico_%j.out
# -------------------------------
ml GCC/10.3.0
ml CMake/3.14.1
ml CUDA

eval "$(/home/neumalu/anaconda3/bin/conda shell.bash hook)"

conda activate zico
export OMP_NUM_THREADS=14

save_dir=./save_dir/ZiCo_NAS_ImageNet_flops450M_50ep
mkdir -p ${save_dir}

resolution=224
epochs=50


horovodrun -np 16 python ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
  --dist_mode single --workers_per_gpu 8 \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZiCo/ZiCo_imagenet1k_flops450M_res224.txt \
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
  



