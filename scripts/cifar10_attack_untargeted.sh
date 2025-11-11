
CUDA_VISIBLE_DEVICES=0,1 \

python attack.py \
    --target_model_name='vgg19' \
    --dataset_name=cifar10 \
    --dataset_root='data' \
    --generator_path='save/cifar10/meta/rand_meta_4000/checkpoints/checkpoint_4000.pth.tar' \
    --surrogate_model_names=resnet18 \
    --max_query=2000 --class_num=10 --linf=0.0325 \
    --down_sample_x=1 \
    --down_sample_y=1  \
    --attack_method=cgattack --finetune_glow --finetune_reload --finetune_perturbation