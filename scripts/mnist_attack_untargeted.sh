
CUDA_VISIBLE_DEVICES=0,1 \

python attack.py \
    --target_model_name='vgg19' \
    --dataset_name=mnist \
    --dataset_root='data/MNIST' \
    --generator_path='save/mnist/meta/1515/checkpoints/checkpoint_6000.pth.tar'\
    --surrogate_model_names=resnet18 \
    --max_query=10000 --class_num=10 --linf=0.1 \
    --down_sample_x=1 \
    --down_sample_y=1  \
    --attack_method=cgattack --finetune_glow --finetune_reload --finetune_perturbation    