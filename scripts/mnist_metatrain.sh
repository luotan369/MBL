

train_dataset_root="data/data6000train/data_mnist_mi_resnet18.npy"   
valid_dataset_root="data/data6000valid/data_mnist_mi_resnet18.npy"
dataset_name='mnist'
logroot='./save/mnist/meta'
epochs=10
batchsize=10
lr=0.001
adv_loss=True

model_path='save/mnist/mto/checkpoint_5900.pth.tar'  

support_set='resnet18'
query_set='vgg19'
curriculum=True 
unifor=False  
acc_l2=False 
transformer=False 
noshuffle=False  #
rd_uni=False  

CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --dataset_name=${dataset_name} --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=64 --y_hidden_channels=256 \
  --x_hidden_size=128 --flow_depth=8 --num_levels=3 --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=100000000 --log_gap=10 --inference_gap=1000000 --lr=${lr} --max_grad_clip=0 \
  --max_grad_norm=10 --save_gap=100  --regularizer=0 --adv_loss=${adv_loss} \
  --learn_top=False --model_path=${model_path} --tanh=False --only=True --margin=5.0 --clamp=True \
  --name=${name} --support_set=${support_set} --query_set=${query_set} --down_sample_x=1 --down_sample_y=1 --meta_iteration=5 \
  --curriculum=${curriculum} --unifor=${unifor} --acc_l2=${acc_l2} --rd_uni=${rd_uni} --transformer=${transformer} --noshuffle=${noshuffle}