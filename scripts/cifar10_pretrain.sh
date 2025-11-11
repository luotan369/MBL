

x_size=(3,32,32) # 
y_size=(3,32,32)


xhidden=64    
xsize=128    
yhidden=256  
depth=8      
level=3      
learn_top=False
batchsize=10   
testgap=10000000000  
loggap=10     
savegap=100  
infergap=100000 
grad_clip=0     
grad_norm=10   
adv_loss=False  
only=True       
tanh=False      
clamp=True      
model_path=''  
support_set='Resnet50'  
query_set='Resnet18'    
lr=0.001  
margin=5.0              
regularizer=0   


logroot="./save/cifar10/"  
train_dataset_root="data/data5000train/data_cifar10_mi_resnet18.npy"  
valid_dataset_root="data/data5000valid/data_cifar10_mi_resnet18.npy"  
curriculum=True
unifor=False
down_sample_x=1
down_sample_y=1
meta_iteration=3
CUDA_VISIBLE_DEVICES=0,1 \
python train.py --dataset_name=cifar10 \
  --x_size=${x_size}  --y_size=${y_size}\
  --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=${xhidden} --y_hidden_channels=${yhidden} \
  --x_hidden_size=${xsize} --flow_depth=${depth} --num_levels=${level} --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=${testgap} --log_gap=${loggap} --inference_gap=${infergap} --lr=${lr} --max_grad_clip=${grad_clip} \
  --max_grad_norm=${grad_norm} --save_gap=${savegap}  --regularizer=${regularizer} --adv_loss=${adv_loss} \
  --learn_top=${learn_top} --model_path=${model_path} --tanh=${tanh} --only=${only} --margin=${margin} --clamp=${clamp} \
  --name=${name} --support_set=${support_set} --query_set=${query_set} --down_sample_x=${down_sample_x} --down_sample_y=${down_sample_y} \
  --meta_iteration=${meta_iteration}  --curriculum=${curriculum} --unifor=${unifor}


