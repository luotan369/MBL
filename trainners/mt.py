import copy
import os
import torch
import torch.nn as nn
import datetime
import numpy as np
from torch.utils.data import DataLoader
import time

from utils.tools import preprocess, save_model
from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
import random


class WeightBoundTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=2):
        super().__init__()
   
        self.embed = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4
            ),
            num_layers=num_layers
        )
        

        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        

        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x):
 

        
    
        x = self.embed(x)           # (N, d_model)

        x = x.unsqueeze(1)          # (N, 1, d_model)
        x = x + self.pos_embedding   
      
        x = x.permute(1, 0, 2)      # (seq_len=1, batch_size=N, d_model)
        
  
        x = self.transformer(x)     # (1, N, d_model)
        
 
        x = torch.mean(x, dim=[0,1])  # (d_model)
        

        output = self.fc(x)         # (1)
        

        return 0.0001 + 0.9998 * torch.sigmoid(output)
        
 
class Trainer(object):

    def __init__(self, flow, adv_models, query_set_models, optim, train_set, valid_set, args, cuda):
  

        
        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(args.log_root, args.name if args.name != '' else date)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        args_str = ''
        for key, value in sorted(vars(args).items()):
            args_str += f'{key}: {value}\n'
        with open(os.path.join(self.log_dir, 'args.txt'), 'a') as file_object:
            file_object.write(f'Name: {args.name}\n')
            file_object.write(args_str)

        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.weight_bound_transformer = WeightBoundTransformer()
        self.weight_bound_transformer = self.weight_bound_transformer.to('cuda:0')
        self.trans_optim= torch.optim.Adam(self.weight_bound_transformer.parameters(), lr=0.0002)
        self.pre_weight = [0] * 10 
        # model
        self.flow = flow
        self.flow.eval()
        self.adv_models = adv_models  
        self.query_set_models = query_set_models
        self.adv_model_id = 0

        self.optim = optim

 
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.unifor = args.unifor 
        self.transformer = args.transformer
        self.noshuffle = args.noshuffle
        self.acc_l2 = args.acc_l2 
        self.rd_uni = args.rd_uni
        if self.unifor: 
            if self.rd_uni:
                print('加载数据集时打乱次序')
                self.trainingset_loader = DataLoader(train_set,
                                                    batch_size=self.batch_size,  # batch_size 10
                                                    shuffle=True,
                                                    drop_last=True)
            else:
                print('加载数据集时不打乱次序')
                self.trainingset_loader = DataLoader(train_set,
                                                     batch_size=self.batch_size,  # batch_size 10
                                                     shuffle=False,
                                                     drop_last=True)
        else:
            print('加载数据集时打乱次序')
            self.trainingset_loader = DataLoader(train_set,
                                                 batch_size=self.batch_size,  # batch_size 10
                                                 shuffle=True,
                                                 drop_last=True)
        self.validset_loader = DataLoader(valid_set,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False)

        self.num_epochs = args.num_epochs
        self.global_step = args.num_steps
        self.label_scale = args.label_scale
        self.label_bias = args.label_bias
        self.x_bins = args.x_bins
        self.y_bins = args.y_bins
        self.margin = args.margin

        self.num_epochs = args.num_epochs
        self.log_gap = args.log_gap
        self.inference_gap = args.inference_gap
        self.test_gap = args.test_gap
        self.save_gap = args.save_gap
        self.target = args.target
        self.target_label = args.target_label
        if self.target:
            self.target_label = torch.tensor(self.target_label).cuda().unsqueeze(0).expand(self.batch_size, -1)
            print('Target Attack Training: target label: ', args.target_label)
        self.openset = args.openset
        self.openset_top5_labels_list = [41, 394, 497, 776, 911]
        self.args = args

        # device
        self.cuda = cuda
        self.label_num = 1000


        # meta training:
        self.meta_iteration = args.meta_iteration
        self.meta_test_batch = 1
        self.temp_meta_path = args.name + '.pth'

 
        self.linf = 8. / 255
        self.adversary_list = []
        self.adversary_iter_num = 10
        for i in range(len(self.adv_models)):
            self.adversary_list.append(  
                LinfPGDAttack(
                    self.adv_models[i], loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
                    nb_iter=self.adversary_iter_num, eps_iter=2. / 255, rand_init=True, clip_min=0.0,
                    clip_max=1.0, targeted=False))
        print('Meta Learner Initialization done.')

    def norm_l2(self, gradients):
    
        l2_norm = 0
        for g in gradients:
            if g is not None:
                l2_norm += torch.sum(g ** 2)  
        return torch.sqrt(l2_norm)
    
    def norm_l2_t(self, gradients):
     
        l2_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in gradients if g is not None]), p=2)
        return l2_norm


    def weight_compute(self,l2g_list, n):

        flag=0  
        n_list=[] 
        w = []  
        temp_w=[]  
        for i,x in enumerate(l2g_list):
            if x==0:
                n_list.append(i)
                flag=1

        n-=len(n_list) 

        if flag==0:
            sum = 0
            for l2g in l2g_list:
                sum += 1 / l2g
            for l2g in l2g_list:
                wi = (n / l2g) / sum
                w.append(wi)
                temp_w.append(wi)
        else:
            sum = 0
            for l2g in l2g_list:
                if l2g !=0:
                    sum += 1 / l2g
            for i,l2g in enumerate(l2g_list):
                if l2g !=0:
                    wi = (n / l2g) / sum
                    w.append(wi)
                    temp_w.append(wi)
                else:
                    w.append(0)
 
        if len(temp_w)==1:
            temp_w[0]=torch.tensor(1)
            temp_weights =temp_w
        elif len(temp_w)==0: 
            temp_weights =[torch.tensor(1/10) for _ in range(10)]
        else:
            if max(temp_w)==min(temp_w):
                for i in range(len(temp_w)):
                    temp_w[i]=torch.tensor(1/len(temp_w))
                temp_weights =temp_w
            else:
                normalized_weights = [(i - min(temp_w)) / (max(temp_w) - min(temp_w)) for i in temp_w] 
            
                lower_bound = 1/10
                upper_bound = 1
            
                temp_weights = [lower_bound + (upper_bound - lower_bound) * a for a in normalized_weights]
  
        scaled_weights=[]
        
        if flag==0:
            scaled_weights=temp_weights
        else:
            counter=0
            stacked_tensor = torch.stack(temp_weights) 
            stacked_tensor = stacked_tensor.to(torch.float32) 
            temp_mean = torch.mean(stacked_tensor) 
      
            for i in w:
                if i !=0:
                    scaled_weights.append(temp_weights[counter])
                    counter+=1
                else:
                    scaled_weights.append(torch.tensor(1/8))  

        return scaled_weights

    

    def transformer_compute_v1(self,l2g_list, n,weight_bound_transformer):
        flag=0  # 
        n_list=[] #
        w = []  #
        temp_w=[]  # 
        upper_bound=0
        lower_bound=0
        for i,x in enumerate(l2g_list):
            if x==0:
                n_list.append(i)
                flag=1

        n-=len(n_list)  

        if flag==0:
            sum = 0
            for l2g in l2g_list:
                sum += 1 / l2g
            for l2g in l2g_list:
                wi = (n / l2g) / sum
                w.append(wi)
                temp_w.append(wi)
        else:
            sum = 0
            for l2g in l2g_list:
                if l2g !=0:
                    sum += 1 / l2g
            for i,l2g in enumerate(l2g_list):
                if l2g !=0:
                    wi = (n / l2g) / sum
                    w.append(wi)
                    temp_w.append(wi)
                else:
                    w.append(torch.tensor(0).to('cuda'))
    
        if len(temp_w)==1:
            temp_w[0]=torch.tensor(1)
            temp_weights =temp_w
        elif len(temp_w)==0: 
            temp_weights =[torch.tensor(1/10) for _ in range(10)]  
        else:
            if max(temp_w)==min(temp_w):
                for i in range(len(temp_w)):
                    temp_w[i]=torch.tensor(1/len(temp_w))
                temp_weights =temp_w
            else:
                normalized_weights = [(i - min(temp_w)) / (max(temp_w) - min(temp_w)) for i in temp_w]  
                stacked_tensor = torch.stack(w).unsqueeze(1)  # Initialize stacked_tensor first
                device = stacked_tensor.device  # Get the device of stacked_tensor
                w = [tensor.to(device) for tensor in w]  # Move all tensors in w to the same device
                bounds = weight_bound_transformer(stacked_tensor)  
                tensor_list = bounds.flatten().tolist()
                # upper_bound=max(tensor_list)
                lower_bound=min(tensor_list)         
                temp_weights = [lower_bound + (1 - lower_bound) * a for a in normalized_weights]
  
        scaled_weights=[]
        
        if flag==0:
            scaled_weights=temp_weights
        else:
            counter=0
            stacked_tensor = torch.stack(temp_weights) # new add
            stacked_tensor = stacked_tensor.to(torch.float32) 
            temp_mean = torch.mean(stacked_tensor) # new add  
      
            for i in w:
                if i !=0:
                    scaled_weights.append(temp_weights[counter])
                    counter+=1
                else:
                    if 1==lower_bound:
                        scaled_weights.append(torch.tensor(1/8))  
                    else:
                        scaled_weights.append(torch.tensor(((1/8)/(1-1/10))*(1-lower_bound)))  
              

    
    def transformer_compute(self,l2g_list, n,weight_bound_transformer):
 

        flag=0  
        n_list=[] 
        w = []  #
        temp_w=[]  
        count=0 

        stack_tensor = torch.stack(l2g_list) # new add
        stack_tensor = stack_tensor.to(torch.float32) #new add
        temp_mean = torch.mean(stack_tensor) # new add

        for i,x in enumerate(l2g_list):
            if x==0:
                l2g_list[i]=temp_mean
            else:
                count+=1
            n_list.append(i) 


        sum = 0
        if count==1 or count ==0:
            temp_w=[torch.tensor(1/10) for _ in range(10)]
            if count==1:
         
                temp_w[n_list[0]]=torch.tensor(1)
            return temp_w 
        else:
            for l2g in l2g_list:
                if l2g !=0:
                    sum += 1 / l2g
                wi = (n / l2g) / sum
                temp_w.append(wi)      
   
        stacked_tensor = torch.stack(temp_w).unsqueeze(1)

        bounds = weight_bound_transformer(stacked_tensor) 
        tensor_list = bounds.flatten().tolist()
   
        upper_bound=max(tensor_list)
        lower_bound=min(tensor_list)
        if max(temp_w)==min(temp_w):
            for i in range(len(temp_w)):
                temp_w[i]=torch.tensor(1/n) 
            scaled_weights =temp_w
        else:
            if upper_bound==lower_bound:
                for i in range(len(temp_w)):
                    temp_w[i]=torch.tensor(lower_bound)
            normalized_weights = [(i - min(temp_w)) / (max(temp_w) - min(temp_w)) for i in temp_w]  
            scaled_weights = [lower_bound + (upper_bound - lower_bound) * a for a in normalized_weights]
        with open ('save/gradients_out.txt', 'a') as file:
            file.write(f"上下限:{tensor_list}\n")
            file.write(f"正则化后权重:{scaled_weights}\n")  

        return scaled_weights

    def mean_gradient_loss1(self, losses, model, optimizer,weight_bound_transformer):

        
        l2_graients_list = []  
        loss_sum = 0 
       
        for i,loss in enumerate(losses):
            
  
            _loss = loss

            if loss < 0:
                loss=-loss
            

            optimizer.zero_grad()
            model.zero_grad()

           
            loss.backward(retain_graph=True)

            current_gradients=[]
            for param in model.parameters():
                if param.grad is not None:
                    current_gradients.append(param.grad.clone())
                else:
 
                    current_gradients.append(torch.zeros_like(param))

            l2_gradients = self.norm_l2_t(current_gradients)  
            l2_graients_list.append(l2_gradients)

        if self.transformer:
            weights = self.transformer_compute_v1(l2_graients_list, len(l2_graients_list),weight_bound_transformer)  # edit weight_compute
        else:
            weights = self.weight_compute(l2_graients_list, len(l2_graients_list))
        if self.acc_l2:
            all_zeros = all(x == 0 for x in self.pre_weight)
            if not all_zeros:
                for i in range(len(weights)):
                    weights[i]=self.pre_weight[i]*(2/5)+weights[i]*(3/5)  # todo
                with open ('save/gradients_out.txt', 'a') as file:
                        file.write(f"preweights:{self.pre_weight}\n")

        self.pre_weight = weights


        model.zero_grad()
        optimizer.zero_grad()

        for loss, weight in zip(losses, weights):
            loss_sum += loss * weight

        return loss_sum

    def adv_loss(self, y, label, model, optim,weight_bound_transformer):

        loss = 0.0
        for adv_model in self.adv_models:
            logits = adv_model(y)  

            if not self.target:

                one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                label = label.reshape(-1, 1)  
                one_hot.scatter_(1, label, 1) 
                one_hot = one_hot.bool()  
                diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits), -1), dim=1)[0]  # todo
              
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin 

            else:

                target_one_hot = torch.zeros_like(logits, dtype=torch.uint8)
                target_one_hot.scatter_(1, self.target_label, 1)
                target_one_hot = target_one_hot.bool()
   
                diff = -logits[target_one_hot]
                margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin

            if self.unifor:
  
                loss += self.mean_gradient_loss1(margin, model, optim,weight_bound_transformer)  
            else:
                loss += margin.mean()
                
        loss /= len(self.adv_models)
        return loss

    def augmentation(self, x, true_lab, no_adv=False):
        if self.args.adv_aug and (not no_adv):
            # x = preprocess(x, 1.0, 0.0, self.x_bins, False)
            if self.args.adv_rand:
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]
                iter_num = np.random.randint(0, 20 + 1)

                if iter_num > 0:
                    adversary = self.adversary_list[model_idx]
                    with ctx_noparamgrad_and_eval(model_chosen):
                        x = adversary.perturb(x, None)
                else:
                    x = preprocess(x, 1.0, 0.0, self.x_bins, False)

            elif not no_adv:
                model_idx = np.random.randint(0, len(self.adv_models))
                model_chosen = self.adv_models[model_idx]

                adversary = self.adversary_list[model_idx]

                with ctx_noparamgrad_and_eval(model_chosen):
                    x = adversary.perturb(x, None)
        else:
            x = preprocess(x, 1.0, 0.0, self.x_bins, False)
        return x

    def schedule(self, loss_prob, loss_cls, epoch):
        if loss_prob <= 0:
            return 0.01
        else:
            return 0.02

    def meta_train(self, batch_data, epoch):
        """
        Reptile(一种高效的元学习算法处理) meta training process
        """
        if self.args.curriculum:
            x = batch_data['cln_img']
            label = batch_data['true_lab']
        else:
            x = batch_data[0]
            label = batch_data[1]
        batch_length = len(label)
        if self.cuda:
            x = x.cuda()
            label = label.cuda()
        label = label.long()

       
        processed_x = self.augmentation(x, label, epoch < self.args.adv_epoch)  
        _flow = copy.deepcopy(self.flow)  

        self.optim = torch.optim.Adam(self.flow.parameters(), lr=0.0004, betas=self.args.betas,
                                      weight_decay=self.args.regularizer)  

        # todo
        for i in range(self.meta_iteration): 
            y, logdet = self.flow.decode(processed_x, return_prob=True)  

       
            loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                torch.clamp(torch.clamp(y, -self.linf, self.linf) + x, 0, 1), label, self.flow, self.optim ,self.weight_bound_transformer)  # 这里先限制对抗样本噪声的大小（最大强度Linf），再限制生成的对抗样本的大小【0，1】
            
            loss = loss_cls  
            self.optim.zero_grad()
            if self.transformer:
                self.trans_optim.zero_grad()
            loss.backward()  
            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.max_grad_norm)
            # step
            if self.transformer:
                self.trans_optim.zero_grad()
            self.optim.step()  
 
        lr = self.schedule(loss_prob.data, loss_cls.data, epoch) 
        dic = self.flow.state_dict()  
        keys = list(dic.keys()) 

        meta_state = _flow.state_dict()  
        for key in keys:
            dic[key] = meta_state[key] + lr / batch_length * 2 / self.meta_iteration * (dic[key] - meta_state[key])
    
        
        return loss_prob.data, loss_cls.data

    def meta_test(self):
        def check(model, image, label):
            prob_output = torch.nn.functional.softmax(model(image), dim=1)
            pred_lable = torch.argmax(prob_output, dim=1)
            print(pred_lable.item(), label)
            return pred_lable.item() != label

        mean_loss_prob, mean_loss_cls = 0, 0
        with torch.no_grad():
            for batch_data in self.validset_loader:
                images = batch_data[0]
                labels = batch_data[1]
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                labels = labels.long()
                y, logdet = self.flow.decode(images, return_prob=True)  
                loss_prob, loss_cls = torch.mean(logdet), self.adv_loss(
                    torch.clamp(torch.sign(y) * 8. / 255 + images, 0, 1), labels)
                mean_loss_prob += loss_prob
                mean_loss_cls += loss_cls
        with open(os.path.join(self.log_dir, "meta_test_loss.txt"), "a") as f:
            f.write("mean_loss_prob: {:.5f}, mean_loss_cls: {:.5f}".format(mean_loss_prob, mean_loss_cls) + "\n")
        return mean_loss_prob, mean_loss_cls



    def train(self):
 
        self.flow.train()
        starttime = time.time()
        # run
        num_batchs = len(self.trainingset_loader)  
        batches = list(self.trainingset_loader)
        total_its = self.num_epochs * num_batchs  
        best_loss=float('inf')  

        # parameter_loss = None
        for epoch in range(self.num_epochs):
            mean_loss_prob, mean_loss_cls = 0, 0

  
            if self.unifor and self.noshuffle==False and self.rd_uni==False:
                print("shuffle unifor set")
                indices = list(range(num_batchs))
                random.seed(232 + epoch)
                random.shuffle(indices)
                shuffled_batches = [batches[i] for i in indices]
                self.trainingset_loader = iter(shuffled_batches)

            for batch_id, batch in enumerate(self.trainingset_loader):
     
                loss_prob, loss_cls = self.meta_train(batch, epoch)
                mean_loss_prob += loss_prob
                mean_loss_cls += loss_cls
                if (self.global_step + 1) % self.test_gap == 0:
                    self.meta_test()
                # save model
                if (self.global_step + 1) % self.save_gap == 0:
                    save_model(self.flow, self.optim, self.checkpoints_dir, self.global_step + 1)
                self.global_step = self.global_step + 1
                # TODO change the print
                currenttime = time.time()
                elapsed = currenttime - starttime
                if self.global_step % 50 == 0:
                    print(
                        "Iteration: {}/{} \t Epoch: {}/{} \t Elapsed time: {:.2f} \t Meta train loss prob: {:.4f} \t loss cls: {:.4f}".format(
                            self.global_step, total_its, epoch, self.num_epochs, elapsed, loss_prob, loss_cls))

                if batch_id % self.args.log_gap == 0:
                    mean_loss_prob = float(mean_loss_prob / float(num_batchs))
                    mean_loss_cls = float(mean_loss_cls / float(num_batchs))
                    with open(os.path.join(self.log_dir, "Epoch_NLL.txt"), "a") as f:
                        currenttime = time.time()
                        elapsed = currenttime - starttime
                        f.write(
                            "epoch: {} \t iteration: {}/{} \t elapsed time: {:.2f}\t mean loss prob: {:.5f}\t mean loss cls: {:.5f}".format(
                                epoch, self.global_step, total_its, elapsed, mean_loss_prob, mean_loss_cls) + "\n")
