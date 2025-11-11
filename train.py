import argparse
import ast
import numpy
import torch
import torchvision
import data.datasets as datasets
from utils.tools import count_parameters
from utils.load_models import load_cifar_model, load_imagenet_model, load_mnist_model
from models.cglow import CondGlowModel as Generator
# from trainners import MetaLearner, Learner
from trainners import mt, Learner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train c-Glow')

    # input output path，？这个训练集，验证集分别路径地址的具体作用还不清晰
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("--train_dataset_root", type=str)
    parser.add_argument("--valid_dataset_root", type=str)

    # log root
    parser.add_argument("--log_root", type=str, default="")

    # C-Glow parameters
    parser.add_argument("--x_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--y_size", type=tuple, default=(3, 224, 224))
    parser.add_argument("--x_hidden_channels", type=int, default=128)
    parser.add_argument("--x_hidden_size", type=int, default=64)
    parser.add_argument("--y_hidden_channels", type=int, default=256)
    parser.add_argument("-K", "--flow_depth", type=int, default=8)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--learn_top", type=ast.literal_eval, default=False)

    # Dataset preprocess parameters
    parser.add_argument("--label_scale", type=float, default=1)  
    parser.add_argument("--label_bias", type=float, default=0.0)  
    parser.add_argument("--x_bins", type=float, default=256.0)  
    parser.add_argument("--y_bins", type=float, default=2.0)  
    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.9999))
    parser.add_argument("--eps", type=float, default=1e-8)  
    parser.add_argument("--regularizer", type=float, default=0.0)  
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--Lambda", type=float, default=5e-1)

    # Trainer parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--down_sample_x", type=int, default=8)
    parser.add_argument("--down_sample_y", type=int, default=8)
    parser.add_argument("--max_grad_clip", type=float, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--test_gap", type=int, default=1000) 
    parser.add_argument("--log_gap", type=int, default=1)
    parser.add_argument("--inference_gap", type=int, default=1000)
    parser.add_argument("--save_gap", type=int, default=1000)
    parser.add_argument("--adv_loss", type=ast.literal_eval, default=False)
    parser.add_argument("--tanh", type=ast.literal_eval, default=False)  
    parser.add_argument("--only", type=ast.literal_eval, default=False)
    parser.add_argument("--clamp", type=ast.literal_eval, default=False)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--class_size", type=int, default=-1)
    parser.add_argument("--label", type=int, default=0)

  
    parser.add_argument("--adv_aug", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_rand", type=ast.literal_eval, default=False)
    parser.add_argument("--nes", type=ast.literal_eval, default=False)
    parser.add_argument("--new_form", type=ast.literal_eval, default=False)
    parser.add_argument("--normalize_grad", type=ast.literal_eval, default=False)
    parser.add_argument("--adv_epoch", type=int, default=0)

    # model path
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--name", type=str, default="")  

    # CMA args
    parser.add_argument("--cma_update_max_epoch", type=int, default=20)
    parser.add_argument("--cma_popsize", type=int, default=8)
    parser.add_argument("--cma_sigma0", type=int, default=10000)
    parser.add_argument("--cma_seed", type=int, default=666)
    parser.add_argument("--cma_ftarget", type=int, default=0)  

    # Meta args
    parser.add_argument("--support_set", type=str, default="Resnet18")
    parser.add_argument("--query_set", type=str, default="Resnet18")
   
    parser.add_argument("--curriculum", type=ast.literal_eval, default=False,
                        help="use only hard data")  
    parser.add_argument("--unifor", type=ast.literal_eval, default=False)
    parser.add_argument("--transformer", type=ast.literal_eval, default=False)
    parser.add_argument("--noshuffle", type=ast.literal_eval, default=False)
    parser.add_argument("--acc_l2", type=ast.literal_eval, default=False)
    parser.add_argument("--rd_uni", type=ast.literal_eval, default=False)
    parser.add_argument("--meta_iteration", type=int, default=1)

    
    parser.add_argument("--target", type=ast.literal_eval, default=False)
    parser.add_argument("--target_label", type=int, default=None)
  
    parser.add_argument("--openset", type=ast.literal_eval, default=False)

    args = parser.parse_args()
    cuda = torch.cuda.is_available()  
    print(f'unifor:{args.unifor}')
    print(f'noshuffle:{args.noshuffle}')
    print(f'transformer:{args.transformer}')
    print(f"rd_uni:{args.rd_uni}")

    if args.dataset_name == 'imagenet':
        if args.curriculum or args.train_dataset_root.endswith('.npy'):
            train_set = datasets.AdvTrainDataseti(root_dir=args.train_dataset_root)
        else:
           
            train_set = datasets.imagenet(args.train_dataset_root, mode="train")
        valid_set = datasets.imagenet(args.valid_dataset_root, mode="validation")
        args.x_size, args.y_size = (3, 224, 224), (3, 224, 224)
    elif args.dataset_name == 'cifar10':
        if args.curriculum:
            if args.unifor:
                if args.noshuffle or args.rd_uni:
                    print('加载随机数据集')
                    train_set = datasets.AdvTrainDatasetc(root_dir=args.train_dataset_root)  
                    valid_set = datasets.AdvTrainDatasetc(root_dir=args.train_dataset_root)  
                else:
                    print('加载均匀数据集')
                    train_set = datasets.uniformdata(root_dir=args.train_dataset_root)  
                    valid_set = datasets.uniformdata(root_dir=args.train_dataset_root)  
            else:
                train_set = datasets.AdvTrainDatasetc(root_dir=args.train_dataset_root)  
                valid_set = datasets.AdvTrainDatasetc(root_dir=args.train_dataset_root) 
        else:
          
            train_set = torchvision.datasets.CIFAR10(root=args.train_dataset_root, train=True, download=False,
                                                     transform=torchvision.transforms.ToTensor())  
            valid_set = torchvision.datasets.CIFAR10(root=args.valid_dataset_root, train=False, download=False,
                                                 transform=torchvision.transforms.ToTensor())  

        args.x_size, args.y_size = (3, 32, 32), (3, 32, 32)
    elif args.dataset_name == 'mnist':
        if args.curriculum:
            if args.unifor:
                print('加载均匀数据集')
                train_set = datasets.uniformdata(root_dir=args.train_dataset_root) 
                valid_set = datasets.uniformdata(root_dir=args.train_dataset_root)  
          
            else:
                train_set = datasets.AdvTrainDatasetm(root_dir=args.train_dataset_root)  
                valid_set = datasets.AdvTrainDatasetm(root_dir=args.train_dataset_root)  
        else:
            
            train_set = torchvision.datasets.MNIST(root=args.train_dataset_root, train=True, download=False,
                                                     transform=torchvision.transforms.ToTensor()) 
            valid_set = torchvision.datasets.MNIST(root=args.valid_dataset_root, train=False, download=False,
                                                 transform=torchvision.transforms.ToTensor())  

        args.x_size, args.y_size = (1, 32, 32), (1, 32, 32)
    else:
        raise NotImplementedError

    generator = Generator(args).cuda()  
    print('Data saved file in: ', args.log_root, args.name)
    print("number of param: {}".format(count_parameters(generator)))

    
    optim = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.regularizer)  


    if args.model_path != "":
        print('Loading pre-trained model from path: ', args.model_path)
        state = torch.load(args.model_path, map_location='cuda')  
        optim.load_state_dict(state["optim"])  
        generator.load_state_dict(state["model"])
        del state 

    if args.adv_loss:
       
        support_set_models, query_set_models = [], []
        if args.dataset_name == 'imagenet':
            load_model = load_imagenet_model
        elif args.dataset_name == 'cifar10':
            load_model = load_cifar_model
        elif args.dataset_name == 'mnist':
            load_model = load_mnist_model
        else:
            raise NotImplementedError
        if args.dataset_name == 'cifar10':
            for model_name in args.support_set.split(","):
                support_set_models.append(load_cifar_model(model_name))
            for model_name in args.query_set.split(","):
                query_set_models.append(load_cifar_model(model_name))
        if args.dataset_name == 'mnist':
            for model_name in args.support_set.split(","):
                support_set_models.append(load_mnist_model(model_name))
            for model_name in args.query_set.split(","):
                query_set_models.append(load_mnist_model(model_name))

        trainer = mt.Trainer(generator, support_set_models, query_set_models, optim, train_set, valid_set,
                                      args, cuda)
    else:
        trainer = Learner.Trainer(generator, optim, train_set, valid_set, args, cuda)  

    trainer.train()
