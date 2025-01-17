from config import get_config
from Learner import face_learner
import argparse

import random
import torch
import torch.backends.cudnn as cudnn
import os
from pathlib import Path

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    # Random seed, for deterministic behavior
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    # still need to set the work_init_fn to random.seed in train_dataloader, if multi numworkers

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode

    # 'softmax', 'normface', 'normface_alter-grad', 'arcface', 'arcface_alter-grad', 'arcface_origin', 'arcface_origin_detach-diff', 'arcface_adaptivemargin'
    ## -----local config----- ##
    conf.gpu_id = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_id

    conf.data_mode = 'casia'
    conf.head = 'arcface_adaptivemargin'  # 'softmax', 'normface', 'arcface'
    conf.margin = 0.6
    conf.m_mode = 'larger_sqrt'
    conf.detach_diff = True

    conf.work_path = Path('Experiments/Casia_Arcface-adaptivemargin_m0.6-larger-sqrt_detach-diff_B512_lr0.1/')
    conf.model_path = conf.work_path / 'models'
    conf.log_path = conf.work_path / 'log'
    conf.save_path = conf.work_path / 'save'

    conf.batch_size = 512
    conf.lr = 0.1
    conf.epochs = 35
    conf.milestones = [21, 30]
    ## -----local config----- ##

    learner = face_learner(conf)
    learner.train(conf, conf.epochs)