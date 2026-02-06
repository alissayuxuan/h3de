
# package here
import argparse
import os
import time
import numpy as np
import logging
import pandas as pd

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

#from data_utils.dataloader import Molar3D
from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from utils import setgpu, metric_proposal
from models.losses import HNM_propmap
from models.PBiFormer_Unet import PBiFormer_Unet


# super parameters settings here
parser = argparse.ArgumentParser(description='PyTorch Robust Mandibular Molar Landmark Detection')
# the network backbone settings
parser.add_argument('--model_name',metavar='MODEL',default='PBiFormer_Unet',type=str, choices=['PVNet', 'PUNet3D', 'PResidualUNet3D', 'HYATTNet', 'SwinNet','PBiFormer_Unet'])
# the maximum training epochs 
parser.add_argument('--epochs',default=200,type=int,metavar='N')
# the beginning epoch
parser.add_argument('--start_epoch',default=1,type=int)
# the batch size, default 4 for one GPU
parser.add_argument('-b','--batch_size',default=4,type=int)
# the initial learning rate
parser.add_argument('--lr','--learning_rate',default=0.001,type=float)
# the path for loading pretrained model parameters
parser.add_argument('--resume',default='',type=str)
# the weight decay
parser.add_argument('--weight-decay','--wd',default=0.0005,type=float)
# the path to save the model parameters
parser.add_argument('--save_dir',default='./SavePath/gruber_yolol',type=str)
# the settings of gpus, multiGPU can use '0,1' or '0,1,2,3'
parser.add_argument('--gpu', default='0', type=str)
# the early stop parameter
parser.add_argument('--patient',default=20,type=int)
# the loss HNM_heatmap for baseline heatmap regression, HNM_propmap for yolol
parser.add_argument('--loss_name', default='HNM_propmap',type=str)
# the path of dataset
# before training please download the dataset and put it in "../mmld_dataset"
parser.add_argument('--data_path',
                    default='./gruber_dataset_cutouts',
                    type=str,
                    metavar='N',
                    help='data path')
# the classes
parser.add_argument('--n_class',default=35,type=int, help='number of landmarks 35')
# the downsample times
parser.add_argument('--shrink',default=4,type=int,metavar='shrink')
# the anchor balls default r=[0.5u, 0.75u, 1u, 1.25u]
parser.add_argument('--anchors',
                    default=[0.5, 0.75, 1., 1.25],
                    type=list,
                    metavar='anchors',
                    help='the anchor balls to predict')
# the test flag | -1 for train, 0 for eval, 1 for test |
parser.add_argument('--test_flag',default=-1,type=int, choices=[-1, 0, 1])
# the data type | full for dataset with complete landmarks | mini for mini dataset with uncomplete landmarks | all for default dataset
parser.add_argument('--data_type', default='all',type=str)
parser.add_argument('--master_df_path', default='./gruber_dataset_cutouts/cutouts/master_df.csv',type=str)
parser.add_argument('--input_shape', default=[215, 215, 128], type=int, nargs=3,
                    help='Input shape as three integers (H W D)')

parser.add_argument('--project_gt', action='store_true', help='project GT to surface')

DEVICE = torch.device("cuda" if True else "cpu")

def main(args):
    logging.info(args)
    cudnn.benchmark = True
    setgpu(args.gpu)

    ########################### model init #############################################
    net = globals()[args.model_name](n_class=args.n_class, n_anchor=len(args.anchors))
    loss = globals()[args.loss_name](n_class=args.n_class, device=DEVICE)

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    logging.info(args)
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])


    net = net.to(DEVICE)
    loss = loss.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net = DataParallel(net)

    # using Adam optimizer for network training
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.98),
                                 weight_decay=args.weight_decay)

    if args.resume and checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # the lr decayed with rate 0.98 each epoch
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

    master_df = pd.read_csv(args.master_df_path)

    ########################## network testing ########################################
    # if the test_flag > -1, calculate the MRE and SDR (%) for val and test set
    if args.test_flag > -1:
        args.batch_size = 1
        if args.test_flag == 0:
            test_transform = transforms.Compose([
                tr.CenterCrop(), # added by alissa
                tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors),
                #tr.Normalize(),
                tr.ToTensor(),
                ])
            phase = 'val'
        else:
            test_transform = transforms.Compose([
                tr.CenterCrop(), # center crop for validation
                tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors),
                #tr.Normalize(),
                tr.ToTensor(),
                ])
            phase = 'test'
        #test_dataset = Molar3D(transform=test_transform,
        #              phase=phase,
        #              parent_path=args.data_path,
        #              data_type=args.data_type)

        test_dataset = VertebraePOI(
            master_df=master_df,
            transform=test_transform,
            phase=phase,
            data_type=args.data_type,
            input_shape=args.input_shape,
            project_gt=args.project_gt
        )
        testloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)
        os.makedirs("./hz_yolol_gruber_all/", exist_ok=True) # added by alissa
        test(testloader, net, args)
        return


    ########################## data preparation ########################################
    # if the test_flag <= -1, begin network training
    # train set and validation set preprocessing
    train_transform = transforms.Compose([
        tr.RandomCrop(), # zoom and random crop for data augumentation
        tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors), # generate the anchor proposal
        #tr.Normalize(),
        tr.ToTensor(),
    ])
    #train_dataset = Molar3D(transform=train_transform,
    #                  phase='train',
    #                  parent_path=args.data_path,
    #                  data_type = args.data_type)

    #train_df = master_df[master_df['split'] == 'train'].iloc[:2]

    train_dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        data_type=args.data_type,
        input_shape=args.input_shape,
        project_gt=args.project_gt
    )

    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8)

    eval_transform = transforms.Compose([
        tr.CenterCrop(), # center crop for validation
        tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors),
        #tr.Normalize(),
        tr.ToTensor(),
    ])
    #eval_dataset = Molar3D(transform=eval_transform,
    #                  phase='val',
    #                  parent_path=args.data_path,
    #                  data_type=args.data_type)

    eval_dataset = VertebraePOI(
        master_df=master_df,
        transform=eval_transform,
        phase='val',
        data_type=args.data_type,
        input_shape=args.input_shape,
        project_gt=args.project_gt
    )

    evalloader = DataLoader(eval_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8)


    ########################## network training ##########################################
    # begin training here
    break_flag = 0. # counting for early stop
    low_loss = 100.
    total_loss = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        # train in one epoch
        train(trainloader, net, loss, epoch, optimizer)
        if optimizer.param_groups[0]['lr'] > args.lr * 0.03:
            scheduler.step()

        # validation in one epoch
        break_flag += 1
        eval_loss = evaluation(evalloader, net, loss, epoch)
        total_loss.append(eval_loss)
        if low_loss > eval_loss:
            low_loss = eval_loss
            break_flag = 0
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save(
                {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(save_dir, 'model_incomp_p.ckpt'))
            logging.info(
                '************************ model saved successful ************************** !\n'
            )
       
        if break_flag > args.patient:
            break
                

def train(data_loader, net, loss, epoch, optimizer):
    start_time = time.time()
    net.train()
    total_train_loss = []
    for i, sample in enumerate(data_loader):
        data = sample['image']
        proposals = sample['proposals']
        data = data.to(DEVICE)
        proposals = proposals.to(DEVICE)
        proposal_map = net(data)
        optimizer.zero_grad()
        cur_loss = loss(proposal_map, proposals)
        total_train_loss.append(cur_loss.item())
        cur_loss.backward()
        optimizer.step()
        
    logging.info(
        'Train--Epoch[%d], lr[%.6f], total loss: [%.6f], time: %.1f s!'
        % (epoch, optimizer.param_groups[0]['lr'], np.mean(total_train_loss), time.time() - start_time))
    
    print('PRINT: Train--Epoch[%d], lr[%.6f], total loss: [%.6f], time: %.1f s!'
        % (epoch, optimizer.param_groups[0]['lr'], np.mean(total_train_loss), time.time() - start_time))

def evaluation(dataloader, net, loss, epoch):
    start_time = time.time()
    net.eval()
    total_loss = []
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            proposals = sample['proposals']
            data = data.to(DEVICE)
            proposals = proposals.to(DEVICE)
            proposal_map = net(data)
            cur_loss = loss(proposal_map, proposals)
            total_loss.append(cur_loss.item())   
            
    logging.info(
        'Eval--Epoch[%d], total loss: [%.6f],  time: %.1f s!'
        % (epoch, np.mean(total_loss),  time.time() - start_time))

    print('PRINT: Eval--Epoch[%d], total loss: [%.6f],  time: %.1f s!'
        % (epoch, np.mean(total_loss),  time.time() - start_time))
    logging.info(
        '***************************************************************************'
    )
    return np.mean(total_loss)


def test(dataloader, net, args):
    start_time = time.time()
    net.eval()
    total_mre = []
    total_mean_mre = []
    N = 0
    total_hits = np.zeros((8, args.n_class))
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            landmarks = sample['landmarks']
            spacing = sample['spacing']
            filename = sample['filename']
            data = data.to(DEVICE)
            proposal_map = net(data)
            mre, hits = metric_proposal(proposal_map, spacing.numpy(),
                     landmarks.numpy(), "./hz_gruber_all/"+filename[0], shrink=args.shrink, anchors=args.anchors, 
                     n_class=args.n_class)            
            total_hits += hits
            total_mre.append(np.array(mre))
            N += data.shape[0]
            cur_mre = []
            for cdx in range(len(mre[0])):
                if mre[0][cdx]>0:
                    cur_mre.append(mre[0][cdx])
            total_mean_mre.append(np.mean(cur_mre))
            print("#: No.", i, "--the current MRE is [%.4f] "%np.mean(cur_mre))
    total_mre = np.concatenate(total_mre, 0)

    
    ################################# molar print ##############################################
    names = [
        '81', '82', '83', '84', '85', '86', '87', '88', '89',
        '101', '102', '103', '104', '105', '106', '107', '108',
        '109', '110', '111', '112', '113', '114', '115', '116',
        '117', '118', '119', '120', '121', '122', '123', '124',
        '125', '127'
        ]
    IDs = ["MRE", "SD", "2.0", "2.5", "3.0", "4."]
    form = {"metric": IDs}
    mre = []
    sd = []
    cur_hits = total_hits[:4] / total_hits[4:]

    ############################## each class mre ##############################################
    for i, name in enumerate(names):
        cur_mre = []
        for j in range(total_mre.shape[0]):
            if total_mre[j,i] > 0:
                cur_mre.append(total_mre[j,i])
        cur_mre = np.array(cur_mre)
        mre.append(np.mean(cur_mre))
        sd.append(np.sqrt(np.sum(pow(np.array(cur_mre) - np.mean(cur_mre), 2)) / (N-1)))
    
    mre = np.stack(mre, 0)
    sd = np.stack(sd, 0)
    total = np.stack([mre, sd], 0)
    
    total = np.concatenate([total, cur_hits], 0)
    for i, name in enumerate(names):
        form[name] = total[:, i]
    df = pd.DataFrame(form, columns = form.keys())
    # write each landmark MRE to xlsx file
    df.to_excel( 'yolol__gruber_test.xlsx', index = False, header=True)

    ########################### total mre ######################################################
    mmre = np.mean(total_mean_mre)
    sd = np.sqrt(np.sum(pow(np.array(total_mean_mre) - mmre, 2)) / (N-1))
    
    total_hits = np.sum(total_hits, 1)
    logging.info(
        'Test-- MRE: [%.2f] + SD: [%.2f], 2.0 mm: [%.4f], 2.5 mm: [%.4f], 3.0 mm: [%.4f], 4.0 mm: [%.4f], using time: %.1f s!' %(
            mmre, sd, 
            total_hits[0] / total_hits[4],
            total_hits[1] / total_hits[5],
            total_hits[2] / total_hits[6],
            total_hits[3] / total_hits[7],
            time.time()-start_time))
    logging.info(
        '***************************************************************************'
    )

if __name__ == '__main__':
    """
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s,%(lineno)d: %(message)s\n',
                        datefmt='%Y-%m-%d(%a)%H:%M:%S',
                        filename=os.path.join(args.save_dir, 'log.txt'),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    """


    import sys  # ← WICHTIG: sys importieren!
    
    global args
    args = parser.parse_args()
    
    # Create directories
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    log_file = os.path.join(args.save_dir, 'log.txt')
    
    # ==================== NEUES LOGGING SETUP ====================
    # 1. Root Logger aufräumen
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # 2. Formatter definieren
    formatter = logging.Formatter(
        fmt='%(asctime)s,%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d(%a)%H:%M:%S'
    )
    
    # 3. File Handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 4. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    # =============================================================
    
    # Info ausgeben
    print("\n" + "="*80)
    print("Training Configuration")
    print(f"  Log file: {log_file}")
    print(f"  Model: {args.model_name}")
    print(f"  GPU: {args.gpu}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print("="*80 + "\n")
    
    # Initial logging
    logging.info("="*80)
    logging.info("Training script started")
    logging.info(f"Configuration: {args}")
    logging.info("="*80)

    main(args)
    
    