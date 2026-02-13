import argparse
import os
import time
import numpy as np
import pandas as pd

import logging

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

#from data_utils.dataloader import Molar3D
from data_utils.spine_dataloader import VertebraePOI
import data_utils.transforms as tr
from utils import setgpu, metric, metric_proj
from data_utils.transforms import LandMarkToGaussianHeatMap
from models.losses import HNM_heatmap
from models.BiFormer_Unet import BiFormer_Unet
from models.UNet import UNet3D


# super parameters settings here
parser = argparse.ArgumentParser(description='PyTorch landmarking baselin heatmap regression')
# the network backbone settings
parser.add_argument('--model_name',metavar='MODEL',default='BiFormer_Unet',type=str, choices=['VNet', 'UNet3D', 'ResidualUNet3D','BiFormer_Unet'])
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
parser.add_argument('--save_dir',default='./SavePath/gruber-fixed_size',type=str)
# the settings of gpus, multiGPU can use '0,1' or '0,1,2,3'
parser.add_argument('--gpu', default='0', type=str)
# the early stop parameter
parser.add_argument('--patient',default=20,type=int)
# the loss HNM_heatmap for baseline heatmap regression, HNM_propmap for yolol
parser.add_argument('--loss_name', default='HNM_heatmap',type=str)
# the path of dataset
# before training please download the dataset and put it in "../mmld_dataset"
parser.add_argument('--data_path',
                    default='./gruber_dataset_cutouts',
                    type=str,
                    metavar='N',
                    help='data path')
# the classes
parser.add_argument('--n_class',default=35,type=int, help='number of landmarks 35')
# the radius of gaussian heatmap's mask
parser.add_argument('-R','--focus_radius', default=20,type=int)
# the test flag | -1 for train, 0 for eval, 1 for test |
parser.add_argument('--test_flag',default=-1,type=int, choices=[-1, 0, 1])
parser.add_argument('--master_df_path', default='./gruber_dataset_cutouts/cutouts/master_df.csv',type=str)

parser.add_argument('--input_shape', default=[215, 215, 144], type=int, nargs=3,
                    help='Input shape as three integers (H W D)')
parser.add_argument('--project_gt', action='store_true', help='project GT to surface')

parser.add_argument('--project_pred', action='store_true', help='project predictions to surface')




DEVICE = torch.device("cuda" if True else "cpu")
def main(args):
    cudnn.benchmark = True
    setgpu(args.gpu)
    ########################### model init #############################################
    net = globals()[args.model_name](n_class=args.n_class)
    loss = globals()[args.loss_name](R=args.focus_radius)

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
    # the lr decayed with rate 0.98 each epoch
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)


    master_df = pd.read_csv(args.master_df_path)
    ########################## network testing ########################################
    # if the test_flag > -1, calculate the MRE and SDR (%) for val and test set
    if args.test_flag > -1:
        args.batch_size = 1

        if args.test_flag == 0:
            test_transform = transforms.Compose([
                tr.RandomCrop(size=[128,128,64]), # 
                #tr.Normalize(),
                tr.ToTensor(),
                ])
            phase = 'val'
        else:
            test_transform = transforms.Compose([
                tr.CenterCrop(size=[128,128,64]), # center crop for validation
                #tr.Normalize(),
                tr.ToTensor(),
                ])
            phase = 'test'
        #test_dataset = Molar3D(transform=test_transform,
        #              phase=phase,
        #              parent_path=args.data_path)

        test_dataset = VertebraePOI(
            master_df=master_df,
            transform=test_transform,
            phase=phase,
            input_shape=args.input_shape,
            project_gt=args.project_gt
        )

        testloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4)
        
        #os.makedirs("./hz_baseline_gruber_all/", exist_ok=True) # added by alissa
        os.makedirs(os.path.join(args.save_dir, "hz_baseline_gruber_all"), exist_ok=True)

        #test(testloader, net)

        if args.project_pred:
            test_proj(testloader, net, args)
        else:
            test(testloader, net, args)
        return
        

    # generate Gaussian Heatmap using pytorch GPU tensor    
    l2h = LandMarkToGaussianHeatMap(R=args.focus_radius, 
                                    n_class=args.n_class,
                                    GPU=DEVICE, 
                                    img_size=(128,128,64))
                                    # img_size=(128,128,64))

    ########################## data preparation ########################################
    # if the test_flag <= -1, begin network training
    # train set and validation set preprocessing
    train_transform = transforms.Compose([
        tr.RandomCrop(size=[128,128,64]), # zoom and random crop for data augumentation
        #tr.Normalize(),
        tr.ToTensor(),
    ])
    #train_dataset = Molar3D(transform=train_transform,
    #                  phase='train',
    #                  parent_path=args.data_path)


    train_dataset = VertebraePOI(
        master_df=master_df,
        transform=train_transform,
        phase='train',
        input_shape=args.input_shape,
        project_gt=args.project_gt
    )

    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8)

    eval_transform = transforms.Compose([
        tr.CenterCrop(size=[128,128,64]), # center crop for validation
        #tr.Normalize(),
        tr.ToTensor(),
    ])
    #eval_dataset = Molar3D(transform=eval_transform,
    #                  phase='val',
    #                  parent_path=args.data_path)

    eval_dataset = VertebraePOI(
        master_df=master_df,
        transform=eval_transform,
        phase='val',
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
        train(trainloader, net, loss, epoch, optimizer, l2h)
        if optimizer.param_groups[0]['lr'] > args.lr * 0.03:
                scheduler.step()

        # validation in one epoch
        break_flag += 1
        eval_loss = evaluation(evalloader, net, loss, epoch, l2h)
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
                }, os.path.join(save_dir, 'model.ckpt'))
            logging.info(
                '************************ model saved successful ************************** !\n'
            )

        if break_flag >args.patient:
            break


def train(data_loader, net, loss, epoch, optimizer, l2h):
    start_time = time.time()
    net.train()
    total_train_loss = []
    for i, sample in enumerate(data_loader):
        data = sample['image']
        landmark = sample['landmarks']
        heatmap_batch = l2h(landmark)
        data = data.to(DEVICE)
        # print(data.size(),"train-----------------------") #torch.Size([4, 1, 128, 128, 64])
        heatmap = net(data)
        optimizer.zero_grad()
        cur_loss = loss(heatmap, heatmap_batch)
        total_train_loss.append(cur_loss.item())
        cur_loss.backward()
        optimizer.step()

    logging.info(
        'Train--Epoch[%d], lr[%.6f], total loss: [%.6f], time: %.1f s!'
        % (epoch, optimizer.param_groups[0]['lr'], np.mean(total_train_loss), time.time() - start_time))
    

def evaluation(dataloader, net, loss, epoch, l2h):
    start_time = time.time()
    net.eval()
    total_loss = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            landmark = sample['landmarks']
            heatmap_batch = l2h(landmark)
            data = data.to(DEVICE)
            # print(data.size(),"eval----------") 
            heatmap= net(data)
            cur_loss = loss(heatmap, heatmap_batch)
            total_loss.append(cur_loss.item())
            
    logging.info(
        'Eval--Epoch[%d], total loss: [%.6f],  time: %.1f s!'
        % (epoch, np.mean(total_loss),  time.time() - start_time))
    logging.info(
        '***************************************************************************'
    )
    return np.mean(total_loss)


def test(dataloader, net):
    start_time = time.time()
    net.eval()
    total_mre = []
    total_mean_mre = []
    N = 0
    total_hits = np.zeros((8, 14))
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            landmarks = sample['landmarks']
            spacing = sample['spacing']
            data = data.to(DEVICE)
            # print(data.size(),"test----------") #torch.Size([1, 1, 128, 128, 64]) test---------- torch.Size([1, 1, 512, 512, 256]) test----------
            heatmap = net(data)
            
            mre, hits = metric(heatmap.cpu().numpy(), 
                     spacing.numpy(), 
                     landmarks.cpu().numpy())
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

    
    ################################ molar print##############################################
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

    ########################### total mre ######################################################
    mre = np.stack(mre, 0)
    sd = np.stack(sd, 0)
    total = np.stack([mre, sd], 0)
    total = np.concatenate([total, cur_hits], 0)
    for i, name in enumerate(names):
        form[name] = total[:, i]
    df = pd.DataFrame(form, columns = form.keys())
    df.to_excel( 'baseline_gruber_test.xlsx', index = False, header=True)

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

def test_proj(dataloader, net, args):
    start_time = time.time()
    net.eval()
    total_mre = []
    total_mean_mre = []
    
    total_mre_projected = []
    total_mean_mre_projected = []
    
    total_mse = []
    total_mean_mse = []
    total_mse_projected = []
    total_mean_mse_projected = []

    N = 0
    total_hits = np.zeros((8, args.n_class))
    total_hits_projected = np.zeros((8, args.n_class))
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data = sample['image']
            landmarks = sample['landmarks']
            surface = sample.get('surface', None) 
            spacing = sample['spacing']
            data = data.to(DEVICE)
            
            heatmap = net(data)
            
            # Call metric_proj (with surface)
            if surface is not None:
                results = metric_proj(
                    heatmap.cpu().numpy(), 
                    spacing.numpy(), 
                    landmarks.cpu().numpy(),
                    surface_masks=surface.numpy()
                )

                mre, hits = results['mre']
                mse, _ = results['mse']
                mre_proj, hits_proj = results['mre_projected']
                mse_proj, _ = results['mse_projected']

                total_mre_projected.append(mre_proj)
                total_mse_projected.append(mse_proj)
                total_hits_projected += hits_proj
                
                
                # Current projected MRE
                cur_mre_proj = []
                for cdx in range(len(mre_proj[0])):
                    if mre_proj[0][cdx] > 0:
                        cur_mre_proj.append(mre_proj[0][cdx])
                total_mean_mre_projected.append(np.mean(cur_mre_proj))
                print("#: No.", i, "--the current projected MRE is [%.4f]" % np.mean(cur_mre_proj))
                
                cur_mse_proj = []
                for cdx in range(len(mse_proj[0])):
                    if mse_proj[0][cdx] > 0:
                        cur_mse_proj.append(mse_proj[0][cdx])
                total_mean_mse_projected.append(np.mean(cur_mse_proj))
                print("#: No.", i, "--the current projected MSE is [%.4f]" % np.mean(cur_mse_proj))


            else:
                mre, hits = metric(
                    heatmap.cpu().numpy(), 
                    spacing.numpy(), 
                    landmarks.cpu().numpy()
                )
            
            total_hits += hits
            total_mre.append(np.array(mre))
            total_mse.append(np.array(mse))
            N += data.shape[0]
            
            # Current MRE
            cur_mre = []
            for cdx in range(len(mre[0])):
                if mre[0][cdx] > 0:
                    cur_mre.append(mre[0][cdx])
            total_mean_mre.append(np.mean(cur_mre))
            print("#: No.", i, "--the current MRE is [%.4f]" % np.mean(cur_mre))
    
            cur_mse = []
            for cdx in range(len(mse[0])):
                if mse[0][cdx] > 0:
                    cur_mse.append(mse[0][cdx])
            total_mean_mse.append(np.mean(cur_mse))
            print("#: No.", i, "--the current MSE is [%.4f]" % np.mean(cur_mse))
    
    total_mre = np.concatenate(total_mre, 0)

    total_mse = np.concatenate(total_mse, 0)

    
    # Concatenate projected MRE
    if len(total_mre_projected) > 0:
        total_mre_projected = np.concatenate(total_mre_projected, 0)
        total_mse_projected = np.concatenate(total_mse_projected, 0)
    
    ################################ molar print ##############################################
    names = [
        '81', '82', '83', '84', '85', '86', '87', '88', '89',
        '101', '102', '103', '104', '105', '106', '107', '108',
        '109', '110', '111', '112', '113', '114', '115', '116',
        '117', '118', '119', '120', '121', '122', '123', '124',
        '125', '127'
    ]

    IDs = ["MRE", "SD", "RMSE", "2.0", "2.5", "3.0", "4."]
    form = {"metric": IDs}
    mre = []
    sd = []
    rmse = []
    
    mre_proj = []
    sd_proj = []
    rmse_proj = []
    
    cur_hits = total_hits[:4] / total_hits[4:]
    
    # Projected hits
    if len(total_mre_projected) > 0:
        cur_hits_projected = total_hits_projected[:4] / total_hits_projected[4:]

    ############################## each class mre ##############################################
    for i, name in enumerate(names):
        # Projected metrics
        if len(total_mre_projected) > 0:
            cur_mre_proj = []
            for j in range(total_mre_projected.shape[0]):
                if total_mre_projected[j, i] > 0:
                    cur_mre_proj.append(total_mre_projected[j, i])
            
            cur_mse_proj = []
            for j in range(total_mse_projected.shape[0]):
                if total_mse_projected[j, i] > 0:
                    cur_mse_proj.append(total_mse_projected[j, i])            
            
            cur_mre_proj = np.array(cur_mre_proj)
            cur_mse_proj = np.array(cur_mse_proj)


            mre_proj.append(np.mean(cur_mre_proj))
            sd_proj.append(np.sqrt(np.sum(pow(np.array(cur_mre_proj) - np.mean(cur_mre_proj), 2)) / (N-1)))
            rmse_proj.append(np.sqrt(np.mean(cur_mse_proj))) # cur_mse_proj is already squared

        # Original MRE
        cur_mre = []
        for j in range(total_mre.shape[0]):
            if total_mre[j, i] > 0:
                cur_mre.append(total_mre[j, i])

        cur_mse = []
        for j in range(total_mse.shape[0]):
            if total_mse[j, i] > 0:
                cur_mse.append(total_mse[j, i])

        cur_mre = np.array(cur_mre)
        cur_mse = np.array(cur_mse)

        mre.append(np.mean(cur_mre))
        rmse.append(np.sqrt(np.mean(cur_mse)))
        sd.append(np.sqrt(np.sum(pow(np.array(cur_mre) - np.mean(cur_mre), 2)) / (N-1)))

    ########################### Original results ######################################################
    mre = np.stack(mre, 0)
    sd = np.stack(sd, 0)
    rmse = np.stack(rmse, 0)
    
    total = np.stack([mre, sd, rmse ], 0)
    total = np.concatenate([total, cur_hits], 0)
    
    for i, name in enumerate(names):
        form[name] = total[:, i]
    
    df = pd.DataFrame(form, columns=form.keys())
    df.to_excel('baseline_gruber_test.xlsx', index=False, header=True)

    # Projected results
    if len(total_mre_projected) > 0:
        mre_proj = np.stack(mre_proj, 0)
        sd_proj = np.stack(sd_proj, 0)
        rmse_proj = np.stack(rmse_proj, 0)
        total_proj = np.stack([mre_proj, sd_proj, rmse_proj], 0)
        total_proj = np.concatenate([total_proj, cur_hits_projected], 0)
        
        form_proj = {"metric": IDs}
        for i, name in enumerate(names):
            form_proj[name] = total_proj[:, i]
        
        df_proj = pd.DataFrame(form_proj, columns=form_proj.keys())
        df_proj.to_excel('baseline_gruber_test_projected.xlsx', index=False, header=True)

    ########################### total mre ######################################################
    mmre = np.mean(total_mean_mre)
    sd = np.sqrt(np.sum(pow(np.array(total_mean_mre) - mmre, 2)) / (N-1))
    mrmse = np.sqrt(np.mean(total_mean_mse))

    total_hits_sum = np.sum(total_hits, 1)
    logging.info(
        'Test-- MRE: [%.2f] + SD: [%.2f], RMSE: [%.2f], 2.0 mm: [%.4f], 2.5 mm: [%.4f], 3.0 mm: [%.4f], 4.0 mm: [%.4f], using time: %.1f s!' % (
            mmre, sd, mrmse,
            total_hits_sum[0] / total_hits_sum[4],
            total_hits_sum[1] / total_hits_sum[5],
            total_hits_sum[2] / total_hits_sum[6],
            total_hits_sum[3] / total_hits_sum[7],
            time.time()-start_time))
    logging.info('*' * 79)

    # Projected summary
    if len(total_mre_projected) > 0:
        mmre_proj = np.mean(total_mean_mre_projected)
        sd_proj = np.sqrt(np.sum(pow(np.array(total_mean_mre_projected) - mmre_proj, 2)) / (N-1))
        mrmse_proj = np.sqrt(np.mean(total_mean_mse_projected))
        
        total_hits_projected_sum = np.sum(total_hits_projected, 1)
        logging.info(
            'Test projected-- MRE: [%.2f] + SD: [%.2f], RMSE: [%.2f], 2.0 mm: [%.4f], 2.5 mm: [%.4f], 3.0 mm: [%.4f], 4.0 mm: [%.4f], using time: %.1f s!' % (
                mmre_proj, sd_proj, mrmse_proj,
                total_hits_projected_sum[0] / total_hits_projected_sum[4],
                total_hits_projected_sum[1] / total_hits_projected_sum[5],
                total_hits_projected_sum[2] / total_hits_projected_sum[6],
                total_hits_projected_sum[3] / total_hits_projected_sum[7],
                time.time()-start_time))
        logging.info('*' * 79)
    
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
    
    