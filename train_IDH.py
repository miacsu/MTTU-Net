# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,IDH_network,Grade_netwoek
from models.unet import UNet3D
import torch.distributed as dist
from models import criterions
from contextlib import nullcontext
from data.BraTS_IDH import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.pcgrad import PCGrad
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from models.criterions import MultiTaskLossWrapper,FocalLoss_seg
from sklearn.metrics import roc_auc_score,accuracy_score
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='jhcheng', type=str)

parser.add_argument('--experiment', default='TransBraTS_IDH', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBraTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/public/home/dataset/BraTS_2020/', type=str)

parser.add_argument('--train_dir', default='MICCAI_BraTS2020_TrainingData', type=str)

parser.add_argument('--valid_dir', default='MICCAI_BraTS2020_TrainingData', type=str)

parser.add_argument('--test_dir', default='MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='IDH_train_1.txt', type=str)#IDH_all.txt

parser.add_argument('--valid_file', default='IDH_valid_1.txt', type=str) #IDH_test.txt

parser.add_argument('--test_file', default='IDH_test.txt', type=str)

parser.add_argument('--dataset', default='brats_IDH', type=str)

parser.add_argument('--model_name', default='TransBraTS', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=155, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1234, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=1000, type=int)

parser.add_argument('--save_freq', default=750, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank()
    torch.manual_seed(args.seed+rank)
    torch.cuda.manual_seed(args.seed+rank)
    random.seed(args.seed+rank)
    np.random.seed(args.seed+rank)

    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    seg_model = Decoder_modual()
    IDH_model = IDH_network()
   
    criterion = getattr(criterions, args.criterion)#args.criterion
    idh_criterion = getattr(criterions,'idh_cross_entropy')#idh_focal_loss, idh_cross_entropy
    #criterion = FocalLoss_seg()
    MTL = MultiTaskLossWrapper(2,loss_fn=[criterion,idh_criterion])

    nets = {
        'en': torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(args.local_rank),
        'seg': torch.nn.SyncBatchNorm.convert_sync_batchnorm(seg_model).cuda(args.local_rank),
        'idh': torch.nn.SyncBatchNorm.convert_sync_batchnorm(IDH_model).cuda(args.local_rank),
        'mtl':MTL.cuda(args.local_rank)
    }
    param = [p for v in nets.values() for p in list(v.parameters())]

    DDP_model = {
        'en': nn.parallel.DistributedDataParallel(nets['en'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'seg': nn.parallel.DistributedDataParallel(nets['seg'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'idh': nn.parallel.DistributedDataParallel(nets['idh'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True),
        'mtl':nn.parallel.DistributedDataParallel(nets['mtl'], device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    }

    # DDP_model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
    #                                             find_unused_parameters=True)

    optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # optimizer = PCGrad(torch.optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad))

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        writer = SummaryWriter()

    resume = ''

    #writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        DDP_model['en'].load_state_dict(checkpoint['en_state_dict'])
        DDP_model['seg'].load_state_dict(checkpoint['seg_state_dict'])
        DDP_model['idh'].load_state_dict(checkpoint['idh_state_dict'])
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, args.mode)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu)+1) // 2

    train_loader = DataLoader(dataset= train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info('Samples for valid = {}'.format(len(valid_set)))

    start_time = time.time()
    torch.set_grad_enabled(True)

    best_epoch = 0
    min_loss = 100.0

    for epoch in range(args.start_epoch, args.end_epoch):
        DDP_model['en'].train()
        DDP_model['seg'].train()
        DDP_model['idh'].train()
        DDP_model['mtl'].train()
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        epoch_train_loss = 0.0
        epoch_train_seg_loss = 0.0
        epoch_train_idh_loss = 0.0

        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            # optimizer.adjust_learning_rate(epoch, args.end_epoch, args.lr)

            optimizer.zero_grad()
            x, target,idh = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)
            weight = torch.tensor([57, 91]).float().cuda(args.local_rank, non_blocking=True)

            x1_1, x2_1, x3_1,x4_1, encoder_output = DDP_model['en'](x)
            y = DDP_model['seg'](x1_1, x2_1, x3_1,encoder_output)
            idh_out = DDP_model['idh'](x4_1, encoder_output)

            loss,seg_loss,idh_loss,loss1,loss2,loss3,seg_std, idh_std,log_var_1,log_var_2= DDP_model['mtl']([y,idh_out],[target,idh],[None,weight])

            reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
            seg_std = all_reduce_tensor(seg_std, world_size=num_gpu).data.cpu().numpy()
            idh_std = all_reduce_tensor(idh_std, world_size=num_gpu).data.cpu().numpy()
            seg_vars = all_reduce_tensor(log_var_1, world_size=num_gpu).data.cpu().numpy()
            idh_vars = all_reduce_tensor(log_var_2, world_size=num_gpu).data.cpu().numpy()

            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            reduce_seg_loss = all_reduce_tensor(seg_loss, world_size=num_gpu).data.cpu().numpy()
            reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
            reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
            reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

            epoch_train_loss += reduce_loss/len(train_loader)
            epoch_train_seg_loss += reduce_seg_loss/len(train_loader)
            epoch_train_idh_loss += reduce_idh_loss/len(train_loader)

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f} seg_loss: {:.5f} idh_loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||  seg_std:{:.4f} idh_std:{:.4f} seg_vars:{:.4f} idh_vars:{:.4f}'
                             .format(epoch, i, reduce_loss,reduce_seg_loss,reduce_idh_loss, reduce_loss1, reduce_loss2, reduce_loss3,seg_std,idh_std,seg_vars,idh_vars))
            loss.backward()
            optimizer.step()

        idh_probs = []
        idh_class = []
        idh_target = []
        with torch.no_grad():
            DDP_model['en'].eval()
            DDP_model['seg'].eval()
            DDP_model['idh'].eval()
            DDP_model['mtl'].eval()
            epoch_valid_loss = 0.0
            epoch_seg_loss = 0.0
            epoch_idh_loss = 0.0
            epoch_dice_1 = 0.0
            epoch_dice_2 = 0.0
            epoch_dice_3 = 0.0
            for i, data in enumerate(valid_loader):
                # [t.cuda(args.local_rank, non_blocking=True) for t in data]
                x,target,idh = data

                x = x.cuda(args.local_rank, non_blocking=True)
                target = target.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)

                encoder_outs = DDP_model['en'](x)
                seg_out = DDP_model['seg'](encoder_outs[0],encoder_outs[1],encoder_outs[2],encoder_outs[4])
                idh_out = DDP_model['idh'](encoder_outs[3], encoder_outs[4])

                valid_loss, seg_loss, idh_loss, loss1, loss2, loss3,std_1,std_2,var_1, var_2 = DDP_model['mtl']([seg_out, idh_out], [target, idh],
                                                                                  [None, None])
                epoch_valid_loss += valid_loss / len(valid_loader)

                epoch_seg_loss += seg_loss / len(valid_loader)
                epoch_idh_loss += idh_loss / len(valid_loader)

                epoch_dice_1 += loss1 / len(valid_loader)
                epoch_dice_2 += loss2 / len(valid_loader)
                epoch_dice_3 += loss3 / len(valid_loader)

                idh_pred = F.softmax(idh_out, 1)
                # idh_pred = idh_out.sigmoid()
                idh_pred_class = torch.argmax(idh_pred, dim=1)
                # idh_pred_class = (idh_pred > 0.5).float()
                idh_probs.append(idh_pred[0][1].cpu())
                # idh_probs.append(idh_pred[0])
                idh_class.append(idh_pred_class.item())
                idh_target.append(idh.item())

            accuracy = accuracy_score(idh_target,idh_class)
            auc = roc_auc_score(idh_target,idh_probs)

            if args.local_rank == 0:

                if min_loss >= epoch_valid_loss:
                    min_loss = epoch_valid_loss
                    best_epoch = epoch
                    logging.info('there is an improvement that update the metrics and save the best model.')

                    file_name = os.path.join(checkpoint_dir, 'model_epoch_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'en_state_dict': DDP_model['en'].state_dict(),
                        'seg_state_dict': DDP_model['seg'].state_dict(),
                        'idh_state_dict': DDP_model['idh'].state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    },
                        file_name)
                logging.info('Epoch:{}[best_epoch:{} min_total_loss:{:.5f} ||epoch_valid_loss:{:.5f} | seg_loss: {:.5f} | idh_loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} | idh_acc: {:.5f} | idh_auc:{:.5f}||'
                             .format(epoch, best_epoch,min_loss,epoch_valid_loss, epoch_seg_loss,epoch_idh_loss,epoch_dice_1,epoch_dice_2,epoch_dice_3,accuracy, auc))

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'en_state_dict': DDP_model['en'].state_dict(),
                    'seg_state_dict': DDP_model['seg'].state_dict(),
                    'idh_state_dict': DDP_model['idh'].state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('epoch_train_loss:',epoch_train_loss,epoch)
            writer.add_scalar('epoch_valid_loss:', epoch_valid_loss, epoch)
            writer.add_scalar('seg_loss:', epoch_train_seg_loss, epoch)
            writer.add_scalar('idh_loss:', epoch_train_idh_loss, epoch)
            writer.add_scalar('valid_seg_loss:',epoch_seg_loss,epoch)
            writer.add_scalar('valid_idh_loss:', epoch_idh_loss, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'en_state_dict': DDP_model['en'].state_dict(),
            'seg_state_dict': DDP_model['seg'].state_dict(),
            'idh_state_dict': DDP_model['idh'].state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()







