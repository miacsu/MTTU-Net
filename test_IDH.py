import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS_IDH import BraTS
from predict import validate_softmax
from models.TransBraTS.TransBraTS_skipconnection import TransBraTS,Decoder_modual,IDH_network,Grade_netwoek


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--root', default='/public/home/dataset/BraTS_2020/', type=str)

parser.add_argument('--valid_dir', default='MICCAI_BraTS2020_ValidationData', type=str)

parser.add_argument('--valid_file', default='IDH_test.txt', type=str)

parser.add_argument('--output_dir', default='output', type=str)

parser.add_argument('--submission', default='submission', type=str)

parser.add_argument('--visual', default='visualization', type=str)

parser.add_argument('--experiment', default='TransBraTS_IDH', type=str) #TransBraTS

parser.add_argument('--test_date', default='2021-11-28', type=str)

parser.add_argument('--test_file', default='model_epoch_best.pth', type=str) #model_epoch_last

parser.add_argument('--use_TTA', default=True, type=bool)

parser.add_argument('--post_process', default=True, type=bool)

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--model_name', default='TransBraTS', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1', type=str)

parser.add_argument('--num_workers', default=4, type=int)

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    seg_model = Decoder_modual()
    IDH_model = IDH_network()
    Grade_model = Grade_netwoek()
    model = torch.nn.DataParallel(model).cuda()
    seg_model = torch.nn.DataParallel(seg_model).cuda()
    IDH_model = torch.nn.DataParallel(IDH_model).cuda()
    Grade_model =torch.nn.DataParallel(Grade_model).cuda()
    #only_seg=

    # dict_model = {'en':model,'seg':seg_model,'idh':IDH_model,'grade':Grade_model}
    dict_model = {'en': model, 'seg': seg_model,'idh':IDH_model}

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment+args.test_date, args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        dict_model['en'].load_state_dict(checkpoint['en_state_dict'])
        dict_model['seg'].load_state_dict(checkpoint['seg_state_dict'])
        # dict_model['grade'].load_state_dict(checkpoint['grade_state_dict'])
        dict_model['idh'].load_state_dict(checkpoint['idh_state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment+args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("valid_loader",valid_loader)


    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment+args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment+args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=dict_model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


