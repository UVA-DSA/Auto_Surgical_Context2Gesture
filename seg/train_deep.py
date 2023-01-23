from genericpath import exists
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
import os
import numpy as np
import tqdm
import datetime
import math
from Deeplabv3_rob import Deeplabv3_rob
from validation import val_multi
import glob
from load_dataset import Load_Dataset
from tensorboardX import SummaryWriter
import time
import datetime
#from focalloss import FocalLoss
import torch.nn as nn
import pdb
from collections import deque

device_ids = [0]

parse=argparse.ArgumentParser()
num_classes=1

task = ['Suturing' ,'Needle_Passing','Knot_Tying'] # 'ring' with Needle_Passing, grasper with: 'Suturing' ,'Needle_Passing','Knot_Tying',thread trained with Suturing and Knot Tying; needle trained with Suturing and Needle
lra=0.00001


# train with all data using miccai

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lra* (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_filename(task,objects):
    from itertools import chain
    
    def find_dir(ta):
        mask_dir ='/home/student/Documents/Zoey/segmentation/JIGSAWS/{}/mask_output_{}'.format(ta,objects)
        train_file=glob.glob(mask_dir+'/*')
        if ta =='Suturing':   
            # those have half the resolution
            dir_half = np.array(['Suturing_S07_T01', 'Suturing_S07_T02', 'Suturing_S07_T03','Suturing_S07_T04', 'Suturing_S07_T05', 'Suturing_S08_T01'])
            # those the test set
            dir_test = np.array([ 'Suturing_S02_T04','Suturing_S02_T01', 'Suturing_S03_T04','Suturing_S03_T05','Suturing_S05_T03'])
            dir_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_half)-set(dir_test))

        elif ta=='Needle_Passing':
            dir_half = np.array(['Needle_Passing_S06_T01','Needle_Passing_S06_T03','Needle_Passing_S06_T04','Needle_Passing_S08_T02','Needle_Passing_S08_T04','Needle_Passing_S08_T05','Needle_Passing_S09_T03'])
            dir_test = np.array(['Needle_Passing_S08_T02','Needle_Passing_S04_T01','Needle_Passing_S05_T03','Needle_Passing_S05_T05'])
            dir_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_test)-set(dir_half))
        else:
            dir_half = np.array(['Knot_Tying_S06_T04','Knot_Tying_S06_T05','Knot_Tying_S07_T01','Knot_Tying_S07_T02','Knot_Tying_S07_T03','Knot_Tying_S07_T04','Knot_Tying_S07_T05'])
            dir_test = np.array(['Knot_Tying_S09_T05','Knot_Tying_S05_T05','Knot_Tying_S03_T05','Knot_Tying_S05_T03','Knot_Tying_S03_T02'])
            dir_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_test)-set(dir_half))
        train_file_names =list(chain.from_iterable(list(map(lambda x:glob.glob(mask_dir+'/'+x+'/*'), dir_train))))
        val_file_names = list(chain.from_iterable(list(map(lambda x:glob.glob(mask_dir+'/'+x+'/*'),dir_test))))
        return train_file_names, val_file_names
    if not type(task) is list:
        tra, val = find_dir(task)
        return tra,val
    else:
        total_dir_test = []
        total_dir_train = []
        for ta in task:
            tra,val=find_dir(ta)
            if len(tra)==0:continue
            total_dir_train.extend(tra)
            total_dir_test.extend(val)
        return total_dir_train, total_dir_test
        

def train(data):
    model = Deeplabv3_rob(num_classes=num_classes,pretrained=True)

    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
    batch_size = args.batch_size
    if 'miccai' not in data:
        train_file, val_file = load_filename(task,objects)
        liver_dataset = Load_Dataset(train_file,objects)
        val_dataset= Load_Dataset(val_file,objects)

        dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=5) # drop_last=True
        val_load=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    else:
        if data =='miccai':
            train_file, val_file = load_filename_miccai(objects)
        elif data=='miccai22':
            train_file, val_file = load_filename_miccai22(objects)
            #breakpoint()
        liver_dataset = Load_Dataset(train_file,objects)
        dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=5) # drop_last=True
        val_load = None
    #breakpoint()
   
  
    #criterion = FocalLoss(gamma=6) #500 for needle, thread class
    criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(250.).cuda(device_ids[0]))#torch.nn.MSELoss(reduction='mean')#torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lra)

    
    train_model(model, criterion, optimizer, dataloaders, val_load, num_classes,num_epochs=num_epoch)

def train_model(model, criterion, optimizer, dataload,val_load,num_classes,num_epochs=None):
    loss_list=[]
    dice_list=[]
    logs_dir = '/home/student/Documents/Zoey/segmentation/Deeplabv3_Resnet_Robo-binary/Logs_'+objects+'_'+data+'/T{}'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    logs_dir = logs_dir+'_e'+str(num_epochs) +'/'
    os.makedirs(logs_dir,exist_ok=True)
    # cur_l = []
    # writer = SummaryWriter(logs_dir)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size/args.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_loss =[]
        step = 0
        for _, x, y in dataload:
            if x.shape[0]!=args.batch_size:continue
            step += 1
            inputs = x.cuda(device_ids[0])
            y=y.long()
            labels = y.cuda(device_ids[0])
            optimizer.zero_grad()
            outputs = model(inputs)
            #breakpoint()
            loss = criterion(outputs['out'].float(), labels.float())
            loss.backward()
            optimizer.step()
            tq.update(1)
            epoch_loss.append(loss.item())
            epoch_loss_mean = np.mean(epoch_loss).astype(np.float64)
            tq.set_postfix(loss='{0:.3f}'.format(epoch_loss_mean))
        loss_list.append(epoch_loss_mean)
        tq.close()
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss_mean))
        torch.save(model.module.state_dict(), logs_dir + 'weights_{}.pth'.format(epoch))
        #dice, iou =val_multi(model, criterion, val_load, num_classes,args.batch_size,device_ids,objects)
        # writer.add_scalar('Loss', epoch_loss_mean, epoch)
        # writer.add_scalar('Dice', dice, epoch)
        # writer.add_scalar('IoU', iou, epoch)
        # dice_list.append([dice,iou])
        #adjust_learning_rate(optimizer, epoch)
        
        fileObject = open(logs_dir+'LossList.txt', 'w')
        for ip in loss_list:
            fileObject.write(str(ip))
            fileObject.write('\n')
        fileObject.close()
        
        # cur_l.append(epoch_loss_mean)
        # if len(cur_l)>5:
        #     cur_l.popleft()
        # if np.mean(cur_l)<0.16:
        #     print(f'End epoch {epoch}')
        #     break

        # fileObject = open(logs_dir + 'dice_list.txt', 'w')
        # for ip in dice_list:
        #     fileObject.write(str(ip))
        #     fileObject.write('\n')
        # fileObject.close()

    # writer.close()
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--dataset",  default='JIGSAWS')
    parse.add_argument("--objects", default='grasper')
    parse.add_argument("--epochs", type=int, default=10)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    objects = args.objects #'thread'
    data = args.dataset#'miccai'
    num_epoch = args.epochs#1#00
    print(data,objects,num_epoch)
    start_t = time.time()
    train(data)
    exe_t = time.time()
    total_t = exe_t-start_t
    print(f'total time: '+str(datetime.timedelta(seconds=total_t)))
