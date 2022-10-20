import torch
import argparse
from torch.utils.data import DataLoader
from load_dataset import Load_Dataset
from validation import val_multi
from Deeplabv3_rob import Deeplabv3_rob
from train_deep import load_filename, load_filename_miccai
import time
import datetime

tasks = ['Needle_Passing','Suturing' ,'Knot_Tying'] #'Suturing', ,'Knot_Tying'

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_ids = [0]
parse=argparse.ArgumentParser()
 # grasper
#'/home/aurora/Documents/segmentation/Deeplabv3_resnet/Logs_thread/T20220805_213038_e3/weights_2.pth' # thread
# '/home/aurora/Documents/segmentation/Deeplabv3_resnet/Logs/T20220804_160923/weights_0.pth' half grasper needle thread
from train_deep import num_classes # now try the needle and thread



def test(data):
    device = torch.device("cpu")
    model = Deeplabv3_rob(num_classes=num_classes,pretrained=True)

    model = model.cuda(device_ids[0])
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_load, map_location=device).items()})

    model=model.cuda(device_ids[0])
    criterion =torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(500.).cuda(device_ids[0]))#torch.nn.BCEWithLogitsLoss()#torch.nn.MSELoss(reduction='mean')#torch.nn.CrossEntropyLoss()

    for task in tasks:
        if data =='JIGSAWS':
            _, val_file_names =load_filename(task,objects)
        else:
            _, val_file_names = load_filename_miccai(objects)
        val_dataset = Load_Dataset(val_file_names,objects)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=8,shuffle=False)
    # change number_classes to 2 for binary 
        val_multi(model, criterion, val_loader, 2,batch_size=args.batch_size,device_ids=device_ids,class_type = objects, save_data=True,model_dir=weight_load)
        print('above task '+task)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=1)
    
    parse.add_argument("--dataset",  default='JIGSAWS')
    parse.add_argument("--objects", default='grasper')
    parse.add_argument("--weights", default='/home/aurora/Documents/segmentation/Deeplabv3_resnet/Logs_leftgrasper/T20220812_190615_e20/weights_2.pth')
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    objects = args.objects #'thread'
    data = args.dataset#'miccai'
    #num_epoch = args.epochs#1#00
    weight_load =args.weights #'/home/aurora/Documents/segmentation/Deeplabv3_resnet/Logs_leftgrasper/T20220812_190615_e20/weights_2.pth'
    start_t = time.time()
    test(data)
    exe_t = time.time()
    total_t = exe_t-start_t
    print(f'total time: '+str(datetime.timedelta(seconds=total_t)))