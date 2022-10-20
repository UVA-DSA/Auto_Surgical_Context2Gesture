import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import sys


x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # checked web for normalization
])
y_trans=transforms.ToTensor()

class Load_Dataset(Dataset):
    def __init__(self, filenames,type, aug = False):
        self.file_names = filenames
        self.type = type
        self.aug = aug

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        down_sample = 2
        img_file_name = self.file_names[idx]
        #print(img_file_name)
        
        ori_image,ori = load_image(img_file_name,self.type)
        
        image = x_transforms(ori_image)

        mask = load_mask(img_file_name,self.type)
        if self.aug ==True:
            comp = transforms.RandomRotation(degrees= (0,180))
            ma = torch.from_numpy(mask)
            two = torch.cat((ma,image),0)
            trans = comp(two)
            mask = np.squeeze(trans[0,:,:].numpy())
            mask = mask[np.newaxis,:,:]
            mask = mask.astype(np.uint8)
            image = trans[1:,:,:]



        return ori, image,mask

def load_image(path,objects):
    path_ele = path.replace('mask_output_'+objects,'images')
    new_path = path_ele#"/".join(path_ele)
    try:
        if not 'images' in new_path:
            raise ValueError('wrong directory for images')
    except  Exception as error:
        print('Caught this error: ' + repr(error))
    #print(new_path)
    img_x = Image.open(new_path)
    ori = new_path
    
    return img_x,ori



def load_mask(path,type):
   

    # if objects not in ['needle','thread','grasper']:
    #     print('need to modify load_dataset')
    #     sys.exit()
    if type =='needle': 
        variable = 40
    elif type=='thread':
        variable = 60
    elif type =='leftgrasper' or type =='rightgrasper' or type=='grasper':
        variable = 20
    elif type=='ring':
        variable = 80
    # variable = 60
    mask = cv2.imread(path)
    mask=mask//variable
    mask = np.squeeze(mask[:,:,0])
    mask = mask[np.newaxis,:,:]

    return mask.astype(np.uint8)
