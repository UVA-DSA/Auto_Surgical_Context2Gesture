import numpy as np
from torch import nn
import torch
import tqdm
import math
from PIL import Image as im
import os
import pandas as pd

def val_multi(model: nn.Module, criterion, valid_loader, num_classes,batch_size,device_ids,class_type,save_data=False,model_dir=None, perf=True):
    
    with torch.no_grad():
        model.eval()
        losses = []
        iou_all = []
        dice_all = []
        recall_all = []
        precison_all = []
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        dt_size = len(valid_loader.dataset)
        tq = tqdm.tqdm(total=math.ceil(dt_size / batch_size))
        tq.set_description('Test')
        if perf==False:
            idx = 0
            for ori,inputs in valid_loader: 
                #breakpoint() 
                name =ori[-1].split('/')[-1][:-4]
                inputs = inputs.cuda(device_ids[0])
              
                outputs = model(inputs)
                outputs = outputs['out']
              
                if  num_classes>2:
                    output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                else:
                    output_classes = torch.sigmoid(outputs).data.cpu().numpy()
                output_classes[output_classes>0.95] = 1 # fixed for binary classification
                tq.update(1)
                
                if save_data:
                    #breakpoint()
                    save_path =  os.path.join('/home/student/Documents/GitHub/video-slicing/'+class_type+'_base',model_dir.split('/')[-2],ori[-1].split('/')[-2])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path,exist_ok=True)
                        print(save_path)

                    # np.save(os.path.join(save_path, test_data_naming_pred_gt),data_gt_pred)
                    #breakpoint()
                    data = im.fromarray(np.squeeze(output_classes*40).astype(np.uint8))
                    data.save(os.path.join(save_path, '{}_pred.png'.format(name)))
                    idx += 1
            tq.close()
            return True
        else:
            idx = 0
            for ori, inputs, targets in valid_loader:
                data_gt_pred = [targets]
                #breakpoint()
                ori = ori[0]
                name =ori.split('/')[-1][:-4]
                inputs = inputs.cuda(device_ids[0])
                targets = targets.long()
                targets = targets.cuda(device_ids[0])
                outputs = model(inputs)
                outputs = outputs['out']
                loss = criterion(outputs.float(), targets.float())
                losses.append(loss.item())
                if  num_classes>2:
                    output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                else:
                    output_classes = torch.sigmoid(outputs).data.cpu().numpy()
                output_classes[output_classes<=0.95] = 0
                output_classes[output_classes>0.95] = 1 # fixed for binary classification
                
                # for mse for classes<2
                # if num_classes<3:
                #     target_classes = targets.data.cpu().numpy().argmax(axis=1)
                # else:
                target_classes = targets.data.cpu().numpy()     
                confusion_matrix = calculate_confusion_matrix_from_arrays(
                    output_classes, target_classes, num_classes)
                
                if targets.sum()!=0:
                    iou_all.append(calculate_iou(confusion_matrix)[1])
                    dice_all.append(calculate_dice(confusion_matrix)[1])
                    precison_all.append(calculate_precision(confusion_matrix)[1])
                    recall_all.append(calculate_recall(confusion_matrix)[1])

                tq.set_postfix(loss='{0:.3f}'.format(np.mean(losses)))
                tq.update(1)
                
                data_gt_pred.append( torch.sigmoid(outputs).data.cpu().numpy())
                test_data_naming_pred_gt = '{}_gt_pred.npy'.format(name)
                if save_data:
                    #breakpoint()


                    save_path =  os.path.join('/home/student/Documents/GitHub/video-slicing/'+class_type+'_base',model_dir.split('/')[-2],ori.split('/')[-2])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path,exist_ok=True)
                        print(save_path)

                    np.save(os.path.join(save_path, test_data_naming_pred_gt),data_gt_pred)
                    #breakpoint()
                    data = im.fromarray(np.squeeze(output_classes*40).astype(np.uint8))
                    data.save(os.path.join(save_path, '{}_pred.png'.format(name)))
                    idx += 1
            tq.close()
            confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
            valid_loss = np.mean(losses)  # type: float
            average_iou = np.mean(iou_all)
            average_dices = np.mean(dice_all)
            average_precision = np.mean(precison_all)
            average_recall = np.mean(recall_all)

            #average_iou = np.mean(list(ious.values()))
            #average_dices = np.mean(list(dices.values()))

            print('Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f},  average Precision: {:.4f},  average Recall: {:.4f}'.format(valid_loss, average_iou, average_dices,average_precision,average_recall))


            return average_dices, average_iou


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices


def calculate_precision(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives 
        if denom == 0:
            dice = 0
        else:
            dice = float(true_positives) / denom
        dices.append(dice)
    return dices

def calculate_recall(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice =float(true_positives) / denom
        dices.append(dice)
    return dices

