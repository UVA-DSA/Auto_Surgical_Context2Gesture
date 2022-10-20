import numpy as np
import glob
def load_filename(task):
    from itertools import chain
    mask_dir ='/home/aurora/Documents/segmentation/JIGSAWS-Cogito-Annotations-main/{}/mask_output_grasper'.format(task)
    train_file=glob.glob(mask_dir+'/*')
    if task =='Suturing':   
        # those have half the resolution
        dir_half = np.array(['Suturing_S07_T01', 'Suturing_S07_T02', 'Suturing_S07_T03','Suturing_S07_T04', 'Suturing_S07_T05', 'Suturing_S08_T01'])
        # those the test set
        dir_test = np.array(['Suturing_S08_T04', 'Suturing_S02_T04','Suturing_S02_T01', 'Suturing_S03_T04','Suturing_S03_T05','Suturing_S05_T03'])
        set_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_half)-set(dir_test))

    elif task=='Needle_Passing':
        dir_half = np.array(['Needle_Passing_S06_T01','Needle_Passing_S06_T03','Needle_Passing_S06_T04','Needle_Passing_S08_T02','Needle_Passing_S08_T04','Needle_Passing_S08_T05','Needle_Passing_S09_T03'])
        dir_test = np.array(['Needle_Passing_S04_T03','Needle_Passing_S04_T01','Needle_Passing_S05_T03','Needle_Passing_S05_T05'])
        set_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_test)-set(dir_half))
    else:
        dir_half = np.array(['Knot_Tying_S06_T04','Knot_Tying_S06_T05','Knot_Tying_S07_T01','Knot_Tying_S07_T02','Knot_Tying_S07_T03','Knot_Tying_S07_T04','Knot_Tying_S07_T05'])
        dir_test = np.array(['Knot_Tying_S04_T02','Knot_Tying_S05_T05','Knot_Tying_S03_T05','Knot_Tying_S05_T03','Knot_Tying_S03_T02'])
        set_train = list(set(list(map(lambda x:x.split('/')[-1],train_file)))-set(dir_test)-set(dir_half))
    train_file_names =list(chain.from_iterable(list(map(lambda x:glob.glob(mask_dir+'/'+x+'/*'), set_train))))
    val_file_names = list(chain.from_iterable(list(map(lambda x:glob.glob(mask_dir+'/'+x+'/*'),dir_test))))
    return train_file_names, val_file_names