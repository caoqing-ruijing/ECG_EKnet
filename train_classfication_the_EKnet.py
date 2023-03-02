import enum
import os
import torch
import pandas as pd
import numpy as np

# from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp

import torchvision
from torchvision import transforms as T
# from torchvision.io import read_image

from tqdm import tqdm
from datetime import datetime
import time
from sklearn.metrics import classification_report,roc_auc_score,average_precision_score

# from dataset import dataloader_classfication_png as dataloader
from dataset import dataloader_classfication as dataloader
# from dataset import dataloader_PTB_classfication as dataloader

from tools import dir_utils,losses
from configs.load_yaml import load_yaml
from tools.lr_warmup_scheduler import GradualWarmupScheduler

import gc
import shutil
# import wandb
# from models.unet_e import Model
import multiprocessing as mp
import random

def onehot2pt_dict(seg_label):
    '''
    input: arr = np.array([0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,1,1,1,1,2,2,2,3,3,3,0,2,2,1])
    '''
    def consecutive(data, stepsize=0):
        '''
        [array([0, 0, 0]),array([1, 1, 1, 1]),array([2, 2, 2, 2, 2]),
         array([3, 3, 3, 3]),array([1, 1, 1, 1])]
        '''
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    if isinstance(seg_label, np.ndarray):
        seg_label = np.array(seg_label)
    seg_list = consecutive(seg_label, stepsize=0)

    seg_dict = {}
    passed_count = 0
    for seg_list_i in seg_list:
        key = str(seg_list_i[0])
        if key not in seg_dict:
            seg_dict[key] = []

        # print('seg_list_i',seg_list_i)
        start_pt = passed_count
        end_pt = passed_count+len(seg_list_i)
        
        seg_dict[key].append([start_pt,end_pt])
        passed_count = end_pt
    
    # print('passed_count',passed_count,'len(seg_list)',len(seg_label))
    assert passed_count == len(seg_label)

    return seg_dict


from copy import deepcopy
def calc_f1_threshs(val_pred_all,val_label_all,target_names,out_all_f1_csv,out_max_f1_csv,min_gap_thresh=0.01):
    # df_label=val_label_all
    # df_pred=val_pred_all
    # target_names=target_names
    # target_names_stats = target_names['micro avg','macro avg','weighted avg','samples avg']
    # df=pd.DataFrame()

    t_dict={}
    threshs_cand=[]
    for i in np.arange(0,1,min_gap_thresh):
        threshs_cand.append(i)    
    t_dict['T']=threshs_cand

    # 1. get 100 f1 results from every thresh
    for target_name in target_names:
        t_dict[target_name]=[]
    for t in np.arange(0,1,min_gap_thresh):
        df_pred_c=deepcopy(val_pred_all)
        df_pred_c[df_pred_c<t]=0 
        df_pred_c[df_pred_c>=t]=1
        val_dict = classification_report(val_label_all, df_pred_c,target_names=target_names,output_dict=True)
        for target_name in target_names:        
            # t_dict[target_name].append(0)
            f1_i = val_dict[target_name]['f1-score']
            t_dict[target_name].append(f1_i)
        # for k,v in val_dict.items():
            # t_dict[k][-1]=val_dict[k]['f1-score']

    df_all_thresh=pd.DataFrame.from_dict(t_dict)
    df_all_thresh.to_csv(out_all_f1_csv,encoding='utf-8_sig',index=False)

    # 2.get default 0.5 f1 results 
    # t_defult=[]
    df_pred_c=deepcopy(val_pred_all)
    df_pred_c[df_pred_c<0.5]=0 
    df_pred_c[df_pred_c>=0.5]=1
    val_dict_default = classification_report(val_label_all, df_pred_c,target_names=target_names,output_dict=True)
    # for k,v in val_dict_default.items():
    #     t_defult.append(val_dict_default[k]['f1-score'])
    t_defult = []
    for k in target_names:
        t_defult.append(val_dict_default[k]['f1-score'])

    # 3. get best thresh of each cls from results
    best_f1s=[]
    best_threshs=[]
    for idx,target_name in enumerate(target_names):
        f1_cls_thresh=df_all_thresh[target_name].tolist()
        best_f1 = max(f1_cls_thresh)
        default_f1 = t_defult[idx]

        if abs(default_f1-best_f1) <= 0.03:  # 差距太小不用
            best_threshs.append(0.5)
        else:
            best_threshs.append(threshs_cand[f1_cls_thresh.index(best_f1)])
        best_f1s.append(best_f1)

    # 4. get best pred each cls in 0 1 from best thresh
    # val_pred_all_best = []
    val_pred_all_best=deepcopy(val_pred_all)
    distributions = []
    for idx in range(len(target_names)):
        best_thresh = best_threshs[idx]
        val_pred_all_best[:,idx][val_pred_all_best[:,idx]>=best_thresh]=1
        val_pred_all_best[:,idx][val_pred_all_best[:,idx]<best_thresh]=0
        distributions.append(sum(val_label_all[:,idx]))

    # 5. calc the results again, why not
    assert val_pred_all_best.shape == val_pred_all.shape, 'val_pred_all_best {} != val_pred_all {}'.format(val_pred_all_best.shape,val_pred_all.shape)
    val_dict_best = classification_report(val_label_all, val_pred_all_best,target_names=target_names,output_dict=True)
    best_f1s_re = []
    for k in target_names:
        best_f1s_re.append(val_dict_best[k]['f1-score'])

    df_max=pd.DataFrame()
    df_max['label']=['micro avg','macro avg']+target_names
    df_max['num']=[np.nan,np.nan]+distributions
    df_max['best_thresh']=[np.nan,np.nan]+best_threshs
    # df_max['best_f1']=[val_dict_best['micro avg']['f1-score'],val_dict_best['macro avg']['f1-score']]+best_f1s
    df_max['best_f1']=[val_dict_best['micro avg']['f1-score'],val_dict_best['macro avg']['f1-score']]+best_f1s_re
    df_max['thresh=0.5']=[val_dict_default['micro avg']['f1-score'],val_dict_default['macro avg']['f1-score']]+t_defult
    df_max.to_csv(out_max_f1_csv,encoding='utf-8_sig',index=False)
    print(df_max)
    print(out_max_f1_csv)
    return df_pred_c,val_pred_all_best


import matplotlib.pyplot as plt
def draw_plt(inputs,label=None,outputs=None,idx=None,title='',save_path=None,sample_rates=500):
    '''
    inputs ([32, 1, 5000]) label/outputs ([32, 5000]) 
    '''
    # print('draw_plt inputs',inputs.shape,'idx',idx)
    # print('inputs[idx,:,:]',inputs[idx,:,:].shape)
    # print('-'*12)
    inputs_i = np.squeeze(inputs[idx,:,:],0)

    # label_i = label[idx,:]
    # outputs_i = outputs[idx,:]

    # save_path = './a.png'
    #x横轴 间隔 大窗口
    gap = int(0.04*sample_rates)
    xgaps = [i for i in range(0,len(inputs_i),gap)]

    gap_big = int(5*0.04*sample_rates)
    xgaps_big = [i for i in range(0,len(inputs_i)+gap_big,gap_big)] #0.2s 一大格 900->1000
    
    #y轴 间隔 大窗口
    ygaps_big = [-1,-0.5,0,0.5,1,1.5,2,2.5] #0.5mv 一大格

    x_labels = []
    for i in xgaps_big:
        #隔整数秒显示
        if (i*0.002) % 1 == 0:
            x_labels.append('{:.1f}s'.format(i*0.002))
        else:
            x_labels.append('')

    y_labels = ['{:.1f}'.format(i) for i in ygaps_big]

    fig = plt.figure(figsize=(90,8))

    colors = ['k','r','b']
    x = np.array([i for i in range(len(inputs_i))])

    if label is not None:
        values_loop = [outputs[idx,:],label[idx,:]]
    else:
        values_loop = [outputs[idx,:]]


    for i,sem_value in enumerate(values_loop):
        ax = fig.add_subplot(len(values_loop), 1, i+1)
        ax.plot(inputs_i,'k',alpha=0.99,linewidth=2,label='inputs')
        seg_dict = onehot2pt_dict(sem_value)
        for k,v in seg_dict.items():
            if len(v) > 0:
                for v_i in v:
                    start_pt,end_pt = v_i[0],v_i[1]
                    ax.plot(x[start_pt:end_pt], inputs_i[start_pt:end_pt], color=colors[int(k)],linewidth=4)

        ax.set_title(title,fontsize=35)

        #画X轴 时间段 和 y轴 mv
        ax.set_xticks(xgaps_big, minor=False)        
        ax.set_xticklabels(x_labels,fontsize=25)

        ax.set_yticks(ygaps_big, minor=False)        
        ax.set_yticklabels(y_labels,fontsize=25)

        ax.xaxis.grid(True, which='major',color='k', linewidth=2)
        ax.yaxis.grid(True, which='major',color='k', linewidth=2)

        ax.set_xlabel('time [s]')
        ax.set_ylabel('signal mV')
        ax.set_xlim(-gap_big, len(inputs_i)+gap_big)
        ax.set_ylim(ygaps_big[0], ygaps_big[-1])

    # plt.legend(['inputs','pred','label'])
    plt.legend()
    # plt.grid(True)
    plt.savefig(save_path,dpi=70)
    plt.close()
    # return plt
    return torchvision.io.read_image(save_path)


def draw_lr(train_lr,save_path):
    '''
    created by ws
    '''

    length_train=[]
    length_all=[]
    for i in range(len(train_lr)):
        length_all.append(i)
        if i==0 :
            length_train.append(str(i))
        elif i%10==0:
            length_train.append(str(i))
        else:
            length_train.append('')

    fig, ax = plt.subplots(figsize=(len(train_lr)/10,10))
    ax.plot(length_all,train_lr,'r',alpha=0.99,linewidth=2,label='learning_rate')
    
    ax.set_xticks(length_all, minor=False) #画线的index 不能有空值
    ax.set_xticklabels(length_train,fontsize=25) #需要标出来的空值
    ax.set_xlabel('epoch',fontsize=25)
    ax.set_ylabel('learning_rate',fontsize=25)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path,dpi=70)
    plt.close()
    

def draw_loss(train_loss,val_loss,learning_rates,save_path,type='loss'):
    '''
    created by ws
    '''

    length_train=[]
    length_all=[]
    for i in range(len(train_loss)):
        length_all.append(i)
        if i==0 :
            length_train.append(str(i))
        elif i%10==0:
            length_train.append(str(i))
        else:
            length_train.append('')
    
    if type == 'loss':
        min_train=min(train_loss)
        min_val=min(val_loss)
    else:
        min_train=max(train_loss)
        min_val=max(val_loss)
    set_ylabel = type

    min_t_index=train_loss.index(min_train)
    min_v_index=val_loss.index(min_val)

    # fig=plt.figure(figsize=(50,4))
    fig, ax = plt.subplots(figsize=(len(train_loss)/10,10))
    ax.plot(length_all[3:],train_loss[3:],'r',alpha=0.99,linewidth=2,label='train_loss')
    ax.plot(length_all[3:],val_loss[3:],'b',alpha=0.99,linewidth=2,label='val_loss')
    ax.plot(learning_rates,'g',alpha=0.7,linewidth=2,label='lr_curve_only')

    ax.plot(min_t_index,min_train,'r*',markersize=16)
    ax.plot(min_v_index,min_val,'b*',markersize=16)
    plt.text(min_t_index,min_train,"{:.5f}".format(min_train)+' e'+str(min_t_index),ha='left',va='bottom',weight='bold',rotation=45,fontsize=20)
    plt.text(min_v_index,min_val,"{:.5f}".format(min_val)+' e'+str(min_v_index),ha='left',va='bottom',weight='bold',rotation=45,fontsize=20)
    
    ax.set_xticks(length_all, minor=False) #画线的index 不能有空值
    ax.set_xticklabels(length_train,fontsize=25) #需要标出来的空值
    ax.set_xlabel('epoch',fontsize=25)
    ax.set_ylabel(set_ylabel,fontsize=25)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path,dpi=70)
    plt.close()


import ast
from PIL import Image, ImageFont, ImageDraw 
# def draw_text(img_path,label_text,subject_valid_target_l,subject_valid_target_p,out_save):            
def draw_text(args):

    img_path,label_text,subject_valid_target_l,subject_valid_target_p,out_save = args

    # creating a image object 
    im = Image.open(img_path) 
    draw = ImageDraw.Draw(im) 
    font = ImageFont.truetype('/home/raid_24T/qiaoran_data24T/SimHei.ttf',size=35) 
    draw.text((10, 10), '原标注: '+label_text, fill ="black", font = font, align ="left") 
    draw.text((10, 55), '用标注: '+subject_valid_target_l, fill ="blue", font = font, align ="left") 
    draw.text((10, 100), '预测为: '+subject_valid_target_p, fill ="red", font = font, align ="left") 
    im.save(out_save)



def load_pertrain(model,model_path='./'):
    pertrain_model = torch.load(model_path)

    try:
        pertrain_dict = pertrain_model.state_dict()
    except:
        pertrain_dict = pertrain_model

    model_dict = model.state_dict()
    pertrained_dict = {}
    for k,v in pertrain_dict.items():
        if k in model_dict:
            if pertrain_dict[k].size() == model_dict[k].size():
                pertrained_dict[k]=v
            else:
                print('{} filter'.format(k))
        else:
            print('{} filter'.format(k))
    model_dict.update(pertrained_dict)
    model.load_state_dict(model_dict)
    return model


def main(yaml_file,test_mode=False):
    ######### prepare environment ###########
    if torch.cuda.is_available():
        device = torch.device('cuda') 
        device_ids = [i for i in range(torch.cuda.device_count())]
        print('===> using GPU {} '.format(device_ids))
    else:
        device = torch.device('cpu')
        print('===> using CPU !!!!!')

    opt = load_yaml(yaml_file,saveYaml2output=True)
    
    try_time = opt.TRY_TIME+'_'+opt.RUN_DATE
    
    epoch = opt.OPTIM.NUM_EPOCHS

    model_dir  = opt.SAVE_DIR+'models/'
    visu_dir  = opt.SAVE_DIR+'visu/'

    dir_utils.mkdir_with_del(model_dir)
    dir_utils.mkdir_with_del(visu_dir)

    ######### dataset ###########
    # train_dataset = dataloader.Noise_Dataset(opt.DATASET.TRAIN_CSV, leads=opt.DATASET_CUSTOME.LEADS, 
    #                                             date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
    #                                             n_max_cls=opt.DATASET_CUSTOME.OUT_C,
    #                                             random_crop=True,
    #                                             transform = dataloader.get_transform(train=True)
    #                                             )

    # train_dataset = dataloader.Custome_Dataset(opt.DATASET.TRAIN_CSV, n_max_cls=opt.DATASET_CUSTOME.N_CLS, 
    #                                         transform = dataloader.get_transform(train=True))

    # val_dataset = dataloader.Custome_Dataset(opt.DATASET.VAL_CSV, n_max_cls=opt.DATASET_CUSTOME.N_CLS, 
    #                                         transform = dataloader.get_transform(train=True))

    df = pd.read_csv(opt.DATASET.STATS_CSV)
    labels_names = df['names'].tolist()
    if hasattr(opt.DATASET_CUSTOME,'operation_style'): 
        operation_style = opt.DATASET_CUSTOME.operation_style
    else:
        operation_style = 'ECG'

    train_dataset = dataloader.Custome_Dataset(opt.DATASET.TRAIN_CSV,opt.DATASET.DATA_DIR,
                                            labels_names=labels_names, 
                                            leads=opt.DATASET_CUSTOME.LEADS, 
                                            patch_size=opt.DATASET_CUSTOME.PATCH_SIZE, 
                                            operation_style=operation_style, 
                                            transform = dataloader.get_transform(train=True))

    val_dataset = dataloader.Custome_Dataset(opt.DATASET.VAL_CSV,opt.DATASET.DATA_DIR,
                                            labels_names=labels_names, 
                                            leads=opt.DATASET_CUSTOME.LEADS, 
                                            patch_size=opt.DATASET_CUSTOME.PATCH_SIZE, 
                                            operation_style=operation_style, 
                                            transform = dataloader.get_transform(train=False))


    # train_dataset = dataloader.Noise_Dataset(opt.DATASET.TRAIN_CSV, 
    #                                         leads=opt.DATASET_CUSTOME.LEADS, 
    #                                         date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
    #                                         random_crop=True,
    #                                         normlize_singal=opt.DATASET_CUSTOME.NORMLIZE_SINGAL,
    #                                         transform = dataloader.get_transform(train=True)
    #                                         )

    # val_dataset = dataloader.Noise_Dataset(opt.DATASET.VAL_CSV,     
    #                                         leads=opt.DATASET_CUSTOME.LEADS, 
    #                                         date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
    #                                         random_crop=False,
    #                                         normlize_singal=opt.DATASET_CUSTOME.NORMLIZE_SINGAL,
    #                                         transform = dataloader.get_transform(train=False)
    #                                         )


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, 
                                                    shuffle=True, num_workers=6,
                                                    prefetch_factor=3,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=int(opt.OPTIM.BATCH_SIZE//4), 
                                                    shuffle=False, num_workers=6,
                                                    prefetch_factor=3,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    # drop_last=True,
                                                    )

    dataset_sizes = {'train':len(train_dataset),
                     'val':len(val_dataset)}

    
    df_stats = pd.read_csv(opt.DATASET.STATS_CSV)
    target_names = df_stats['names'].tolist()

    print('===> Loading datasets done')

    ######### model ###########
    # if opt.MODEL.MODE == 'EKPnet':
    #     from models.EKPnet import Model
    # elif opt.MODEL.MODE == 'EKPnet_v1':
    #     from models.EKPnet_v1 import Model
    # elif opt.MODEL.MODE == 'EKPnet_v2s':
    #     from models.EKPnet_v2s import Model
    # else:
    #     print('{} unrecoginze model'.format(opt.MODEL.MODE))
    #     assert 1>2

    # model = Model(
    #             input_c=opt.DATASET_CUSTOME.INPUT_LEADS, \
    #             input_length=opt.DATASET_CUSTOME.INPUT_LENGTH, \
    #             patch_size=opt.DATASET_CUSTOME.PATCH_SIZE, \
    #             leads_input=opt.DATASET_CUSTOME.LEADS, \

    #             embedding_1d_2d=opt.MODEL.embedding_1d_2d, \
                
    #             embed_dim=opt.MODEL.embed_dim, \
    #             keep_ratio=opt.MODEL.keep_ratio, \
    #             local_depth=opt.MODEL.local_depth, \
    #             num_heads=opt.MODEL.num_heads, \
    #             pufication_style=opt.MODEL.pufication_style, \
    #             self_depth=opt.MODEL.self_depth, \

    #             normalize_before=opt.MODEL.normalize_before, \
    #             mode_stage=opt.MODEL.mode_stage, \
    #             cls_cross_depth=opt.MODEL.cls_cross_depth, \
    #             num_classes=opt.MODEL.num_classes, \

    #             ).to(device)

    '''
    orginal 不需要 knowledge_emb,pos_embed, 随便给一个占位
    '''
    if opt.MODEL.pos_emb_style == 'orginal':
        time_emb,lead_emb = None,None
    else:
        finetune_path = opt.DATASET_CUSTOME.Pertrain_Path
        pertrain_model = torch.load(finetune_path) 
        time_emb = pertrain_model['time_emb.weight'] #12*256
        lead_emb = pertrain_model['lead_emb.weight'] #12*256

    if opt.MODEL.MODE == 'EPKnet':
        from models.EPKnet import Model
    elif opt.MODEL.MODE == 'The_EKnet':
        from models.The_EKnet import Model
    else:
        print('{} unrecoginze model'.format(opt.MODEL.MODE))
        assert 1>2

    if hasattr(opt.MODEL,'time_emb_order'):
        time_emb_order = opt.MODEL.time_emb_order
    else:
        time_emb_order = 'orginal'

    model = Model(
                input_c=opt.DATASET_CUSTOME.INPUT_LEADS, \
                input_length=opt.DATASET_CUSTOME.INPUT_LENGTH, \
                patch_size=opt.DATASET_CUSTOME.PATCH_SIZE, \
                leads_input=opt.DATASET_CUSTOME.LEADS, \

                embedding_1d_2d=opt.MODEL.embedding_1d_2d, \
                
                embed_dim=opt.MODEL.embed_dim, \
                local_depth=opt.MODEL.local_depth, \
                num_head=opt.MODEL.num_head, \

                cls_cross_depth=opt.MODEL.cls_cross_depth, \

                mode_stage=opt.MODEL.mode_stage, \
                num_classes=opt.MODEL.num_classes, \

                pos_emb_style=opt.MODEL.pos_emb_style,
                lead_pos_embedding=(time_emb,lead_emb),
                pos_merging_style=opt.MODEL.pos_merging_style,
                
                classfierer_type=opt.MODEL.classfierer_type,
                ).to(device)



    if opt.DATASET_CUSTOME.Using_Pertrain is True:
        print('-'*10+'using pertrain!!') 
        model_path = opt.DATASET_CUSTOME.Pertrain_Path
        model = load_pertrain(model,model_path=model_path)

    # try:
    #     model_path = opt.DATASET_CUSTOME.Pertrain_Path
    #     model = load_pertrain(model,model_path=model_path)
    # except Exception as e:
    #     print(e)

    # from models.swin_transformer_1d import Model
    # model = Model(in_c=1,out_c=opt.DATASET_CUSTOME.OUT_C,resdiual_output=False).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids = device_ids)




    ######### optim ########### d
    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)
    
    # from optimizer.adan import Adan
    # optimizer = Adan(model.parameters(),lr=new_lr, betas=(0.98, 0.92, 0.99), eps=1e-8,
    #              weight_decay=0.02, max_grad_norm=0.0, no_prox=False,)


    warmup_epochs = 1
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=opt.OPTIM.NUM_EPOCHS)

    # CE_criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1]).to(device))
    # DiceLoss = losses.DiceLoss().to(device)
    # criterion = losses.focal_loss(alpha=None,gamma=2.,reduction='mean',ignore_index=opt.OPTIM.Focal_ignore_idx,device=device)

    # criterion = nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    print('===> model done')

    grad_scaler = amp.GradScaler()

    start_epoch = 1
    since = time.time()
    best_val_f1,best_macro_val_f1 = 0,0
    best_save_path = ''

    results_dict = {'epoch':[],
                    'train_loss':[],
                    'val_loss':[],
                    'lr':[],
                    # 'val_loss':[],
                    # 'val_accuracy':[]
                    "micro_f1_train_subs":[],
                    "micro_f1_val_subs":[],
                    "macro_f1_train_cls":[],
                    "macro_f1_val_cls":[],

                    "val_micro_roc":[],
                    "val_macro_roc":[],
                    "val_micro_ap":[],
                    "val_macro_ap":[],
                    }
    
    results_dict_f1 = {'epoch':[],        
                    "micro_f1_train_subs":[],
                    "micro_f1_val_subs":[],
                    "macro_f1_train_cls":[],
                    "macro_f1_val_cls":[],
                    }
    results_dict_aoc = {'epoch':[],
                    "val_micro_roc":[],
                    "val_macro_roc":[],
                    }
    results_dict_ap = {'epoch':[],
                    "val_micro_ap":[],
                    "val_macro_ap":[],
                    }
    cls_name_pr = []

    # opt.DRAW_EPOCHS = 
    epoch_split_n = int(opt.OPTIM.NUM_EPOCHS / opt.DRAW_EPOCHS)

    torch.autograd.set_detect_anomaly(True)
    # for epoch in range(start_epoch, 2 + 1):
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_train_loss,epoch_train_CE_loss,epoch_train_dice_loss,epoch_train_focal_loss = 0,0,0,0

        # train_pred_all,train_label_all = [],[]
        # val_pred_all,val_label_all = [],[]

        train_pred_all,train_label_all = [],[]
        val_pred_all,val_label_all = [],[]
        val_pred_all_score = []
        #### train ####
        model.train()
        for i, data in enumerate(train_dataloader):
        # for i, data in enumerate(val_dataloader):
        # for i, data in enumerate(tqdm(train_dataloader), 0):
            # if i >=10:
            #     break
            inputs = data['input'].to(device)
            # labels = data['label'].to(device) 
            labels = data['label_onehot'].to(device) 
            data_file = data['data_file']

            input_lead_loc = data['input_lead_loc'].to(device) 
            
            optimizer.zero_grad()
            # with torch.set_grad_enabled(True):
            torch.set_grad_enabled(True)
            # with amp.autocast():

            # print('inputs',inputs.shape)
            # print('input_lead_loc',input_lead_loc.shape)

            outputs = model(inputs,
                            mask_ratio=opt.DATASET_CUSTOME.MASK_RATIO,
                            # pure_ways=opt.MODEL.pure_ways,
                            # fast_pure_ways=opt.MODEL.fast_pure_ways,
                            )


            # if opt.MODEL.MODE == 'vit_1d_cls_embedding':
            if opt.MODEL.classfierer_type in ['cls_embeeing','cls_embeeing_seperate']:
                criterion = nn.CrossEntropyLoss() # 函数自己有 logsfotmax 不需要额外加softmax
                preds = outputs.reshape((outputs.shape[0]*outputs.shape[1], -1)) # [bs, n_cls, 2] -> [bs*n_cls,2]
                labels_flatten = labels.reshape(labels.shape[0]*labels.shape[1]).to(torch.int64) # [bs, n_cls] -> [bs*n_cls]
                loss = criterion(preds, labels_flatten)

                outputs_idx = torch.max(preds, 1)[1]
                outputs_idx = outputs_idx.reshape((outputs.shape[0],outputs.shape[1])) #[bs*n_cls,2] max ->[bs*n_cls] -> [bs,n_cls]
            else:
                criterion = nn.BCELoss() 
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)

                outputs_idx = outputs.detach().clone()
                outputs_idx[outputs_idx>=0.5]=1
                outputs_idx[outputs_idx<0.5]=0

            loss.backward()
            optimizer.step()

            train_pred_all.append(outputs_idx.cpu().detach().numpy())
            train_label_all.append(labels.cpu().detach().numpy())
            epoch_train_loss += loss.item() * inputs.size(0)

        train_loss_mean = epoch_train_loss / dataset_sizes['train']
        train_pred_all = np.concatenate(train_pred_all, axis=0)
        train_label_all = np.concatenate(train_label_all, axis=0)

        #### Evaluation ####
        model.eval()
        epoch_val_loss,epoch_val_CE_loss,epoch_val_dice_loss,epoch_val_focalLoss = 0,0,0,0
        # for data in val_dataloader:
        for i, data in enumerate(val_dataloader):
            # if i >=10:
            #     break
            inputs = data['input'].to(device)
            # labels = data['label'].to(device) 
            labels = data['label_onehot'].to(device) 
            data_file = data['data_file']
            input_lead_loc = data['input_lead_loc'].to(device) 

            torch.set_grad_enabled(False)
            
            if opt.MODEL.fast_pufication == 'random_mask':
                val_mask_ratio = 0.
            elif opt.MODEL.fast_pufication == 'score_embedding':
                val_mask_ratio = opt.DATASET_CUSTOME.MASK_RATIO
            else:
                raise ValueError(f'fast_pufication {opt.MODEL.fast_pufication} not support')

            outputs = model(inputs,
                            mask_ratio=0,
                            # pure_ways='atten',
                            # fast_pure_ways='atten',
                            )


            # outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)
            # loss = criterion(outputs, labels)

            # if opt.MODEL.MODE == 'vit_1d_cls_embedding':
            if opt.MODEL.classfierer_type in ['cls_embeeing','cls_embeeing_seperate']:
                criterion = nn.CrossEntropyLoss() # one target index multi cls
                preds = outputs.reshape((outputs.shape[0]*outputs.shape[1], -1)) # [bs, n_cls, 2] -> [bs*n_cls,2]

                labels_flatten = labels.reshape(labels.shape[0]*labels.shape[1]).to(torch.int64) # [bs, n_cls] -> [bs*n_cls]

                loss = criterion(preds, labels_flatten)

                preds_score = torch.softmax(preds,1)
                outputs_idx = torch.max(preds_score, 1)[1]
                outputs_idx = outputs_idx.reshape((outputs.shape[0],outputs.shape[1])) #[bs*n_cls,2] max ->[bs*n_cls] -> [bs,n_cls]

                # outputs_score = torch.max(preds_score, 1)[0]
                outputs_score = preds_score[:,1]
                outputs_score = outputs_score.reshape((outputs.shape[0],outputs.shape[1])) #[bs*n_cls,2] max ->[bs*n_cls] -> [bs,n_cls]
            else:
                criterion = nn.BCELoss()
                outputs = torch.sigmoid(outputs)
                outputs_score = outputs.clone()
                loss = criterion(outputs, labels)
                outputs[outputs>=0.5]=1
                outputs[outputs<0.5]=0
                outputs_idx = outputs

            # outputs[outputs>=0.5]=1
            # outputs[outputs<0.5]=0
            val_pred_all.append(outputs_idx.cpu().detach().numpy())
            val_pred_all_score.append(outputs_score.cpu().detach().numpy())
            val_label_all.append(labels.cpu().detach().numpy())

            epoch_val_loss += loss.item()* inputs.size(0)

        val_pred_all = np.concatenate(val_pred_all, axis=0)
        val_pred_all_score = np.concatenate(val_pred_all_score, axis=0)
        val_label_all = np.concatenate(val_label_all, axis=0)

        val_loss_mean = epoch_val_loss / dataset_sizes['val']
        scheduler.step()

        # if epoch >30:
        # if best_val_loss == 0:
        #     best_val_loss = val_loss_mean

        # if best_val_loss < val_loss_mean:
        save_path = model_dir+'model_epoch_{}_val_{:.6f}.pth'.format(epoch,val_loss_mean)
        # torch.save({'epoch': epoch, 
        #             'state_dict': model.state_dict(),
        #             'optimizer' : optimizer.state_dict()
        #             }, save_path)
        
        # torch.save(model, save_path)
        if torch.cuda.device_count() > 1: #DataParallel 带有 module, save时候要去掉
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model, save_path)

        best_val_loss = val_loss_mean
        # print(save_path)

        train_dict = classification_report(train_label_all, train_pred_all,target_names=target_names,output_dict=True)
        val_dict = classification_report(val_label_all, val_pred_all, target_names=target_names,output_dict=True)

        # print('val_label_all',val_label_all,val_label_all.shape)
        # print('val_pred_all',val_pred_all_score,val_pred_all_score.shape)
        # np.save('./val_label_all', val_label_all)
        # np.save('./val_pred_all', val_pred_all_score)


        val_dict_roc = roc_auc_score(val_label_all, val_pred_all_score, average=None)
        val_micro_roc = roc_auc_score(val_label_all, val_pred_all_score, average='micro')
        val_macro_roc = roc_auc_score(val_label_all, val_pred_all_score, average='macro')

        val_dict_ap = average_precision_score(val_label_all, val_pred_all_score, average=None)
        val_micro_ap = average_precision_score(val_label_all, val_pred_all_score, average='micro')
        val_macro_ap = average_precision_score(val_label_all, val_pred_all_score, average='macro')
        
        # print(classification_report(val_label_all, val_pred_all, target_names=target_names))
        # print('train_dict',train_dict)
        # print('val_dict',val_dict)

        train_micro_avg_f1 = train_dict['micro avg']['f1-score']
        train_macro_avg_f1 = train_dict['macro avg']['f1-score']
        val_micro_avg_f1 = val_dict['micro avg']['f1-score']
        val_macro_avg_f1 = val_dict['macro avg']['f1-score']
        
        '''
        '''
        if epoch == 1:
            best_val_f1 = val_micro_avg_f1
            best_save_path = save_path
            best_epoch = epoch
            
        if val_micro_avg_f1 > best_val_f1 :
            best_val_f1 = val_micro_avg_f1
            best_save_path = save_path
            best_epoch = epoch
        elif val_macro_avg_f1 > best_macro_val_f1 :
            best_macro_val_f1 = val_macro_avg_f1
            best_save_path = save_path
            best_epoch = epoch


        results_dict['macro_f1_train_cls'].append(train_macro_avg_f1*100)
        results_dict['macro_f1_val_cls'].append(val_macro_avg_f1*100)
        results_dict['micro_f1_train_subs'].append(train_micro_avg_f1*100)
        results_dict['micro_f1_val_subs'].append(val_micro_avg_f1*100)

        results_dict['val_micro_roc'].append(val_micro_roc*100)
        results_dict['val_macro_roc'].append(val_macro_roc*100)
        results_dict['val_micro_ap'].append(val_micro_ap*100)
        results_dict['val_macro_ap'].append(val_macro_ap*100)

            
        results_dict_f1['epoch'].append(epoch)
        results_dict_aoc['epoch'].append(epoch)
        results_dict_ap['epoch'].append(epoch)

        results_dict_f1['macro_f1_train_cls'].append(train_macro_avg_f1*100)
        results_dict_f1['macro_f1_val_cls'].append(val_macro_avg_f1*100)
        results_dict_f1['micro_f1_train_subs'].append(train_micro_avg_f1*100)
        results_dict_f1['micro_f1_val_subs'].append(val_micro_avg_f1*100)

        results_dict_aoc['val_micro_roc'].append(val_micro_roc*100)
        results_dict_aoc['val_macro_roc'].append(val_macro_roc*100)
        results_dict_ap['val_micro_ap'].append(val_micro_roc*100)
        results_dict_ap['val_macro_ap'].append(val_macro_roc*100)

        for cls_idx,cls_name in enumerate(target_names):
            r = train_dict[cls_name]['recall']*100
            p = train_dict[cls_name]['precision']*100
            f1 = train_dict[cls_name]['f1-score']*100

            r_v = val_dict[cls_name]['recall']*100
            p_v = val_dict[cls_name]['precision']*100
            f1_v = val_dict[cls_name]['f1-score']*100

            roc_v = val_dict_roc[cls_idx]*100
            ap_v = val_dict_ap[cls_idx]*100

            if '{}_f1'.format(cls_name) not in results_dict_f1: 
                results_dict_f1['{}_f1'.format(cls_name)]=[]
                results_dict_f1['{}_p'.format(cls_name)]=[]
                results_dict_f1['{}_r'.format(cls_name)]=[]
                results_dict_aoc['{}'.format(cls_name)]=[]
                results_dict_ap['{}'.format(cls_name)]=[]
                cls_name_pr.append('{}_p'.format(cls_name))
                cls_name_pr.append('{}_r'.format(cls_name))

            results_dict_f1['{}_f1'.format(cls_name)].append(f1_v)
            results_dict_f1['{}_p'.format(cls_name)].append(p_v)
            results_dict_f1['{}_r'.format(cls_name)].append(r_v)
            results_dict_aoc['{}'.format(cls_name)].append(roc_v)
            results_dict_ap['{}'.format(cls_name)].append(ap_v)
        

        if 'accuracy' not in train_dict:
            train_dict['accuracy'] = 0.0
        if 'accuracy' not in val_dict:
            val_dict['accuracy'] = 0.0 

        # results_dict['val_accuracy'].append(val_dict['accuracy']*100)

        results_dict['epoch'].append(epoch)
        results_dict['train_loss'].append(train_loss_mean)
        results_dict['val_loss'].append(val_loss_mean)
        results_dict['lr'].append(scheduler.get_lr()[0])
        
        # print('results_dict',results_dict)
        if epoch > 5:
            large_ratio = results_dict['train_loss'][4]/new_lr
            draw_loss(results_dict['train_loss'],results_dict['val_loss'],np.array(results_dict['lr'])*large_ratio,opt.SAVE_DIR+'loss.png',type='loss')
        draw_loss(results_dict['micro_f1_train_subs'],results_dict['micro_f1_val_subs'],np.array(results_dict['lr'])*(50/new_lr),opt.SAVE_DIR+'micro_f1.png',type='micro f1')
        draw_loss(results_dict['macro_f1_train_cls'],results_dict['macro_f1_val_cls'],np.array(results_dict['lr'])*(50/new_lr),opt.SAVE_DIR+'macro_f1.png',type='macro f1')
        draw_lr(results_dict['lr'],opt.SAVE_DIR+'lr.png')

        # assert 1>2
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}s \t train Loss: {:.6f} train macro f1: {:.2f}  \t val loss: {:.6f} val macro f1: {:.2f} \t LearningRate {:.8f}".format(
                epoch, time.time()-epoch_start_time, train_loss_mean,train_macro_avg_f1*100, val_loss_mean,val_macro_avg_f1*100, scheduler.get_lr()[0]))
        # print('train CE loss {:.1f}: {:.6f}, val CE loss: {:.6f}, train Dice loss {:.1f}: {:.6f},  val Dice loss: {:.6f}, train focal loss {:.1f}: {:.6f}, val focal loss: {:.6f}, '.format(
        #     opt.OPTIM.CE_ratio,train_CE_loss_mean,val_CE_loss_mean,opt.OPTIM.Dice_ratio,train_dice_loss_mean,val_dice_loss_mean,opt.OPTIM.Focal_ratio,train_focal_loss_mean,val_focal_loss_mean)
        #     )
        print("------------------------------------------------------------------")

        # results_dict.pop('macro_f1_train_cls')
        # results_dict.pop('micro_f1_train_subs')
        # print(results_dict)
        df = pd.DataFrame.from_dict(results_dict).round(6)
        # df = df.sort_values(by=['timeint'])
        df = df.drop(['macro_f1_train_cls'], axis=1)
        df = df.drop(['micro_f1_train_subs'], axis=1)
        df = df.drop(['lr'], axis=1)

        df.to_csv(opt.SAVE_DIR+'results.csv',index=False,encoding='utf-8_sig')
        # for k,v in results_dict_f1.items():
        #     print(k,v,len(v))
        try:
            df = pd.DataFrame.from_dict(results_dict_f1).round(6)
        except:
            for k,v in results_dict_f1.items():
                print(k,v,len(v))
            assert 1>2
        df.to_csv(opt.SAVE_DIR+'results_pr_f1.csv',index=False,encoding='utf-8_sig')    
        df = df.drop(cls_name_pr, axis=1)
        df.to_csv(opt.SAVE_DIR+'results_f1.csv',index=False,encoding='utf-8_sig')    


        df = pd.DataFrame.from_dict(results_dict_aoc).round(6)
        df.to_csv(opt.SAVE_DIR+'results_aoc.csv',index=False,encoding='utf-8_sig')

        df = pd.DataFrame.from_dict(results_dict_ap).round(6)
        df.to_csv(opt.SAVE_DIR+'results_ap.csv',index=False,encoding='utf-8_sig')


    analya_last = False
    if analya_last is True:
        from val_draw_atten_multi_label import analysis_atten
        # best_epoch = 300
        # best_save_path = '/home/raid_24T/qiaoran_data24T/All_project_model_output/ecg_clsffication/ECG_children_1100_res34_n_cls15_T0_2021_12_01-14_12/models/model_epoch_300_val_0.000023.pth'
        # draw_dir = opt.SAVE_DIR+'e{}/'.format(best_epoch)
        # '/home/raid_24T/qiaoran_data24T/儿科心电图/clean_data/xml数据/draw/'

        # draw_dir = '/home/raid_24T/qiaoran_data24T/Ruijing_data/ECG_data/40W/orginal_png/'
        # data_file = '{}{}.json'.format(self.data_dir,names_i)

        print(best_save_path)
        model = torch.load(best_save_path)
        model.eval()
        torch.set_grad_enabled(False)

        val_pred_all,val_label_all,data_files = [],[],[]
        val_pred_all_threshs = []
        # for data in val_dataloader:
        for i, data in enumerate(val_dataloader):
            # if i >=10:
            #     break
            inputs = data['input'].to(device)
            labels = data['label_onehot'].to(device) 
            data_file = data['data_file']
            input_lead_loc = data['input_lead_loc'].to(device) 

            outputs = model(inputs,
                            input_lead_loc,
                            mask_ratio=0.0,
                            random_mask_in_lead=False)
            
            # outputs = model(inputs)

            # if opt.MODEL.MODE == 'vit_1d_cls_embedding':
            # if opt.MODEL.classfierer_type in ['cls_embeeing']:
            if opt.MODEL.classfierer_type in ['cls_embeeing','cls_embeeing_seperate']:
                preds = outputs.reshape((outputs.shape[0]*outputs.shape[1], -1)) # [bs, n_cls, 2] -> [bs*n_cls,2]
                # labels_flatten = labels.reshape(labels.shape[0]*labels.shape[1]).to(torch.int64) # [bs, n_cls] -> [bs*n_cls]

                outputs_idx = torch.max(preds, 1)[1]
                outputs_idx = outputs_idx.reshape((outputs.shape[0],outputs.shape[1])) #[bs*n_cls,2] max ->[bs*n_cls] -> [bs,n_cls]
                outputs_score = torch.max(preds, 1)[0]
                outputs_score = outputs_score.reshape((outputs.shape[0],outputs.shape[1])) #[bs*n_cls,2] max ->[bs*n_cls] -> [bs,n_cls]
                val_pred_all_threshs.append(outputs_score.cpu().detach().numpy())

            else:
                outputs = torch.sigmoid(outputs)
                val_pred_all_threshs.append(outputs.cpu().detach().numpy())

                outputs[outputs>=0.5]=1
                outputs[outputs<0.5]=0
                outputs_idx = outputs

            # outputs = torch.sigmoid(outputs)

            # outputs[outputs>=0.5]=1
            # outputs[outputs<0.5]=0

            val_pred_all.append(outputs_idx.cpu().detach().numpy())
            val_label_all.append(labels.cpu().detach().numpy())
            data_files+=data_file

        val_pred_all = np.concatenate(val_pred_all, axis=0)
        val_label_all = np.concatenate(val_label_all, axis=0)
        val_pred_all_float_raw = np.concatenate(val_pred_all_threshs, axis=0)
        # print('val_pred_all',val_pred_all.shape) # (326, 15)
        # print('val_label_all',val_label_all.shape) # (326, 15)
        # print('val_pred_all_float_raw',val_pred_all_float_raw.shape) # (326, 15)
        # print('data_files',len(data_files)) # 326


        out_all_f1_csv = opt.SAVE_DIR+'analysis_{}_all_thresh_f1.csv'.format(best_epoch)
        out_max_f1_csv = opt.SAVE_DIR+'analysis_{}_best_thresh_f1.csv'.format(best_epoch)
        # val_pred_all_defaule,val_pred_all_best = calc_f1_threshs(val_pred_all_float_raw,val_label_all,target_names,out_all_f1_csv,out_max_f1_csv,min_gap_thresh=0.01)
        # val_pred_all = val_pred_all_best


        def prepare_dict():
            df = pd.read_csv('/home/raid_24T/qiaoran_data24T/Ruijing_data/ECG_data/110w/ocr_result_110w/save_data_profile_v3_110w_matching_label_w_pdf_json_dropnan.csv')
            ECG_ID = df['ECG_ID'].tolist()
            # file_path = df['file_path'].tolist()
            clean_arrythmias = df['clean_arrythmias'].tolist()
            full_label_dict = {}
            for k,v in zip(ECG_ID,clean_arrythmias):
                label_text = ''
                v=ast.literal_eval(v)
                for v_i in v:
                    label_text+='{} '.format(v_i)
                full_label_dict[str(k)]=label_text
            return full_label_dict


        def get_label_pred(subject_all_idx,target_names,subject_all_idx_p_raw=None):
            subject_valid_target = ''
            for subject_all_idx_i,subject_all_result_i in enumerate(subject_all_idx):
                if subject_all_result_i == 1:  
                    # subject_valid_target.append(target_names[subject_all_idx_i])
                    subject_valid_target += '{} '.format(target_names[subject_all_idx_i])
                    if subject_all_idx_p_raw is not None:
                        subject_valid_target += '{:.2f} '.format(subject_all_idx_p_raw[subject_all_idx_i])
            return subject_valid_target


        full_label_dict = prepare_dict()

        df_stats = pd.read_csv(opt.DATASET.STATS_CSV)
        target_names = df_stats['names'].tolist()

        analysis_dir=opt.SAVE_DIR+'analysis/'
        dir_utils.mkdir_with_del(analysis_dir)
        # save_path = model_dir+'model_epoch_{}_val_{:.6f}.pth'.format(epoch,val_loss_mean)

        # for idx in val_pred_all.shape[0]:
        # args_all = []

        val_dict = classification_report(val_label_all, val_pred_all, target_names=target_names,output_dict=True)

        name_list = ['micro_subs','macro_cls']
        distibutuon_list = [np.nan,np.nan]
        resutls_list = [val_dict['micro avg']['f1-score'],val_dict['macro avg']['f1-score']]

        n_right,n_miss,n_over = [np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]


        model.train()
        torch.set_grad_enabled(True)

        for idx,cls_name in tqdm(enumerate(target_names)):
            args_all = {'right':[],'miss':[],'over':[]}
            n_right_i,n_miss_i,n_over_i = 0,0,0

            val_pred_all_i = val_pred_all[:,idx]
            val_label_all_i = val_label_all[:,idx]

            val_pred_all_i = val_pred_all_i.tolist()
            val_label_all_i = val_label_all_i.tolist()

            val_pred_all_i = [int(i) for i in val_pred_all_i]
            val_label_all_i = [int(i) for i in val_label_all_i]

            analysis_dir_cls_name = '{}{}_{}/'.format(analysis_dir,idx,cls_name)

            analysis_dir_cls_name_right = analysis_dir_cls_name+'right/'
            analysis_dir_cls_name_miss = analysis_dir_cls_name+'miss/'
            analysis_dir_cls_name_over = analysis_dir_cls_name+'over/'

            name_list.append(cls_name)
            distibutuon_list.append(sum(val_label_all_i))
            f1_v = val_dict[cls_name]['f1-score']*100
            resutls_list.append(f1_v)

            # n_miss_counter = 0
            # n_right = 0
            # false_neg = 0
            
            # if cls_name
            for i in range(len(val_pred_all_i)):
                pred_i,label_i,data_file_ori_i = val_pred_all_i[i],val_label_all_i[i],data_files[i]

                file_name = os.path.basename(data_file_ori_i)
                filename, file_extension = os.path.splitext(file_name)
                # draw_dir_path_i = draw_dir+filename+'.png'

                # draw_dir_path_i = data_file_ori_i.replace('orginal_json','orginal_png')
                # draw_dir_path_i = draw_dir_path_i.replace('.json','.png')
                draw_dir_path_i = data_file_ori_i.replace("/home/qiaoran_m1/qiaoran/Holter_P_QRS_T/dataset/raw_data/orginal_jsonzip_ecgzip/",'/home/raid_24T/qiaoran_data24T/Ruijing_data/ECG_data/110w/orginal_png/')
                draw_dir_path_i = draw_dir_path_i.replace('.ecgzip','.png')
                if not os.path.exists(draw_dir_path_i):
                    assert 1>2,f"{draw_dir_path_i} not found"

                out_path_name_i = filename+'_p_{}_la_{}.png'.format(pred_i,label_i)

                # args_all = {'right':[],'miss':[],'over':[]}

                subject_all_idx_p = val_pred_all[i,:]
                subject_all_idx_l = val_label_all[i,:]

                # subject_valid_target_p = get_label_pred(subject_all_idx_p,target_names)
                subject_valid_target_l = get_label_pred(subject_all_idx_l,target_names)

                subject_all_idx_p_raw = val_pred_all_float_raw[i,:]
                subject_valid_target_p = get_label_pred(subject_all_idx_p,target_names,subject_all_idx_p_raw=subject_all_idx_p_raw)

                label_text = full_label_dict[filename]

                if (pred_i == label_i) and (pred_i==1):
                    if len(args_all['right']) == 0: 
                        dir_utils.mkdir_without_del(analysis_dir_cls_name)
                        dir_utils.mkdir_without_del(analysis_dir_cls_name_right)
                    out_path_i = analysis_dir_cls_name_right+out_path_name_i
                    n_right_i+=1

                    if opt.MODEL.classfierer_type in ['fc']:
                        visu_path = f'{analysis_dir_cls_name_right}p{pred_i}_l{label_i}_{filename}'
                        if n_right_i <= 5:
                            if n_right_i <= 5:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=True)
                            else:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=False)

                            visu_path_i = visu_path + '_data.png'
                            assert os.path.exists(visu_path_i), f"{visu_path_i} not found"
                            args_all['right'].append([visu_path + '_data.png',label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])
                    else:
                        args_all['right'].append([draw_dir_path_i,label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])


                elif pred_i==0 and label_i==1:
                    if len(args_all['miss']) == 0: 
                        dir_utils.mkdir_without_del(analysis_dir_cls_name)
                        dir_utils.mkdir_without_del(analysis_dir_cls_name_miss)
                    out_path_i = analysis_dir_cls_name_miss+out_path_name_i  

                    subject_valid_target_p += 'miss {} {:.3f}'.format(cls_name,subject_all_idx_p_raw[idx])
                    args_all['miss'].append([draw_dir_path_i,label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])
                    n_miss_i+=1

                    if opt.MODEL.classfierer_type in ['fc']:
                        visu_path = f'{analysis_dir_cls_name_miss}p{pred_i}_l{label_i}_{filename}'
                        if n_miss_i <= 5:
                            if n_miss_i <= 5:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=True)
                            else:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=False)

                            visu_path_i = visu_path + '_data.png'
                            assert os.path.exists(visu_path_i), f"{visu_path_i} not found"
                            args_all['miss'].append([visu_path + '_data.png',label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])
                    else:
                        args_all['miss'].append([draw_dir_path_i,label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])

                elif pred_i==1 and label_i==0:
                    if len(args_all['over']) == 0: 
                        dir_utils.mkdir_without_del(analysis_dir_cls_name)
                        dir_utils.mkdir_without_del(analysis_dir_cls_name_over)
                    out_path_i = analysis_dir_cls_name_over+out_path_name_i
                    args_all['over'].append([draw_dir_path_i,label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])
                    n_over_i+=1

                    if opt.MODEL.classfierer_type in ['fc']:
                        visu_path = f'{analysis_dir_cls_name_over}p{pred_i}_l{label_i}_{filename}'
                        if n_over_i <= 5:
                            if n_over_i <= 5:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=True)
                            else:
                                analysis_atten(opt,data_file_ori_i,model,opt.DATASET_CUSTOME.LEADS,device,target_names=target_names,
                                                visu_path=visu_path,force_idx=idx,all_pred=None,draw_layer=False)
            
                            visu_path_i = visu_path + '_data.png'
                            assert os.path.exists(visu_path_i), f"{visu_path_i} not found"
                            args_all['over'].append([visu_path + '_data.png',label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])
                    else:
                        args_all['over'].append([draw_dir_path_i,label_text,subject_valid_target_l,subject_valid_target_p,out_path_i])

                # else:
                #     assert 1>2, f'pred_i {pred_i} label_i{label_i}'

                # visu_path_i = visu_path + '_data.png'
                # assert os.path.exists(visu_path_i), f"{visu_path_i} not found"

                # if not os.path.exists(draw_dir_path_i):
                #     assert 1>2,'{} not exixts'.format(draw_dir_path_i)

            n_right.append(n_right_i)
            n_miss.append(n_miss_i)
            n_over.append(n_over_i)
            
            random.shuffle(args_all['right'])
            random.shuffle(args_all['miss'])
            random.shuffle(args_all['over'])

            # pool = mp.Pool(8)
            # result = pool.map(draw_text, args_all['right'][:30])
            # result = pool.map(draw_text, args_all['miss'][:30])
            # result = pool.map(draw_text, args_all['over'][:30])
            # pool.close()
            # pool.join()


        df = pd.DataFrame()
        df['names']=name_list
        df['f1']=resutls_list
        df['n_label']=distibutuon_list
        df['miss']=n_right
        df['over']=n_miss
        df['right']=n_over
        df.to_csv(opt.SAVE_DIR+'analysis_{}.csv'.format(best_epoch),index=False,encoding='utf-8_sig')
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="train")
    parser.add_argument("-c", "--config", type=str, 
                        default=None,
                        help="path to yaml file")
    args = parser.parse_args()

    main(args.config,test_mode=False)

