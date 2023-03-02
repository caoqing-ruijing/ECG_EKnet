import enum
import os,sys
import torch
import pandas as pd
import numpy as np

# from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.cuda import amp

import torchvision
from torchvision import transforms as T
# from torchvision.io import read_image


from tqdm import tqdm
from datetime import datetime
import time
from sklearn.metrics import classification_report

# from dataset import dataloader_semseg as dataloader
from dataset import dataloader_MAE as dataloader

from tools import dir_utils,losses
from configs.load_yaml import load_yaml
from tools.lr_warmup_scheduler import GradualWarmupScheduler

import timm.optim.optim_factory as optim_factory

import math
import gc
# import wandb
# from models.unet_e import Model
import multiprocessing as mp


import matplotlib.pyplot as plt
import copy

def draw_plt(arg):
    def get_mask_idx(mask_i_valve,patch_size=250):
        n_patch = int(len(mask_i_valve)/patch_size)
        mask_idxs,patch_pts = [],[]
        for i in range(n_patch):
            start_pt = i*patch_size
            end_pt = i*patch_size+patch_size
            patch_mask = mask_i_valve[start_pt:end_pt]
            patch_pts.append(end_pt)
            if np.sum(patch_mask) == 0:
                if len(mask_idxs) >= 1:
                    #如果patch首位相接，合并
                    last_pt_i = mask_idxs[-1][-1]
                    if start_pt == last_pt_i:
                        mask_idxs[-1][-1]=end_pt
                    else:
                        mask_idxs.append([start_pt,end_pt])
                else:                            
                    mask_idxs.append([start_pt,end_pt])
        return mask_idxs,patch_pts[:-1]
    mask_i=arg[0]
    pred_i=arg[1]
    input_i=arg[2]
    leads=arg[3]
    leads_to_draw=copy.deepcopy(leads)
    save_path=arg[4]
    patch_size = arg[5]

    lead_position={}
    lead_to_dict = ['I','aVR','V1','V4','I I','aVL','V2','V5','III','aVF','V3','V6']
    for i in range(len(lead_to_dict)):
        lead_position[lead_to_dict[i]]=i+1
    lead_position['II_1']=5
    # print(lead_position)
    
    sample_rates = 500
    gap_big = int(5*0.04*sample_rates)
    xgaps_big = [i for i in range(0,len(input_i[0])+gap_big,gap_big)] 

    
    if 'II_1' in leads:
        signal_II_input=input_i[leads.index('II_1')].tolist()+input_i[leads.index('II_2')].tolist()\
                        +input_i[leads.index('II_3')].tolist()+input_i[leads.index('II_4')].tolist()
        signal_II_mask=mask_i[leads.index('II_1')].tolist()+mask_i[leads.index('II_2')].tolist()\
                        +mask_i[leads.index('II_3')].tolist()+mask_i[leads.index('II_4')].tolist()
        signal_II_pred=pred_i[leads.index('II_1')].tolist()+pred_i[leads.index('II_2')].tolist()\
                        +pred_i[leads.index('II_3')].tolist()+pred_i[leads.index('II_4')].tolist() 
        
        xgaps_big1 = [i for i in range(0,len(signal_II_input)+gap_big,gap_big)]         
        x_labels1 = []
        for i in xgaps_big1:
            if (i*0.002) % 1 == 0:
                x_labels1.append('{:.1f}s'.format(i*0.002))
            else:
                x_labels1.append('')
        leads_to_draw.append('II')
        leads.append('II')
        leads_to_draw.remove('II_2')
        leads_to_draw.remove('II_3')
        leads_to_draw.remove('II_4')
    ygaps_big = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]
    x_labels = []
    for i in xgaps_big:
        if (i*0.002) % 1 == 0:
            x_labels.append('{:.1f}s'.format(i*0.002))
        else:
            x_labels.append('')
    y_labels = ['{:.1f}'.format(i) for i in ygaps_big]
    ratio_l = len(input_i[0])/1250 #2.5s 
    #-----画图--------
    fig = plt.figure(figsize=(int(13*ratio_l*4),4*7))
    for l in  range(len(leads_to_draw)):
        lead=leads_to_draw[l]

        if lead=='II':
            ax=plt.subplot(4,1,4)
            # print('lead_position[lead]',lead_position[str(lead)])
            ax.set_xticks(xgaps_big1, minor=False)
            ax.set_xticklabels(x_labels1,fontsize=25)
            ax.set_yticklabels(y_labels,fontsize=25)
            ax.set_ylabel('signal mV',fontsize=25)

            ax.plot(signal_II_input,color='k',alpha=0.8,linewidth=3,label='original')
            # ax.plot(signal_II_pred,color='deepskyblue',alpha=0.7,linewidth=3,label='pred') 
            # ax.plot(signal_II_mask,color='r',alpha=0.9,linewidth=2,label='mask')   
            ax.set_xlabel('time [s]',fontsize=25)
            ax.set_xlim(0, len(signal_II_mask)) 
            ax.legend(loc="upper right")
            mask_idxs,patch_pts = get_mask_idx(signal_II_mask,patch_size=patch_size)
            # print('signal_II_mask',len(signal_II_mask),signal_II_mask)
            # print('mask_idxs',mask_idxs)
            for mask_idx_i in mask_idxs:
                start_pti,end_pti = mask_idx_i
                x_i = [i for i in range(start_pti,end_pti)]
                ax.plot(x_i,signal_II_pred[start_pti:end_pti],color='orangered',alpha=0.9,linewidth=3,label='pred')
                
            pts = []
            for patch_pt_x in patch_pts:
                pts.append([patch_pt_x,signal_II_input[patch_pt_x]])
            plt.plot(*zip(*pts), marker='o', color='deepskyblue',alpha=0.9,linewidth=3,ls='')

        else:
            input_i_valve=input_i[leads.index(lead),:]
            mask_i_valve=mask_i[leads.index(lead),:]
            pred_i_valve=pred_i[leads.index(lead),:]

            ax=plt.subplot(4,4,lead_position[lead])
            ax.plot(input_i_valve,color='k',alpha=0.8,linewidth=3,label='original')
            mask_idxs,patch_pts = get_mask_idx(mask_i_valve,patch_size=patch_size)        
            ax.set_xticks(xgaps_big, minor=False)  
            ax.set_xticklabels(x_labels,fontsize=25)
            ax.set_xlim(0, len(mask_i_valve))

            for mask_idx_i in mask_idxs:
                start_pti,end_pti = mask_idx_i
                x_i = [i for i in range(start_pti,end_pti)] 
                ax.plot(x_i,pred_i_valve[start_pti:end_pti],color='orangered',alpha=0.9,linewidth=3,label='pred')

            pts = []
            for patch_pt_x in patch_pts:
                pts.append([patch_pt_x,input_i_valve[patch_pt_x]])
            plt.plot(*zip(*pts), marker='o', color='deepskyblue',alpha=0.9,linewidth=3,ls='')

            # ax1.plot(pred_i_valve,color='deepskyblue',alpha=0.7,linewidth=3,label='pred')
            # ax1.plot(mask_i_valve,color='r',alpha=0.9,linewidth=2,label='mask')# linestyle='dashed'
            list1=['I','I I','III','II_1','II']
            if lead in list1:
                ax.set_yticklabels(y_labels,fontsize=25)
                ax.set_ylabel('signal mV',fontsize=25)
            else:     
                ax.set_yticklabels('',fontsize=25)

        ax.set_title(lead,fontsize=30)
        ax.set_yticks(ygaps_big, minor=False)        
        ax.xaxis.grid(True, which='major',color='grey', linewidth=1)
        ax.yaxis.grid(True, which='major',color='grey', linewidth=1)
        # ax1.set_xlabel('time [s]',fontsize=25)
        ax.set_ylim(ygaps_big[0], ygaps_big[-1])
            
        plt.suptitle('result',fontsize = 40,weight='bold')
        plt.subplots_adjust(left=0.125,bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.2)# wspace:左右间距；hspace：上下间距

    plt.savefig(save_path)
    plt.close()
    # print('done')

def draw_one_line(train_lr,save_path):
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
    
def draw_two_line(train_loss,val_loss,save_path,mode='loss'):
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

    if mode == 'loss':
        min_train=min(train_loss)
        min_val=min(val_loss)
    else:
        min_train=max(train_loss)
        min_val=max(val_loss)

    min_t_index=train_loss.index(min_train)
    min_v_index=val_loss.index(min_val)

    # fig=plt.figure(figsize=(50,4))
    fig, ax = plt.subplots(figsize=(len(train_loss)/10,10))
    ax.plot(length_all[3:],train_loss[3:],'r',alpha=0.99,linewidth=2,label='train_loss')
    ax.plot(length_all[3:],val_loss[3:],'b',alpha=0.99,linewidth=2,label='val_loss')

    ax.plot(min_t_index,min_train,'r*',markersize=16)
    ax.plot(min_v_index,min_val,'b*',markersize=16)
    plt.text(min_t_index,min_train,"{:.5f}".format(min_train)+' e'+str(min_t_index),ha='left',va='bottom',weight='bold',rotation=45,fontsize=20)
    plt.text(min_v_index,min_val,"{:.5f}".format(min_val)+' e'+str(min_v_index),ha='left',va='bottom',weight='bold',rotation=45,fontsize=20)
    
    ax.set_xticks(length_all, minor=False) #画线的index 不能有空值
    ax.set_xticklabels(length_train,fontsize=25) #需要标出来的空值
    ax.set_xlabel('epoch',fontsize=25)
    ax.set_ylabel('loss',fontsize=25)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path,dpi=70)
    plt.close()

def load_pertrain(model,model_path='./'):
    pertrain_model = torch.load(model_path)
    pertrain_dict = pertrain_model.state_dict()


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


def tran_mask(model,input,pred,mask=None):
    if isinstance(model, nn.DataParallel):
        input = model.module.patchify(input)
    else:
        input = model.patchify(input)

    pred = pred.detach().cpu().numpy()
    if mask is None: 
        mask = np.ones((pred.shape[0],pred.shape[1]))
    else:
        mask = mask.cpu().numpy()
    # print('mask',mask.shape)
    # print('input',input.shape)
    # print('pred',pred.shape)

    # mask = mask.cpu().numpy()

    orginals = input.cpu().numpy()
    input_masks,pred_masks = [],[]
    for i in range(input.shape[0]):
        # pred_i = pred[i,:,:]
        mask_i = mask[i,:]
        input_i = orginals[i,:,:]

        idx_keep = np.where(mask_i==0)
        idx_remove = np.where(mask_i==1)
        
        masked_input = np.zeros(input_i.shape)
        pred_mask = np.zeros(input_i.shape)

        # masked input
        masked_input[idx_keep] = input_i[idx_keep]

        #pred masked input
        # pred_mask[idx_keep] = input_i[idx_keep]
        # pred_mask[idx_remove] = pred_i[idx_remove]

        input_masks.append(masked_input)
        # pred_masks.append(pred_mask)

    input_masks = torch.from_numpy(np.array(input_masks)) #(2, 60, 250)
    # pred_masks = torch.from_numpy(np.array(pred_masks))
    orginals = torch.from_numpy(orginals)

    input_masks = model.unpatchify(input_masks)
    # pred_masks = model.unpatchify(pred_masks)
    orginals = model.unpatchify(orginals)

    input_masks = input_masks.cpu().numpy() #(2, 12, 1250)
    # pred_masks = pred_masks.cpu().numpy()
    orginals = orginals.cpu().numpy()
    return orginals,input_masks,pred



def draw_cm_avg_atten(args):
    '''
    input shape  cam [ [bs, n_patch, n_patch],
                        [bs, n_patch, n_patch], 
                        [bs, n_patch, n_patch], 
                        ...]
    '''
    cam_list_avg,sample_idx,leads_n_patch,save_path,n_patch_per_lead= args
    n_layers = len(cam_list_avg)

    num_tokens = cam_list_avg[0].shape[-1]
    # n_columns = math.sqrt(n_layers)
    n_columns = math.ceil(math.sqrt(n_layers))
    n_rows = math.ceil(n_layers / n_columns)
    assert n_rows*n_columns >= len(cam_list_avg),'n_columns{} * n_rows{} > n_block {}'.format(n_columns,n_rows,n_layers)

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns*10, n_rows*10)) #h,w
    # it = iter(cam_list_avg)

    n_block_idx = 0
    for i in range(n_rows):
        for j in range(n_columns):
            if n_block_idx >= n_layers:
                break

            att_map = cam_list_avg[n_block_idx][sample_idx]
            im = axs[i, j].imshow(att_map, cmap='cool', extent = (0,num_tokens,num_tokens,0))
            # extent = (0,n_patch,0,n_patch)) # interpolation='nearest' extent (left, right, bottom, top)
            # Create colorbar
            axs[i, j].set_xticks(np.arange(num_tokens)) #30
            axs[i, j].set_yticks(np.arange(num_tokens))

            axs[i, j].xaxis.set_ticklabels(leads_n_patch,fontsize=20)
            axs[i, j].yaxis.set_ticklabels(leads_n_patch,fontsize=20)

            axs[i, j].set_title(f'layer {n_block_idx}',fontsize=40) #60
            for line_i in range(0,len(leads_n_patch),n_patch_per_lead):
                axs[i, j].axhline(y=line_i, color='black', linestyle='--',linewidth=1)
                axs[i, j].axvline(x=line_i, color='black', linestyle='--',linewidth=1)
            cbar = axs[i, j].figure.colorbar(im, ax=axs[i, j],shrink=0.7)
            cbar.ax.tick_params(labelsize=30)
            n_block_idx+=1
    fig.suptitle(f'Average attention map of {n_layers} decoder layers',fontsize=70,verticalalignment='top') # 100
    fig.tight_layout()
    plt.savefig(save_path,dpi=100)
    plt.close()


def draw_cm_head_atten(args):
    '''
    input shape  cam [ [bs, head, n_patch, n_patch],
                        [bs, head, n_patch, n_patch], 
                        [bs, head, n_patch, n_patch], 
                        ...]
    '''
    cam_list_avg,sample_idx,leads_n_patch,n_layer_idx,save_path,n_patch_per_lead = args
    n_head = cam_list_avg[0].shape[1]

    num_tokens = cam_list_avg[0].shape[-1]
    n_columns = math.ceil(math.sqrt(n_head))
    n_rows = math.ceil(n_head / n_columns)
    assert n_rows*n_columns >= len(cam_list_avg),'n_columns{} * n_rows{} > n_block {}'.format(n_columns,n_rows,len(cam_list_avg))

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns*10, n_rows*10)) #h,w
    # it = iter(cam_list_avg)

    n_block_idx = 0
    for i in range(n_rows):
        for j in range(n_columns):
            if n_block_idx >= n_head:
                break

            att_map = cam_list_avg[n_layer_idx][sample_idx][n_block_idx]
            im = axs[i, j].imshow(att_map, cmap='cool', extent = (0,num_tokens,num_tokens,0))
            # extent = (0,n_patch,0,n_patch)) # interpolation='nearest' extent (left, right, bottom, top)
            # Create colorbar
            axs[i, j].set_xticks(np.arange(num_tokens)) #30
            axs[i, j].set_yticks(np.arange(num_tokens))

            axs[i, j].xaxis.set_ticklabels(leads_n_patch,fontsize=20)
            axs[i, j].yaxis.set_ticklabels(leads_n_patch,fontsize=20)

            axs[i, j].set_title(f'head {n_block_idx}',fontsize=40) #60
            for line_i in range(0,len(leads_n_patch),n_patch_per_lead):
                axs[i, j].axhline(y=line_i, color='black', linestyle='--',linewidth=1)
                axs[i, j].axvline(x=line_i, color='black', linestyle='--',linewidth=1)
            cbar = axs[i, j].figure.colorbar(im, ax=axs[i, j],shrink=0.7)
            cbar.ax.tick_params(labelsize=30)
            n_block_idx+=1
    fig.suptitle(f'attention map of {n_layer_idx} decoder layers {n_head} head',fontsize=70,verticalalignment='top') # 100
    fig.tight_layout()
    plt.savefig(save_path,dpi=100)
    plt.close()


def get_cam(model):
    cam_list_head = [] # [ [bs, head, n_patch, n_patch], [bs, head, n_patch, n_patch], ... ]
    cam_list_avg = [] # [ [bs,n_patch,n_patch],[bs,n_patch,n_patch], ... ]
    for blk_idx,blk in enumerate(model.knowledge_self_layers):
        cam_i = blk.attn.get_attention_map() # [bs, head, n_patch, n_patch] [256,16,75,75]
        cam = []
        for k,v in cam_i.items():
            cam.append(v.cpu().detach().numpy())
        cam = np.concatenate(cam, 0) #
        cam = torch.tensor(cam)

        cam_avg = cam.mean(dim=1) # [bs,n_head,n_patch,n_patch] -> [bs,n_patch,n_patch]
        cam_list_head.append(cam.cpu().detach().numpy()) 
        cam_list_avg.append(cam_avg.cpu().detach().numpy()) 
        num_tokens = cam_avg.shape[-1]
    return cam_list_head,cam_list_avg

def get_cam_module(model):
    cam_list_head = [] # [ [bs, head, n_patch, n_patch], [bs, head, n_patch, n_patch], ... ]
    cam_list_avg = [] # [ [bs,n_patch,n_patch],[bs,n_patch,n_patch], ... ]

    for blk_idx,blk in enumerate(model.module.knowledge_self_layers):
        # cam = blk.attn.get_attention_map() # [bs, head, n_patch, n_patch] [256,16,75,75]

        cam_i = blk.attn.get_attention_map() # [bs, head, n_patch, n_patch] [256,16,75,75]
        cam = []
        for k,v in cam_i.items():
            cam.append(v.cpu().detach().numpy())
        cam = np.concatenate(cam, 0) #
        cam = torch.tensor(cam)

        cam_avg = cam.mean(dim=1) # [bs,n_head,n_patch,n_patch] -> [bs,n_patch,n_patch]

        cam_list_head.append(cam.cpu().detach().numpy()) 
        cam_list_avg.append(cam_avg.cpu().detach().numpy()) 

        num_tokens = cam_avg.shape[-1]
    return cam_list_head,cam_list_avg

def get_leads_n_patch(num_tokens,leads_default_order):
    leads_n_patch = []

    n_patch_per_lead = int(num_tokens/len(leads_default_order))
    for lead_i in leads_default_order:
        draw_token_inx = math.ceil(n_patch_per_lead/2)
        for local_token_i in range(n_patch_per_lead):
            if local_token_i == draw_token_inx:
                leads_n_patch.append('{}'.format(lead_i))
            else:
                leads_n_patch.append('')
    assert len(leads_n_patch)==num_tokens,f'leads_n_patch {len(leads_n_patch)} != num_tokens {num_tokens}'
    return  leads_n_patch,n_patch_per_lead





def forward_loss(model, input_x, pred):
    """
    input_x: [N, n_c, 1250]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    if input_x.shape != pred.shape:
        if isinstance(model, nn.DataParallel):
            target = model.module.patchify(input_x)
        else:
            target = model.patchify(input_x)
    else:
        target = input_x

    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    # loss = abs(pred - target)

    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = torch.mean(loss)
    return loss


def forward_loss_time(pred_time,time_orders_default,mask,mask_decoder):
    pred_time = pred_time.squeeze(-1) # bs,n_patch,1 -> bs,n_patch
    loss_time_norm = torch.abs((pred_time+1)/22 - (time_orders_default+1)/22) # norm bs,n_patch
    # loss_time_norm = torch.mean(loss_time_norm)
    loss_time_norm = (loss_time_norm * mask * mask_decoder).sum() / mask.sum()  # mean loss on removed patches

    l1_loss_time = torch.abs(pred_time - time_orders_default)
    # l1_loss_time = torch.mean(l1_loss_time)
    l1_loss_time = (l1_loss_time * mask * mask_decoder).sum() / mask.sum()  # mean loss on removed patches
    return loss_time_norm,l1_loss_time

def forward_loss_lead(pred_lead,lead_orders_default,mask,mask_decoder):
    lead_criterion = nn.CrossEntropyLoss()
    # 先把 pred 和 label * mask 变成 0  再算 loss，  
    pred_lead = pred_lead*mask.unsqueeze(-1).repeat(1, 1, pred_lead.shape[2])*mask_decoder.unsqueeze(-1).repeat(1, 1, pred_lead.shape[2])
    lead_orders_default = lead_orders_default*mask*mask_decoder
    lead_orders_default = lead_orders_default.to(torch.long)

    lead_orders_default_flat = lead_orders_default.reshape((lead_orders_default.shape[0]*lead_orders_default.shape[1])) # [bs, n_patch] -> [bs*n_patch]
    pred_lead_flat = pred_lead.reshape((pred_lead.shape[0]*pred_lead.shape[1],-1)) # [bs, n_patch, 12] -> [bs*n_patch,12]
    _, preds = torch.max(pred_lead_flat, 1) 

    running_corrects_lead = torch.sum(preds == lead_orders_default_flat).cpu().detach().numpy()
    total_samples_lead = preds.shape[0]
    loss_lead = lead_criterion(pred_lead_flat, lead_orders_default_flat)
    return running_corrects_lead,total_samples_lead,loss_lead


def tran_info(info_lead):

    '''
    把 loss 从info_lead 提取出来
    '''
    info_lead = info_lead.cpu().detach().numpy()
    info_lead_new = np.sum(info_lead,axis=0)
    loss_time,loss_lead, corrects_lead,total_lead = info_lead_new
    loss_time/=info_lead.shape[0]
    loss_lead/=info_lead.shape[0]
    return loss_time,loss_lead, corrects_lead,total_lead 


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
    train_dataset = dataloader.MAE_Dataset(opt.DATASET.TRAIN_CSV, data_dir=opt.DATASET.DATA_DIR,leads=opt.DATASET_CUSTOME.LEADS, 
                                                date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
                                                patch_size=opt.DATASET_CUSTOME.PATCH_SIZE,
                                                PTB_random_pos_loc=opt.DATASET_CUSTOME.PTB_random_pos_loc,
                                                
                                                normlize_singal=opt.DATASET_CUSTOME.NORMLIZE_SINGAL,
                                                depress_v1_v6=opt.DATASET_CUSTOME.Depress_v1_v6,
                                                transform = dataloader.get_transform(train=True),
                                                )

    val_dataset = dataloader.MAE_Dataset(opt.DATASET.VAL_CSV, data_dir=opt.DATASET.DATA_DIR,leads=opt.DATASET_CUSTOME.LEADS, 
                                                date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
                                                patch_size=opt.DATASET_CUSTOME.PATCH_SIZE,
                                                PTB_random_pos_loc=opt.DATASET_CUSTOME.PTB_random_pos_loc,
                                                normlize_singal=opt.DATASET_CUSTOME.NORMLIZE_SINGAL,
                                                depress_v1_v6=opt.DATASET_CUSTOME.Depress_v1_v6,
                                                transform = dataloader.get_transform(train=False)
                                                )

    draw_paths = pd.read_csv(opt.DATASET.TEST_CSV)
    test_paths = draw_paths['paths'].tolist()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, 
                                                    shuffle=True, num_workers=30,
                                                    prefetch_factor=200,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.OPTIM.BATCH_SIZE//2, 
                                                    shuffle=False, num_workers=10,
                                                    prefetch_factor=200,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    # drop_last=True,
                                                    )

    dataset_sizes = {'train':len(train_dataset),
                     'val':len(val_dataset)}

    print('===> Loading datasets done')

    ######### model ###########
    finetune_path = opt.DATASET_CUSTOME.Pertrain_Path
    pertrain_model = torch.load(finetune_path) 
    lead_emb = pertrain_model['knowledge_emb.weight'] #12*256
    pos_embed = pertrain_model['pos_embed.weight'] #12*256

    if lead_emb.shape[0] == 12:
        lead_emb = torch.cat([lead_emb[0].unsqueeze(0),
                                lead_emb[1].unsqueeze(0),
                                lead_emb[1].unsqueeze(0),
                                lead_emb[1].unsqueeze(0),
                                lead_emb[1].unsqueeze(0),
                                lead_emb[2:]],dim=0)


    if opt.MODEL.MODE == 'EPKnet':
        from models.EPKnet import Model
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

                add_pos_before_cross=opt.MODEL.add_pos_before_cross, \
                cls_cross_depth=opt.MODEL.cls_cross_depth, \

                pred_full_II_lead=opt.MODEL.pred_full_II_lead, \

                normalize_before=opt.MODEL.normalize_before, \
                mode_stage=opt.MODEL.mode_stage, \
                num_classes=opt.MODEL.num_classes, \

                pufication_module=opt.MODEL.pufication_module,  # none mlp_purification token_purification
                pufication_ratio=opt.MODEL.pufication_ratio,

                lead_split=opt.MODEL.lead_split,
                fast_pufication=opt.MODEL.fast_pufication,
                fast_pufication_head=opt.MODEL.fast_pufication_head,
                Token_Purification_M_head=opt.MODEL.Token_Purification_M_head,

                pos_emb_style=opt.MODEL.pos_emb_style,
                lead_pos_embedding=(lead_emb,pos_embed),
                lead_embedding_trainable=opt.MODEL.lead_embedding_trainable,

                time_emb_source=time_emb_order,
                ).to(device)

    if opt.DATASET_CUSTOME.Using_Pertrain is True:
        print('-'*10+'using pertrain!!') 
        model_path = opt.DATASET_CUSTOME.Pertrain_Path
        model = load_pertrain(model,model_path=model_path)

    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    # try:
    #     model_path = opt.DATASET_CUSTOME.Pertrain_Path
    #     model = load_pertrain(model,model_path=model_path)
    # except Exception as e:
    #     print(e)

    # from models.swin_transformer_1d import Model
    # model = Model(in_c=1,out_c=opt.DATASET_CUSTOME.OUT_C,resdiual_output=False).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = device_ids)

    ######### optim ########### d
    new_lr = opt.OPTIM.LR_INITIAL
    # optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)
    new_lr = new_lr*int(opt.OPTIM.BATCH_SIZE/256)

    param_groups = optim_factory.add_weight_decay(model, opt.OPTIM.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=new_lr, betas=(0.9, 0.95))

    
    warmup_epochs = int((80/800)*opt.OPTIM.NUM_EPOCHS)
    # warmup_epochs = 40
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    # CE_criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1]).to(device))
    # CE_criterion = torch.nn.CrossEntropyLoss()
    # DiceLoss = losses.DiceLoss().to(device)
    # Focal_loss = losses.focal_loss(alpha=None,gamma=2.,reduction='mean',ignore_index=opt.OPTIM.Focal_ignore_idx,device=device)

    print('===> model done')

    # grad_scaler = amp.GradScaler()

    # start_epoch = 0
    since = time.time()
    # best_val_loss = 0

    results_dict = {'epoch':[],
                    'train_loss':[],
                    'val_loss':[],
                    'lr':[],
                    }

    # opt.DRAW_EPOCHS = 
    epoch_split_n = int(opt.OPTIM.NUM_EPOCHS / opt.DRAW_EPOCHS)

    num_tokens = int(len(opt.DATASET_CUSTOME.LEADS)*(opt.DATASET_CUSTOME.INPUT_LENGTH // opt.DATASET_CUSTOME.PATCH_SIZE))
    leads_n_patch,n_patch_per_lead = get_leads_n_patch(num_tokens,opt.DATASET_CUSTOME.LEADS)

    for epoch in range(opt.OPTIM.NUM_EPOCHS):
    # for epoch in range(start_epoch, 2 + 1):
        epoch_start_time = time.time()
        epoch_train_loss=0

        epoch_train_corrects_lead,epoch_train_total_samples_lead = 0,0
        train_loss_time,train_loss_lead,train_loss_rescon = [],[],[]

        #### train ####
        model.train()
        for i, data in enumerate(tqdm(train_dataloader), 0):
            # if i >= 10:
            #     break
        # for i, data in enumerate(train_dataloader):
            inputs = data['input'].to(device)
            labels = data['label'].to(device) 
            input_lead_loc = data['input_lead_loc'].to(device) 
            # label_onehot = data['label_onehot'].to(device) 

            optimizer.zero_grad()
            torch.set_grad_enabled(True)

            # pred = model(inputs,mask_ratio=opt.DATASET_CUSTOME.MASK_RATIO)
            
            pred = model(inputs,
                            mask_ratio=opt.DATASET_CUSTOME.MASK_RATIO,
                            pure_ways=opt.MODEL.pure_ways,
                            fast_pure_ways=opt.MODEL.fast_pure_ways,
                            )

            loss = forward_loss(model, inputs, pred)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss_mean = epoch_train_loss / dataset_sizes['train']

        args = []
        args_old = []

        if isinstance(model, nn.DataParallel):
            orginals,input_masks,pred_masks = tran_mask(model.module,inputs,pred,mask=None)
        else:
            orginals,input_masks,pred_masks = tran_mask(model,inputs,pred,mask=None)


        for i in range(pred.shape[0]):
            if i > 5:
                break
            input_i = orginals[i,:,:]  #.tolist()
            pred_i = pred_masks[i,:,:]  #.tolist()
            mask_i = input_masks[i,:,:]  #.tolist()
            out_path_i='{}e{}_train_{}.png'.format(visu_dir,epoch,i)
            arg = [mask_i,pred_i,input_i,opt.DATASET_CUSTOME.LEADS,out_path_i,opt.DATASET_CUSTOME.PATCH_SIZE]
            args.append(arg)

        '''
        验证 每 n  个epoch一次
        '''
        if epoch % 5 != 0 and epoch != opt.OPTIM.NUM_EPOCHS-1:
            continue

        #### Evaluation ####
        model.eval()
        epoch_val_loss = 0
        epoch_val_corrects_lead,epoch_val_total_samples_lead = 0,0
        val_loss_time,val_loss_lead,val_loss_rescon = [],[],[]
        val_loss_rescon_lead_mask = []
        for val_b,data in enumerate(val_dataloader):
            # if val_b >= 2:
            #     break
            inputs = data['input'].to(device)
            labels = data['label'].to(device) 
            # label_onehot = data['label_onehot'].to(device) 
            data_file = data['data_file']
            input_lead_loc = data['input_lead_loc'].to(device) 

            # torch.set_grad_enabled(False)
            if val_b == len(val_dataloader)-1:
                register_hook = True
            else:
                register_hook = False

            # pred = model(inputs,mask_ratio=opt.DATASET_CUSTOME.MASK_RATIO)
            pred = model(inputs,
                            mask_ratio=opt.DATASET_CUSTOME.MASK_RATIO,
                            pure_ways='atten',
                            fast_pure_ways='atten',
                            )

            loss = forward_loss(model, inputs, pred)

            epoch_val_loss += loss.item()* inputs.size(0)

            idx_test = []
            for idx_i,data_file_i in enumerate(data_file):
                if data_file_i in test_paths:
                    idx_i = test_paths.index(data_file_i)
                    idx_test.append(idx_i)  

        val_loss_mean = epoch_val_loss / dataset_sizes['val']
        if isinstance(model, nn.DataParallel):
            orginals,input_masks,pred_masks = tran_mask(model.module,inputs,pred,mask=None)
        else:
            orginals,input_masks,pred_masks = tran_mask(model,inputs,pred,mask=None)

        for i in range(pred.shape[0]): 
            if len(args) > 19:
                break
            input_i = orginals[i,:,:]  #.tolist()
            pred_i = pred_masks[i,:,:]  #.tolist()
            mask_i = input_masks[i,:,:]  #.tolist()

            out_path_i='{}e{}_val_{}.png'.format(visu_dir,epoch,i)
            arg = [mask_i,pred_i,input_i,opt.DATASET_CUSTOME.LEADS,out_path_i,opt.DATASET_CUSTOME.PATCH_SIZE]
            args.append(arg)

        draw_epochs = [0]+[int(i) for i in range(0,opt.OPTIM.NUM_EPOCHS,3)]

        if epoch in draw_epochs:
            pool = mp.Pool(12)
            result = pool.map(draw_plt, args) 
            # result = pool.map(draw_plt_old, args_old[:30])
            pool.close()
            pool.join()


        save_path = model_dir+'model_epoch_{}_val_{:.6f}.pth'.format(epoch,val_loss_mean)

        # if torch.cuda.device_count() > 1: #DataParallel 带有 module, save时候要去掉
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model, save_path)

                    
        results_dict['epoch'].append(epoch)
        results_dict['train_loss'].append(train_loss_mean)
        results_dict['val_loss'].append(val_loss_mean)
        results_dict['lr'].append(scheduler.get_lr()[0])

        df = pd.DataFrame.from_dict(results_dict)

        # save_path = opt.SAVE_DIR+'loss.png'

        if epoch > 5:
            draw_two_line(results_dict['train_loss'],results_dict['val_loss'],opt.SAVE_DIR+'loss.png')
        draw_one_line(results_dict['lr'],opt.SAVE_DIR+'lr.png')

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}s \t train Loss: {:.6f} val loss: {:.6f} LearningRate {:.8f}".format(
                epoch, time.time()-epoch_start_time, train_loss_mean, val_loss_mean, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

    
        df = pd.DataFrame.from_dict(results_dict)
        # df = df.sort_values(by=['timeint'])
        df.to_csv(opt.SAVE_DIR+'results.csv',index=False)
        torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="train")
    parser.add_argument("-c", "--config", type=str, 
                        default=None,
                        help="path to yaml file")
    args = parser.parse_args()

    main(args.config,test_mode=False)

