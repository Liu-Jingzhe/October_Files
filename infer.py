import os.path as osp
import sys
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import os.path as osp
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader, dataset, Dataset
from torch import nn
import torch.optim as optim
import warnings
from torch import linalg as LA
warnings.filterwarnings("ignore")


class ProjDataset(Dataset):

    def __init__(self, data, classemb,labels):
        self.data = data
        self.classemb = classemb
        self.labels = labels
 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.classemb[idx],self.labels[idx]



class Expert(nn.Module):
    def __init__(self):
        super(Expert,self).__init__()
        self.att_proj = nn.Linear(384, 1)
        self.fwd_proj = nn.Linear(384,384)

    def forward(self,emb):
        #print('emb',emb.shape)
        emb = torch.squeeze(emb)
        #new_emb = self.fwd_proj(emb)
        graph_weight = self.att_proj(emb)
        #print(graph_weight.shape)
        graph_weight = torch.squeeze(graph_weight)
        graph_weight = torch.softmax(graph_weight,dim=-1)
        graph_weight = torch.unsqueeze(graph_weight,dim=-1)
        #print('graph weight', graph_weight[0])
        #graph_emb = (emb+new_emb)*graph_weight
        graph_emb = emb*graph_weight
        #print('graph emb',graph_emb.shape)
        graph_emb = torch.sum(graph_emb, dim=0, keepdim=True)
        #print('graph emb',graph_emb.shape)
        new_graph_emb = self.fwd_proj(graph_emb)
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        new_graph_emb = new_graph_emb / new_graph_emb.norm(dim=1, keepdim=True)
        #graph_emb = graph_emb+new_graph_emb*0.2
        # label_emb = label_emb[0]
        # #print(label_emb.shape)
        # graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        # label_emb = label_emb / label_emb.norm(dim=1, keepdim=True)
        # logits = graph_emb @ label_emb.t()
        return graph_emb, new_graph_emb
    
class Gate(nn.Module):
    def __init__(self):
        super(Gate,self).__init__()
        self.gate = nn.Sequential(nn.Linear(384,1),nn.Sigmoid())
        #self.gate_sig = nn.Sigmoid()

    def forward(self,emb):
        emb = self.gate(emb)
        #gate_value = self.gate_sig(emb)
        gate_value = torch.mean(emb)
        return gate_value

class Att_ProjwithGate(nn.Module):
    def __init__(self):
        super(Att_ProjwithGate,self).__init__()
        self.att_proj = nn.Linear(384, 1)
        self.fwd_proj = nn.Linear(384,384)
        self.gate = nn.Linear(384,384)
        #self.gate_sig = nn.Sigmoid()

    def forward(self,emb):
        #print('emb',emb.shape)
        emb = torch.squeeze(emb)
        
        mask_emb = self.gate(emb)
        aug_emb = emb+mask_emb


        graph_weight = self.att_proj(aug_emb)
        graph_weight = torch.squeeze(graph_weight)
        graph_weight = torch.softmax(graph_weight,dim=-1)
        graph_weight = torch.unsqueeze(graph_weight,dim=-1)
        graph_emb = emb*graph_weight
        #print(graph_weight)
        graph_emb = torch.sum(graph_emb, dim=0, keepdim=True)
        new_graph_emb = self.fwd_proj(graph_emb)
        #print('1',graph_emb)
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        #print('2',graph_emb)
        new_graph_emb = new_graph_emb / new_graph_emb.norm(dim=1, keepdim=True)
        aug_graph_emb = graph_emb+new_graph_emb

        # graph_weight = self.att_proj(mask_emb)
        # graph_weight = torch.squeeze(graph_weight)
        # graph_weight = torch.softmax(graph_weight,dim=-1)
        # graph_weight = torch.unsqueeze(graph_weight,dim=-1)
        # graph_emb = emb*graph_weight
        # graph_emb = torch.squeeze(torch.sum(graph_emb, dim=1))
        # new_graph_emb = self.fwd_proj(graph_emb)
        # graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        # new_graph_emb = new_graph_emb / new_graph_emb.norm(dim=1, keepdim=True)
        # mask_graph_emb = graph_emb+new_graph_emb
        

        return aug_graph_emb


def set_label_names(data, label_csv_path):
    label_pd = pd.read_csv(label_csv_path)
    if hasattr(data, 'label_names'):
        return data 
    label_names = label_pd['name'].tolist()
    data.label_names = label_names
    return data

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt")) for i in range(hop+1)]
    return pretrained_embs

class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def prepare_data(args, mode='train'):
    data_path = osp.join(args.data_saved_path, args.dataset, 'processed', "geometric_data_processed.pt")
    data_dir = osp.join(args.data_saved_path, args.dataset, 'processed')
    data_list = []
    class_emb_list = []
    label_list = []
    if args.task in  ["nc"]:
        if mode=='train':
            prompt_file = os.path.join(data_dir, f"sampled_2_10_train.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"sampled_2_10_test.jsonl")
        label_emb = torch.load(os.path.join(data_dir, f"sb_node_label_emb.pt"))
        
        
    elif args.task in ["lp"]:
        prompt_file = os.path.join(data_dir, f"edge_sampled_2_10_only_test.jsonl")
        label_emb = torch.load(os.path.join(data_dir, f"sb_lp_label_emb.pt"))
    else:
        raise ValueError
    data = torch.load(data_path)[0]
    if hasattr(data, "train_masks") and 'citeseer' not in args.dataset:
        data.train_mask = data.train_masks[0]
        data.val_mask = data.val_masks[0]
        data.test_mask = data.test_masks[0]
    
        del data.train_masks
        del data.val_masks
        del data.test_masks
    elif args.dataset == 'arxiv':
        arxiv_mask = torch.load(osp.join(args.data_saved_path, args.dataset, 'processed', 'arxiv_mask.pt'))
        data.train_mask = arxiv_mask['train']
        data.val_mask = arxiv_mask['valid']
        data.test_mask = arxiv_mask['test']
    set_label_names(data, osp.join(args.data_saved_path, args.dataset, 'processed','categories.csv'))
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]

    index = None
    n = data.num_nodes
    if args.task == "lp":
        ## implement later
        pass 
    else:
        mask = torch.full([n], fill_value=False, dtype=torch.bool)
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
    total_true = 0    
    for line in tqdm(questions):
        #print(line)
        idx = line["id"]
        label = data.y[idx]
        #print(label)
        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        if args.task == "lp":
            mp = MP()
            center_nodes = []
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id] * (args.use_hop + 1)
                center_nodes.append(center_id)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_embs = [pretrained_emb[center_id].cuda()]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index,
                                                                    relabel_nodes=True)
            local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
                        (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
            edge_index = edge_index[:, ~local_edge_mask]
            local_x = pretrained_emb[subset].cuda()
            n = subset.shape[0]
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index)
            edge_index = edge_index.cuda()
            row, col = edge_index
            deg = degree(col, n, dtype=pretrained_emb.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # local_x = pretrained_emb
            for _ in range(args.use_hop):
                local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                graph_embs.append(local_x[mapping])
            graph_emb = torch.stack(graph_embs, dim=1)
        else:
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id]*(args.use_hop+1)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[center_id] for emb in pretrained_emb], dim=1)
            data_list.append(graph_emb)
            class_emb_list.append(label_emb)
            label_list.append(int(label))

    return data_list, class_emb_list, label_list

def infer(model_dict,loader,device,args):
    for i in range(len(model_dict['Expert'])):
        model_dict['Expert'][i].eval()
        model_dict['Gate'][i].eval()
    correct_num = 0
    for step, batch in enumerate(tqdm(loader)):
        data = batch[0].to(device)
        classemb = batch[1].to(device)
        labels = batch[2].to(device)
        graph_embs = []
        new_graph_embs = []
        gate_values = []
        data_copy = torch.squeeze(data)
        data_copy = torch.mean(data_copy,dim=0,keepdim=True)
        for i in range(len(model_dict['Expert'])):

            aug_graph_emb = model_dict['Gate'][i](data)
            avg_emb = torch.unsqueeze(model_dict['avg_emb'][i],dim=0)
            data_copy= aug_graph_emb/ aug_graph_emb.norm(dim=1, keepdim=True)
            avg_emb = avg_emb / avg_emb.norm(dim=1, keepdim=True)
            value = data_copy @avg_emb.t()
            #print(value)
            
            #value = 1/LA.matrix_norm(aug_graph_emb-model_dict['avg_emb'][i])
            #graph_embs.append(emb)
            #new_graph_embs.append(new_emb)
            gate_values.append(value)

        #logits, recon_logits = model(data,classemb)
        gate_values = torch.tensor(gate_values).to(device)
        
        #print(gate_values)
        values, indices = torch.topk(gate_values,args.num_experts)
        values = torch.nn.functional.softmax(values)
        
        #print(values)
        graph_features = None
        print('----------')
        for v,i in zip(values,indices):
            print(model_dict['Name'][i],v)
            emb, new_emb = model_dict['Expert'][i](data)
            v = v.unsqueeze(0).repeat(emb.shape[0],1)
            if graph_features is None:
                graph_features = new_emb*v+emb
            else:
                graph_features = graph_features + new_emb*v+emb
        graph_features = graph_features.to(device)
        #print('graph_features',graph_features.shape)
        graph_emb = torch.unsqueeze(torch.sum(graph_features , dim=0),dim=0)
        #print('graph_emb',graph_emb.shape)
        label_emb = classemb[0]
        #print('label_emb',label_emb.shape)
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        label_emb = label_emb / label_emb.norm(dim=1, keepdim=True)
        logits = graph_emb @ label_emb.t()
        for i,label in enumerate(labels):
            pred = torch.argmax(logits[i])
            if label==pred:
                correct_num = correct_num+1
            print('pred',pred)
            print('label',label)
        if step>5000:
            break
    print('correct_num',correct_num)
    #acc = correct_num/(len(loader))
    acc = correct_num/step
    #print('Total test sample', len(loader))
    print('Total test sample', step)
    print('Accuracy', acc)
    return

def run(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #train_data, train_class_emb, train_label, train_nodes, train_adjs = prepare_data(args,mode='train')
    test_data, test_class_emb, test_label = prepare_data(args,mode='test')
    #train_dataset = ProjDataset(train_data, train_class_emb, train_label, train_nodes, train_adjs)
    test_dataset = ProjDataset(test_data, test_class_emb, test_label)
    #train_loader = DataLoader(train_dataset,batch_size=5)
    test_loader = DataLoader(test_dataset,batch_size=1)
    model_dict = {'Expert':[],'Gate':[],'avg_emb':[],'Name':[]}
    #dataset_list = ['bookchild','bookhis','cora','citeseer','dblp','products','pubmed','sportsfit','wikics']
    dataset_list = ['cora','citeseer','dblp','bookchild','bookhis','products','pubmed','sportsfit','wikics']
    if args.dataset in dataset_list:
        dataset_list.remove(args.dataset)
    for dataname in dataset_list:
        expert = Expert().to(device)
        gate = Att_ProjwithGate().to(device)
        

        att_proj_weights = torch.load('./saved_model/'+dataname+'_'+args.task+'_att_proj.pt', map_location='cpu')
        expert.att_proj.load_state_dict(att_proj_weights)
        

        fwd_proj_weights = torch.load('./saved_model/'+dataname+'_'+args.task+'_fwd_proj.pt', map_location='cpu')
        expert.fwd_proj.load_state_dict(fwd_proj_weights)

        gate_weights = torch.load('./saved_model/'+dataname+'_'+args.task+'_gate.pt', map_location='cpu')
        gate.gate.load_state_dict(gate_weights)
        gate.att_proj.load_state_dict(att_proj_weights)
        gate.fwd_proj.load_state_dict(fwd_proj_weights)

        model_dict['Expert'].append(expert)
        model_dict['Gate'].append(gate)

        
        # data_dir = osp.join(data_saved_path, dataname, 'processed')
        # pretrained_embs = torch.load(os.path.join(data_dir, "sbert_0hop_x.pt"))
        # avg_emb = torch.mean(pretrained_embs,dim=0)


        avg_emb = torch.load('./saved_model/'+dataname+'_'+args.task+'_avg_emb.pt', map_location='cpu')
        avg_emb = avg_emb.to(device)
        model_dict['avg_emb'].append(avg_emb)
        model_dict['Name'].append(dataname)
    
    infer(model_dict,test_loader,device, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--use_moe_gate", type=int, default=1)
    parser.add_argument("--data_saved_path", type=str, default="cache_data_minilm")
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=4)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="products")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="HO")
    parser.add_argument("--category", type=str, default="none")
    args = parser.parse_args()

    run(args)
