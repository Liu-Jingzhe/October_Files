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

class GraphGate_V3(nn.Module):
    def __init__(self, config):
        super(GraphGate_V3,self).__init__()
        self.hidden_size = 384
        self.projector = nn.Linear(384, 384)
        self.gate = nn.Linear(384,1)
        self.pretrain  = config.pretrain_gate

    def forward(self, graph_emb,node_emb=None):     
        graph_emb = self.projector(graph_emb)
        gatevalues = self.gate(graph_emb)
        gatevalues = torch.squeeze(gatevalues)
        gatevalues = torch.softmax(gatevalues,dim=1)
        if self.pretrain:
            node_features = self.projector(node_emb)
            return gatevalues, node_features
        else:
            return gatevalues
        

class Att_Proj(nn.Module):
    def __init__(self):
        super(Att_Proj,self).__init__()
        self.att_proj = nn.Linear(384, 1)
        self.fwd_proj = nn.Linear(384,384)

    def forward(self,emb,label_emb):
        #print('emb',emb.shape)
        emb = torch.squeeze(emb)
        graph_weight = self.att_proj(emb)
        graph_weight = torch.squeeze(graph_weight)
        graph_weight = torch.softmax(graph_weight,dim=-1)
        graph_weight = torch.unsqueeze(graph_weight,dim=-1)
        graph_emb = emb*graph_weight

        graph_emb = torch.squeeze(torch.sum(graph_emb, dim=1))
        new_graph_emb = self.fwd_proj(graph_emb)
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        new_graph_emb = new_graph_emb / new_graph_emb.norm(dim=1, keepdim=True)
        graph_emb = graph_emb+new_graph_emb

        label_emb = label_emb[0]
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        label_emb = label_emb / label_emb.norm(dim=1, keepdim=True)
        logits = graph_emb @ label_emb.t()
        return graph_emb, logits
        


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

    #         graph_emb = torch.mean(graph_emb,dim=1,keepdim=False)
    #         graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
    #         label_emb = label_emb / label_emb.norm(dim=1, keepdim=True)
    #         logits = graph_emb @ label_emb.t()
    #         #print(logits[0])
    #         a = torch.argmax(logits[0])
    #         if a==label:
    #             total_true = total_true+1

    #         print(a)
    #         print(graph_emb.shape)
    # print(label_emb.shape)
    # print('total test sample num', len(questions))
    # print('acc', total_true/len(questions))
    return data_list, class_emb_list, label_list


from tqdm import tqdm


def train(model,loader,optimizer,device, args):
    model.train()
    loss_accum = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = loss_fct.to(device)
    for step, batch in enumerate(tqdm(loader)):
        data = batch[0].to(device)
        if data.shape[0]==1:
            continue
        classemb = batch[1].to(device)
        labels = batch[2].to(device)
        gb, logits = model(data,classemb)
        targets = torch.nn.functional.one_hot(labels,num_classes=len(classemb[0])).to(torch.float32).to(device)
        loss = loss_fct(logits, targets)
        loss_accum += loss.item()
        loss.backward()
        optimizer.step() 
        # if step>3:
        #     exit(0)
    train_loss = loss_accum/(step+1)
    print('Train Loss', train_loss)
    torch.save(model.att_proj.state_dict(),'./saved_model/'+args.dataset+'_'+args.task+'_att_proj.pt')
    print('Model saved ! In '+'./saved_model/'+args.dataset+'_'+args.task+'_att_proj.pt')
    torch.save(model.fwd_proj.state_dict(),'./saved_model/'+args.dataset+'_'+args.task+'_fwd_proj.pt')
    print('Model saved ! In '+'./saved_model/'+args.dataset+'_'+args.task+'_fwd_proj.pt')
    return train_loss
        
def test(model,test_loader, train_loader,device):
    model.eval()
    correct_num = 0
    avg_emb = 0
    for train_step, batch in enumerate(tqdm(train_loader)):
        data = batch[0].to(device)
        if data.shape[0]==1:
            continue
        classemb = batch[1].to(device)
        labels = batch[2].to(device)
        gb, logits = model(data,classemb)
        gb = torch.mean(gb,dim=0)
        avg_emb = avg_emb+gb
    avg_emb = avg_emb/train_step
    torch.save(avg_emb,'./saved_model/'+args.dataset+'_'+args.task+'_avg_emb.pt')
        


    for step, batch in enumerate(tqdm(test_loader)):
        data = batch[0].to(device)
        if data.shape[0]==1:
            continue
        classemb = batch[1].to(device)
        labels = batch[2].to(device)
        gb, logits = model(data,classemb)
        for i,label in enumerate(labels):
            pred = torch.argmax(logits[i])
            if label==pred:
                correct_num = correct_num+1
    print('correct_num',correct_num)
    acc = correct_num/(len(test_loader)*5)
    print('Total test sample', len(test_loader))
    print('Accuracy', acc)



def run(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    train_data, train_class_emb, train_label = prepare_data(args,mode='train')
    test_data, test_class_emb, test_label = prepare_data(args,mode='test')
    train_dataset = ProjDataset(train_data, train_class_emb, train_label)
    test_dataset = ProjDataset(test_data, test_class_emb, test_label)
    train_loader = DataLoader(train_dataset,batch_size=5)
    test_loader = DataLoader(test_dataset,batch_size=5)
    model = Att_Proj().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if not os.path.exists('./saved_model/'):
        os.makedirs('./saved_model/')
    train(model,train_loader,optimizer,device, args)
    test(model,test_loader,train_loader,device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
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
    parser.add_argument("--dataset", type=str, default="wikics")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="HO")
    parser.add_argument("--category", type=str, default="paper")
    args = parser.parse_args()

    run(args)
