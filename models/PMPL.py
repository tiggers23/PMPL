from functools import partial
from models.vit import VisionTransformer

from transformers import BertModel, ViTModel
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MLP_adapter(nn.Module):
    # Non-Linear Transformation in the paper, acting as the translator between modalities.
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int, drop=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, in_dim//hidden_dim)
        self.drop = nn.Dropout(drop)
        self.linear2 = nn.Linear(in_dim//hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.norm(x)
        residual = self.linear1(residual)
        residual = self.relu(residual)
        residual = self.drop(residual)
        residual = self.linear2(residual)
        return residual

class cls_Gate(nn.Module):
    def __init__(self, hidden_size):
        super(cls_Gate, self).__init__()

        self.W1 = nn.Parameter(torch.ones(1, hidden_size))
        self.W2 = nn.Parameter(torch.ones(1, hidden_size))
        self.B1 = nn.Parameter(torch.ones(1, 1))
        self.norm = nn.LayerNorm(hidden_size)
        self._init_weight()

    def forward(self, x1, x2):
        alpha = torch.sigmoid(torch.mm(x1, self.W1.T) + self.B1 + torch.mm(x2, self.W2.T))
        return self.norm(x1*alpha + (1-alpha)*x2)
    
    def _init_weight(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.B1)

class Gate(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_sz):
        super(Gate, self).__init__()

        self.fc1 = MLP_adapter(hidden_size, mlp_hidden_sz, hidden_size)
        self.fc2 = MLP_adapter(hidden_size, mlp_hidden_sz, hidden_size)

        self.W1 = nn.Parameter(torch.ones(1, hidden_size))
        self.W2 = nn.Parameter(torch.ones(1, hidden_size))
        # self.W3 = nn.Parameter(torch.ones(1, hidden_size))
        self.B1 = nn.Parameter(torch.ones(1, 1))
        self.norm = nn.LayerNorm(hidden_size)
        self._init_weight()

    def forward(self, x1, x2, dis=None):
        x1 =self.fc1(x1)
        x2 =self.fc2(x2)

        alpha = torch.sigmoid(torch.matmul(x1, self.W1.T) + self.B1 + torch.matmul(x2, self.W2.T))
        if dis != None:
            alpha = dis.unsqueeze(1).unsqueeze(1) * alpha

        alpha = torch.clamp(alpha, 0.15, 0.85)
        return self.norm(x1*alpha + (1-alpha)*x2)
    
    def _init_weight(self):
        nn.init.kaiming_normal_(self.W1)
        nn.init.kaiming_normal_(self.W2)
        nn.init.kaiming_normal_(self.B1)
        
class triGate(nn.Module):
    def __init__(self, hidden_size, hidden_dim, two_gate=False):
        super(triGate, self).__init__()

        self.text_gate = Gate(hidden_size, hidden_dim)
        self.image_gate = Gate(hidden_size, hidden_dim)
        self.mm_gate = Gate(hidden_size, hidden_dim)
        self.two_gate = two_gate

    def forward(self, text_prompt, image_prompt, mm_memory=None, text_sim=None, image_sim=None):
        if self.two_gate:
            text_memory = self.text_gate(text_prompt, image_prompt, text_sim)
            image_memory = mm_memory
        elif mm_memory != None:
            text_memory = self.text_gate(text_prompt, mm_memory, text_sim)
            image_memory = self.image_gate(image_prompt, mm_memory, image_sim)
            image_memory = self.image_gate(image_prompt, mm_memory, image_sim)
        else:
            text_memory = text_prompt
            image_memory = image_prompt
        mm_memory = self.mm_gate(text_memory, image_memory)

        return mm_memory


class PMPL(nn.Module):
    def __init__(self,                 
                 args = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.args = args
        self.vit_encoder = ViTModel.from_pretrained('../models/google/vit-base-patch16-224-in21k')  
        self.bert_encoder = BertModel.from_pretrained('../models/bert-base-uncased')          

        if args.use_gate:
            self.cls_head = nn.Sequential(
                    #   nn.Linear(self.bert_encoder.config.hidden_size*2, self.bert_encoder.config.hidden_size*2),
                    #   nn.GELU(),
                    nn.Linear(self.bert_encoder.config.hidden_size, args.class_num)
                    )
            self.cls_head_1 = nn.Sequential(
                    #   nn.Linear(self.bert_encoder.config.hidden_size*2, self.bert_encoder.config.hidden_size*2),
                    #   nn.GELU(),
                    nn.Linear(self.bert_encoder.config.hidden_size, args.class_num)
                    )
        else:
            self.cls_head = nn.Sequential(
                    #   nn.Linear(self.bert_encoder.config.hidden_size*2, self.bert_encoder.config.hidden_size*2),
                    #   nn.GELU(),
                    nn.Linear(self.bert_encoder.config.hidden_size, args.class_num)
                    )            
            self.cls_head_1 = nn.Sequential(
                    #   nn.Linear(self.bert_encoder.config.hidden_size*2, self.bert_encoder.config.hidden_size*2),
                    #   nn.GELU(),
                    nn.Linear(self.bert_encoder.config.hidden_size, args.class_num)
                    )            

        self.image_fusion_prompt = nn.ParameterList([nn.Parameter(torch.empty(1, args.prompt_length, self.vit_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])
        self.text_fusion_prompt = nn.ParameterList([nn.Parameter(torch.empty(1, args.prompt_length, self.bert_encoder.config.hidden_size).normal_(std=0.02)) for _ in range(args.n_fusion_layers)])

        self.v2t_trans = nn.ModuleList([MLP_adapter(self.vit_encoder.config.hidden_size, args.mlp_hidden_sz, self.bert_encoder.config.hidden_size) for _ in range(args.n_fusion_layers)])
        self.t2v_trans = nn.ModuleList([MLP_adapter(self.bert_encoder.config.hidden_size, args.mlp_hidden_sz, self.vit_encoder.config.hidden_size) for _ in range(args.n_fusion_layers)])

        self.mm_memory = nn.Parameter(torch.empty(1, args.prompt_length, self.vit_encoder.config.hidden_size).normal_(std=0.02))

        self.text_linear = nn.Linear(self.bert_encoder.config.hidden_size, args.class_num)
        self.image_linear = nn.Linear(self.vit_encoder.config.hidden_size, args.class_num)

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_sim_self = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.gate = nn.ModuleList([triGate(self.bert_encoder.config.hidden_size, args.mlp_hidden_sz, args.two_gate) for _ in range(args.n_fusion_layers+1)])

        self.grad_control()

            
    def forward(self, image, text, labels, text_labels, image_labels, missing_image_label, epoch, use_caloss=False, Train=True,temp_labels=None):
        ca_loss = 0
        image_cls_loss = 0
        text_cls_loss = 0
        n = image.shape[0]
        device = image.device
        txt_input_ids = text.input_ids
        txt_token_type_ids = torch.zeros(txt_input_ids.size(), dtype=torch.long, device=device)
        txt_attn_mask = text.attention_mask

        img_tokens = self.vit_encoder.embeddings(image)
        txt_tokens = self.bert_encoder.embeddings(txt_input_ids, token_type_ids=txt_token_type_ids)

        image_length = img_tokens.shape[1]
        text_length = txt_tokens.shape[1]

        txt_attn_mask = self.get_extended_txt_attn_mask(txt_attn_mask)
        max_prompt_length = self.bert_encoder.config.num_hidden_layers * ( self.args.prompt_length + self.args.n_fusion + self.args.prompt_length )
        batch_extra_attn_mask = torch.ones(n, max_prompt_length).to(device)
        batch_extra_attn_mask = self.get_extended_txt_attn_mask(batch_extra_attn_mask)

        encoder_txt_attn_mask = txt_attn_mask
        bert_layer_id = 0
        vit_layer_id = 0
        for bert_layer_id in range(self.bert_encoder.config.num_hidden_layers-self.args.n_fusion_layers):
            txt_tokens = self.bert_encoder.encoder.layer[bert_layer_id](txt_tokens, encoder_txt_attn_mask)[0]

        for vit_layer_id in range(self.vit_encoder.config.num_hidden_layers-self.args.n_fusion_layers):
            img_tokens = self.vit_encoder.encoder.layer[vit_layer_id](img_tokens)[0]
        
        text_fusion = txt_tokens[:,0,:].unsqueeze(1).repeat(1,self.args.prompt_length,1)
        image_fusion = img_tokens[:,0,:].unsqueeze(1).repeat(1,self.args.prompt_length,1)
        mm_memory = self.gate[0](text_fusion, image_fusion, self.mm_memory)
        text_fusion = mm_memory
        image_fusion = mm_memory
        for fusion_layer_id in range(self.args.n_fusion_layers):
            batch_image_fusion_prompt = self.image_fusion_prompt[fusion_layer_id].expand(n, -1, -1).to(device)
            batch_text_fusion_prompt = self.text_fusion_prompt[fusion_layer_id].expand(n, -1, -1).to(device)
            layer_t2v_prompt_attn_mask = batch_extra_attn_mask[:,:,:,:self.args.prompt_length]

            if self.args.use_adapter:
                t2v_fusion_intermediate = self.t2v_trans[fusion_layer_id](text_fusion)
                v2t_fusion_intermediate = self.v2t_trans[fusion_layer_id](image_fusion)
                if self.args.use_prompt:
                    fusion_txt_attn_mask = torch.cat([encoder_txt_attn_mask, layer_t2v_prompt_attn_mask, layer_t2v_prompt_attn_mask], dim=3)
                    img_tokens = torch.cat([img_tokens, batch_image_fusion_prompt, t2v_fusion_intermediate], dim=1)
                    txt_tokens = torch.cat([txt_tokens, batch_text_fusion_prompt, v2t_fusion_intermediate], dim=1)
                else:
                    fusion_txt_attn_mask = torch.cat([encoder_txt_attn_mask, layer_t2v_prompt_attn_mask], dim=3)
                    img_tokens = torch.cat([img_tokens, t2v_fusion_intermediate], dim=1)
                    txt_tokens = torch.cat([txt_tokens, v2t_fusion_intermediate], dim=1)
            else:
                t2v_fusion_intermediate = None
                v2t_fusion_intermediate = None
                if self.args.use_prompt:
                    fusion_txt_attn_mask = torch.cat([encoder_txt_attn_mask, layer_t2v_prompt_attn_mask], dim=3)
                    img_tokens = torch.cat([img_tokens, batch_image_fusion_prompt], dim=1)
                    txt_tokens = torch.cat([txt_tokens, batch_text_fusion_prompt], dim=1)
                else:
                    fusion_txt_attn_mask = encoder_txt_attn_mask

            img_token = self.vit_encoder.encoder.layer[vit_layer_id + fusion_layer_id + 1](img_tokens)[0]
            txt_token = self.bert_encoder.encoder.layer[bert_layer_id + fusion_layer_id +1](txt_tokens, fusion_txt_attn_mask)[0]

            txt_tokens = txt_token[:, :text_length, :]
            img_tokens = img_token[:, :image_length, :]
            text_fusion = txt_token[:,0,:].unsqueeze(1).repeat(1,self.args.prompt_length,1)
            image_fusion = img_token[:,0,:].unsqueeze(1).repeat(1,self.args.prompt_length,1)
            if self.args.use_layer_gate:
                if use_caloss:
                    with torch.no_grad():
                        text_logits = self.text_linear(txt_token[:,0,:])
                        image_logits = self.image_linear(img_token[:,0,:])
                        mm_logits = self.cls_head(torch.mean(mm_memory,dim=1))

                        text_dis = (self.cos_sim(text_logits, mm_logits) + 1)
                        image_dis = (self.cos_sim(image_logits, mm_logits) + 1)
                else:
                    text_dis = None
                    image_dis = None

                mm_memory = self.gate[fusion_layer_id+1](text_fusion, image_fusion, mm_memory, text_dis, image_dis)
                text_fusion = mm_memory
                image_fusion = mm_memory
            elif not self.args.all_cat:
                mm_memory = text_fusion + mm_memory + image_fusion+mm_memory
                text_fusion = mm_memory
                image_fusion = mm_memory
            elif not self.args.two_gate:
                mm_memory = self.gate[fusion_layer_id+1](text_fusion, image_fusion, mm_memory)
                text_fusion = mm_memory
                image_fusion = mm_memory

        if self.args.setting!='multimodal' and Train:
            if epoch > 0:
                text_golbal_memory = {}
                text_golbal_not_memory = {}
                text_golbal_center = []
                text_golbal_not_center = []

                img_golbal_memory = {}
                img_golbal_not_memory = {}
                img_golbal_center = []
                img_golbal_not_center = []
 
                for i in range(self.args.class_num):
                    text_golbal_memory[i] = []
                    img_golbal_memory[i] = []
                    text_golbal_not_memory[i] = []
                    img_golbal_not_memory[i] = []
                for i in range(mm_memory.shape[0]):
                    for j in range(self.args.class_num):
                        if (self.args.dataset=='mmimdb'and labels[i][j] == 1) or (self.args.dataset!='mmimdb' and labels[i]==j):
                            text_golbal_memory[j].append(txt_tokens[i,0,:].unsqueeze(0))
                            img_golbal_memory[j].append(img_tokens[i,0,:].unsqueeze(0))
                        else:
                            text_golbal_not_memory[j].append(txt_tokens[i,0,:].unsqueeze(0))
                            img_golbal_not_memory[j].append(img_tokens[i,0,:].unsqueeze(0))

                for i in range(self.args.class_num):
                    if len(text_golbal_memory[i])==0:
                        text_golbal_center.append(torch.zeros(1,text_fusion.shape[2]).to(text_fusion.device))
                        img_golbal_center.append(torch.zeros(1,text_fusion.shape[2]).to(text_fusion.device))
                    else:
                        text_golbal_center.append(torch.mean(torch.cat(text_golbal_memory[i],dim=0),dim=0).unsqueeze(0))
                        img_golbal_center.append(torch.mean(torch.cat(img_golbal_memory[i],dim=0),dim=0).unsqueeze(0))
                    if len(text_golbal_not_memory[i])==0: 
                        text_golbal_not_center.append(torch.zeros(1,text_fusion.shape[2]).to(text_fusion.device))
                        img_golbal_not_center.append(torch.zeros(1,text_fusion.shape[2]).to(text_fusion.device))
                    else:
                        text_golbal_not_center.append(torch.mean(torch.cat(text_golbal_not_memory[i],dim=0),dim=0).unsqueeze(0))
                        img_golbal_not_center.append(torch.mean(torch.cat(img_golbal_not_memory[i],dim=0),dim=0).unsqueeze(0))

                text_golbal_center = torch.cat(text_golbal_center, dim=0).unsqueeze(0).repeat(text_fusion.shape[0],1,1)
                text_golbal_not_center = torch.cat(text_golbal_not_center, dim=0).unsqueeze(0).repeat(text_fusion.shape[0],1,1)                
                img_golbal_center = torch.cat(img_golbal_center, dim=0).unsqueeze(0).repeat(text_fusion.shape[0],1,1)
                img_golbal_not_center = torch.cat(img_golbal_not_center, dim=0).unsqueeze(0).repeat(text_fusion.shape[0],1,1)                

                text_features = txt_token[:,0,:].unsqueeze(1).repeat(1,self.args.class_num,1)
                image_features = img_token[:,0,:].unsqueeze(1).repeat(1,self.args.class_num,1)

                text_distance = self.cos_sim_self(text_features, text_golbal_center)
                text_not_distance = self.cos_sim_self(text_features, text_golbal_not_center)
                text_alpha = text_distance - text_not_distance

                image_distance = self.cos_sim_self(image_features, img_golbal_center)
                image_not_distance = self.cos_sim_self(image_features, img_golbal_not_center)
                image_alpha = image_distance - image_not_distance
                
                t_labels = temp_labels + text_alpha
                text_labels = torch.clamp(epoch / (epoch + 2)* text_labels + 2/(epoch+2)*t_labels,0,1)

                i_labels = temp_labels + image_alpha
                image_labels = torch.clamp(epoch / (epoch + 2)* image_labels + 2/(epoch+2) * i_labels,0,1)
            else:
                text_labels = text_labels
                image_labels = image_labels

            image_cls_loss = F.binary_cross_entropy_with_logits(self.image_linear(torch.mean(image_fusion,dim=1)), image_labels.float())
            text_cls_loss = F.binary_cross_entropy_with_logits(self.text_linear(torch.mean(text_fusion,dim=1)), text_labels.float())
        elif Train:
            image_cls_loss += F.cross_entropy(self.text_linear(txt_token[:,0,:]), text_labels.long())
            text_cls_loss += F.cross_entropy(self.image_linear(img_token[:,0,:]), image_labels.long())

        if self.args.use_layer_gate:
            gate_fusion = torch.mean(mm_memory,dim=1)
        else:
            gate_fusion = torch.mean(torch.cat([txt_tokens[:,0,:].unsqueeze(1), img_tokens[:,0,:].unsqueeze(1)],dim=1), dim=1)

        prediction = self.cls_head(gate_fusion)
        if Train:
            if self.args.dataset == 'mmimdb':
                loss = F.binary_cross_entropy_with_logits(prediction, labels)
            else:
                loss = F.cross_entropy(prediction, labels.long())
            return {'loss':loss,'logits':prediction, 'ca_loss':ca_loss, 'image_cls_loss':image_cls_loss, 'text_cls_loss':text_cls_loss, 'text_labels':text_labels, 'image_labels':image_labels}
        else:
            return {'logits':prediction}
 
    def get_extended_txt_attn_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def grad_control(self):
        trainable_modules = [
                            self.image_fusion_prompt, self.text_fusion_prompt,
                            self.v2t_trans.modules(), self.t2v_trans.modules(),
                            self.text_linear.modules(), self.image_linear.modules(),
                            self.cls_head.modules(),
                            self.mm_memory,
                            self.gate.modules(),
                            ]

        for module in self.modules():
            module.requires_grad_(False)

        for module in trainable_modules:
            for item in module:
                item.requires_grad_(True)  

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient   