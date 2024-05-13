# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:00:23 2024

@author: shepe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:07:14 2024

@author: shepe
"""

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d

class AOA_6(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOA_6, self).__init__()
        self.opt = opt
        self.fc_dim = 512
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.ctxR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.aspR_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.inputdim = 3 * 2 * opt.hidden_dim

        self.flatten = torch.nn.Flatten()
        self.lin1 = nn.Linear(self.inputdim, self.fc_dim)
        self.bn1 = BatchNorm1d(self.fc_dim)
        self.lin2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.lin3 = nn.Linear(self.fc_dim, opt.class_dim)
        
        self.lin1_c = nn.Linear(int(self.opt.constr_dim), 64)
        self.bn1_c = BatchNorm1d(64)
        self.lin2_c = nn.Linear(64, 64)
        self.lin3_c = nn.Linear(64, opt.class_dim)
        
        self.obtained = nn.Linear(opt.class_dim*2,opt.class_dim)


        
    def forward(self, inputs):
        text_raw_indices1 = inputs[0] 
        aspect_indices1 = inputs[1] 
        text_raw_indices2 = inputs[2]  
        aspect_indices2 = inputs[3]  
        
        constraints = inputs[4]
        
        
        ctx_len1 = torch.sum(text_raw_indices1 != 0, dim=1)
        asp_len1 = torch.sum(aspect_indices1 != 0, dim=1)
        ctx_len2 = torch.sum(text_raw_indices2 != 0, dim=1)
        asp_len2 = torch.sum(aspect_indices2 != 0, dim=1)
        ctx1 = self.embed(text_raw_indices1) 
        asp1 = self.embed(aspect_indices1) 
        ctx2 = self.embed(text_raw_indices2) 
        asp2 = self.embed(aspect_indices2) 

        # sentence 1
        ctx_out1, (_, _) = self.ctx_lstm(ctx1, ctx_len1.to("cpu")) 
        emb1 = torch.max(ctx_out1, 1)[0] 
        asp_out1, (_, _) = self.asp_lstm(asp1, asp_len1.to("cpu"))
        interaction_mat1 = torch.matmul(ctx_out1, torch.transpose(asp_out1, 1, 2)) 
        alpha1 = F.softmax(interaction_mat1, dim=1) 
        beta1 = F.softmax(interaction_mat1, dim=2) 
        beta_avg1 = beta1.mean(dim=1, keepdim=True) 
        gamma1 = torch.matmul(alpha1, beta_avg1.transpose(1, 2)) 
        weighted_sum1 = torch.matmul(torch.transpose(ctx_out1, 1, 2), gamma1).squeeze(-1) 

        #sentence 2
        ctx_out2, (_, _) = self.ctxR_lstm(ctx2, ctx_len2.to("cpu"))  
        emb2 = torch.max(ctx_out2, 1)[0]  
        asp_out2, (_, _) = self.aspR_lstm(asp2, asp_len2.to("cpu")) 
        interaction_mat2 = torch.matmul(ctx_out2, torch.transpose(asp_out2, 1, 2)) 
        alpha2 = F.softmax(interaction_mat2, dim=1) 
        beta2 = F.softmax(interaction_mat2, dim=2)  
        beta_avg2 = beta2.mean(dim=1, keepdim=True) 
        gamma2 = torch.matmul(alpha2, beta_avg2.transpose(1, 2)) 
        weighted_sum2 = torch.matmul(torch.transpose(ctx_out2, 1, 2), gamma2).squeeze(-1) 

        features =  torch.cat((torch.abs(emb1-emb2), weighted_sum1, weighted_sum2), 1)

        features = self.flatten(features)
        out_1 = F.relu(self.lin1(features))
        out_2 = self.dropout(out_1)
        out_3 = self.bn1(F.relu(self.lin2(out_2)))
        out_4 = self.lin3(out_3)

        out_1_c = self.lin1_c(constraints)
        out_2_c = self.bn1_c(self.lin2_c(out_1_c))
        out_3_c = self.lin3_c(out_2_c)
        
        out = self.obtained(torch.cat((out_4, out_3_c), 1))
        '''
        out_4 = F.softmax(out_4, 1)
        out_3_c = F.softmax(out_3_c, 1)
        out = torch.cat((out_4, out_3_c), dim=1)
        '''

        return out
