import torch
from torch import nn
import torch.nn.functional as F

# Feature-level knowledge transfer 特征层次知识迁移
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss,self).__init__()
    
    def forward(self, emb_T, emb_S, labels):
        """
        emb_T: 教师网络提取的特征 B,dim
        emb_S: 学生网络提取的特征 B,dim
        """
        N = labels.size(0) # batch size 教师网络锚点个数，学生网络靶标个数
        # <f(xt_i),f(xs_j)> 内积 正样本相似度
        pos_scores = torch.exp(F.cosine_similarity(emb_T, emb_S,dim=1)) # N
        
        # 负样本相似度
        neg_scores = torch.zeros(N, device=emb_T.device)
        for i in range(N):
            label = labels[i]
            mask = labels!=label
            # 第 i 个锚点 负样本对
            similarity_matrix = F.cosine_similarity(emb_T[i].unsqueeze(0), emb_S,dim=1) # N
            neg_similarity_matrix = torch.masked_select(similarity_matrix, mask)
            neg_scores[i] = torch.sum(torch.exp(neg_similarity_matrix))
        
        Loss_F = -torch.sum(torch.log(pos_scores/neg_scores))/N
        
        return Loss_F
        
# Instance-level knowledge transfer 实例级别知识迁移
class InstanceLoss(nn.Module):
    def __init__(self):
        super(InstanceLoss,self).__init__()
        
    def forward(self, Et, Es):
        """
        Es: 源域说话人嵌入 B,dim
        Et: 目标域说话人嵌入 B,dim
        """
        ds = torch.mm(Es, Es.T)
        dt = torch.mm(Et, Et.T)
        LI = torch.norm(ds-dt,p=2)/(ds.shape[0]**2)
        
        return LI