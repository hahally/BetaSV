import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
# from datasets import load_dataset
from transformers import UniSpeechSatModel, UniSpeechSatConfig, AutoProcessor
from torch.utils.data import DataLoader
from model.pooling_layer import MQMHASTP
from multiLTdataset import MLTDataset
from model.ECAPA_TDNN import ECAPA_TDNN

class MLTmodel(nn.Module):
    def __init__(self,in_dim=768,hidden_dim=512,scale=8,bottleneck=128,embedding_size=192):
        super(MLTmodel, self).__init__()
        self.reweight = Parameter(torch.ones((12,1,1)))
        self.head_model = UniSpeechSatModel.from_pretrained("microsoft/UniSpeech-SAT-Base")
        self.freeze_model()
        self.down_model = ECAPA_TDNN(in_dim=in_dim, 
                                     hidden_dim=hidden_dim,
                                     scale=scale,
                                     bottleneck=bottleneck,
                                     embedding_size=embedding_size,
                                     asp='MQMHASTP')
        
        # self.down_model.ASP = MQMHASTP(in_dim=hidden_dim*3)
    # 冻结预训练模型
    def freeze_model(self):
        for name, parameter in self.head_model.named_parameters():
            parameter.requires_grad = False

    # 权重归一化
    def norm_weight(self, w):
        return F.softmax(w,dim=0)
    
    def forward(self, inputs):
        # 预训练语音模型
        input_feats = self.head_model(inputs,output_hidden_states=True)['hidden_states'][1:]
        # 提取预训练模型每一层隐藏层特征加权融合

        input_feats = torch.stack(input_feats,dim=1) * self.norm_weight(self.reweight)
        input_feats = torch.sum(input_feats,dim=1)
        # 下游模型 ECAPA_TDNN
        out_embedding = self.down_model(input_feats.transpose(2,1))
        # to do: add MQMHA

        return out_embedding

if __name__ == '__main__':
    from config.config_MLT import Config
    opt = Config()
    train_dataset = MLTDataset(opt=opt, train_mode=True)
    train_loader = DataLoader(train_dataset,
                      num_workers=opt.workers,
                      batch_size=opt.batch_size,
                      shuffle=True,
                      pin_memory=True)
    
    model = MLTmodel()
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    processor = AutoProcessor.from_pretrained("microsoft/UniSpeech-SAT-Base")
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    
    model(inputs['values'])
    
