import torch
from model.repvgg import RepVGGStatsPool,repvgg_model_convert

# 模型部署时，重参数化

if __name__ == '__main__':
    model = RepVGGStatsPool(in_planes=64,embedding_size=256)
    checkpoint = torch.load('exp/Himia_80FBANK_RepVGG_A0_MQMHASTP_all_AAMsoftmax_256/model_39.pkl')
    model.load_state_dict(checkpoint['model'])
    deploy_model = repvgg_model_convert(model, save_path='exp/Himia_80FBANK_RepVGG_A0_MQMHASTP_all_AAMsoftmax_256/RepVGG-A0-deploy.pth')