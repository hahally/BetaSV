from thop import profile
import torch
from model.repvgg import RepVGGStatsPool,repvgg_model_convert

model = RepVGGStatsPool(in_planes=64,embedding_size=256)
model = repvgg_model_convert(model, save_path='./tmp.model')
tensor = torch.rand(1, 80, 200)
# model(tensor)
flops, params = profile(model, (tensor,))
print('flops: ', flops/(1e9), 'params: ', params/1e6)
total = sum([param.nelement() for param in model.parameters()])
print('Number of params: %.2fM' % (total / 1e6))