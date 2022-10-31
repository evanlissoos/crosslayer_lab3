from lib import *
from lab import *

# model = load_model_fquant()
# fquant_to_csv(model, 'models/qint8_fused.model')

model = load_model_base()
model = prune_model_conv(model, 'conv1', 0.1, dim=0)
model = prune_model_conv(model, 'conv2', 0.15, dim=0)
test_model(model)

model = load_model_base()
print(model.conv1.weight.size())
print(model.conv1.bias.size())
print(model.conv2.weight.size())
print(model.conv2.bias.size())
print(model.lin2.weight.size())
print(model.lin2.bias.size())
conv1_prop = 1.0
conv2_prop = 1.0
print((((16*1*3*3 + 16) * conv1_prop + (32*16*3*3 + 32) * conv2_prop + 10*1568+10)*1)/1024)

# model = load_model_base()
# torch.backends.quantized.engine = 'qnnpack'
# model.qconfig = torch.ao.quantization.default_qconfig
# torch.ao.quantization.prepare(model, inplace=True)
# model = train_model(model, 0.001, 2)
# torch.ao.quantization.convert(model, inplace=True)
# test_model(model)