from lib import *
from lab import *

model = load_model_fquant()
layer_to_csv(model.conv1, 'outputs/qint8_conv1.csv')
layer_to_csv(model.conv2, 'outputs/qint8_conv2.csv')
layer_to_csv(model.lin2, 'outputs/qint8_lin2.csv')