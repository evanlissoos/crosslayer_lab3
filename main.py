from lib import *
from lab import *
import struct

model = load_model_base()
# test_model(model)
model.eval()
torch.backends.quantized.engine = 'qnnpack'
model.qconfig = torch.ao.quantization.default_qconfig
model = torch.quantization.fuse_modules(model, [['conv1', 'bn1'], ['conv2', 'bn2']])
model = quantize_model(model, 16, 100)
# test_model(model)
# layer_to_csv(model.conv1, 'conv1.csv')
# layer_to_csv(model.conv2, 'conv2.csv')
# layer_to_csv(model.lin2, 'lin2.csv')
# print(model.scale)

# Read from accelerator test data
images = np.empty((32,1,28,28), dtype=float)
labels = np.empty((32), dtype=int)
image_size = 28*28
with open ('accel/fashion_mnist.bin', 'rb') as f:
    for i in range(32):
        images[i][0] = np.asarray(struct.unpack('f'*image_size, f.read(4*image_size))).reshape(28,28)
        labels[i] = int.from_bytes(f.read(1), 'little')
        # plt.imshow(images[-1])
        # plt.show()

images = torch.tensor(images).float()
labels = torch.tensor(labels)

def test_single(model, images, labels):
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)
    model = model.to(device)
    correct = 0
    total = 0
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    testloss = criterion(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    print('Accuracy for test images: % d %%' % (100 * correct / total))
    accuracy = 100*correct/total
    accuracy.to('cpu')
    return accuracy.item()

test_single(model, images, labels)
