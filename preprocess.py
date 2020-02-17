import torch
from torchvision import transforms, datasets
from Pneumonia import PneumoniaDataset, print_dataset
from use_unet import pack_up_all


root = "/data15/boxi/anomaly detection/data/"
transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]
)

dataset = PneumoniaDataset(root=root, train=False, transform=transform)
print_dataset(dataset, print_time=1000)

for index, (img, label) in enumerate(dataset):
    # try:
    rect1, rect2 = pack_up_all(img)
    print(rect1)
    print(rect2)
    # except Exception as e:
    #     print(e, index)
    #     continue
