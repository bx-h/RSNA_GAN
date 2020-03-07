from torchvision import transforms
from trash_code.Pneumonia import PneumoniaDataset
from trash_code.use_unet import pack_up_all, AddContrast

root = "/data15/boxi/anomaly detection/data/"

class CropLung(object):
    def __call__(self, sample):
        rect1, rect2 = pack_up_all(sample)
        return {'rect1': rect1, 'rect2': rect2, 'img': sample}

transform = transforms.Compose(
    [
        AddContrast(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        CropLung()
    ]
)

dataset = PneumoniaDataset(root=root, train=False, transform=transform)

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#
# for i_batched, sample_batched in enumerate(dataloader):
#     print(i_batched, sample_batched['rect1'], sample_batched['rect2'], sample_batched['img'].size())
#
#     if i_batched == 3:
#         plt.figure()
#         plt.imshow(sample_batched['img'])
#         plt.show()
#         break

i, target, path = dataset[0]
print(i)