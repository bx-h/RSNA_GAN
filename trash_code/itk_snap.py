import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from lungdataset import dcm_loader
from trash_code.use_unet import crop_roi


root = '/data15/boxi/anomaly detection/data/itksnap'
origin_path = root + '/move/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm'
mask_path = root + '/move_itk/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.nii.gz'

all_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    # convert a PIL or ndarray (H, W, C)[0, 255] to tensor(C, H, W)[0.0, 1.0]
    # ndarray dtype = uint8
])

def tensor_to_ndarray(tensor):
    img_arr = tensor.numpy() * 255
    img_arr = img_arr.astype('uint8')
    img_new = np.transpose(img_arr, (1, 2, 0))
    return img_new


mask_file = nib.load(mask_path).get_fdata()  # [h, w, c]
mask_file = all_to_tensor(mask_file).permute(0, 2, 1) # [c, h, w]


plt.subplot(1, 3, 1)
plt.imshow(tensor_to_ndarray(mask_file).squeeze(), cmap=plt.cm.bone)
plt.axis('off')


trsf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor()
])
origin_file = dcm_loader(origin_path)
origin_file = trsf(origin_file)

plt.subplot(1, 3, 2)
plt.imshow(tensor_to_ndarray(origin_file).squeeze(), cmap=plt.cm.bone)
plt.axis('off')

crop_img = crop_roi(origin_file.unsqueeze(0), mask_file.unsqueeze(0))
plt.subplot(1, 3, 3)
plt.imshow(crop_img.squeeze(), cmap=plt.cm.bone)
plt.axis('off')

plt.show()

