import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from scipy import ndimage

from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental, Vgg19Experimental

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

# Image manipulation util functions

def load_image(img_path, target_shape=None, invert=False):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img

def prepare_img(img_path, target_shape, device, invert=False):
    img = load_image(img_path, target_shape=target_shape, invert=invert)

    # normalize using ImageNet's mean
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img

def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])  # [:, :, ::-1] converts rgb into bgr (opencv constraint...)

def generate_out_img_name(config):
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    if 'reconstruct_script' in config:
        suffix = f'_o_{config["optimizer"]}_h_{str(config["height"])}_m_{config["model"]}{config["img_format"][1]}'
    else:
        suffix = f'_o_{config["optimizer"]}_i_{config["init_method"]}_h_{str(config["height"])}_m_{config["model"]}_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["img_format"][1]}'
    return prefix + suffix

def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations, should_display=False):
    saving_freq = config['saving_freq']
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

    # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
    if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()

def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')

# End of image manipulation util functions

def prepare_model(model):
    experimental = True
    if model == 'vgg16':
        if experimental:
            model = Vgg16Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg16(requires_grad=False, show_progress=True)
    elif model == 'vgg19':
        if experimental:
            model = Vgg19Experimental(requires_grad=False, show_progress=True)
        else:
            model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')
    
    return model

def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

class GramMSELoss(torch.nn.Module):
    def forward(self, input, target):
        out = torch.nn.MSELoss()(gram_matrix(input), target)
        return(out)

def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rgb_to_grayscale(img):
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError(f'Expected img with shape (3, H, W), but got {img.shape}')
    r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.unsqueeze(0)

def dist_cv2(img, device):
    img = img.squeeze(axis=0)
    # convert dimension (3, 256, 256) to (256, 256, 3)
    img = np.transpose(img.to('cpu').detach().numpy(), (1, 2, 0))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.blur(img, (1, 1))
    img = np.asarray(img, dtype=np.uint8)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    img = np.asarray(img, dtype=np.uint8)
    cv.imwrite('test-1.png', img)
    img = ndimage.grey_erosion(img, size=(3, 3))

    img_dist = cv.distanceTransform(img, cv.DIST_L2, 5)
    cont_dist = torch.from_numpy(img_dist).float().to(device)

    f = cont_dist.unsqueeze(0)
    a = torch.cat([f, f, f], 0)
    cv.imwrite('test-2.png', np.transpose(a.to('cpu').detach().numpy() * 255, (1, 2, 0)))
    a = a.unsqueeze(0)
    return a
