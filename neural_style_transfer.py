import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import cv2 as cv
import argparse

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def edge_detection(img):
    if img.dim() == 4:  # If input is (batch_size, channels, height, width)
        img = img.squeeze(0)  # Remove batch dimension if exists

    if img.dim() == 3 and img.size(0) == 3:  # (channels, height, width)
        img_gray = rgb_to_grayscale(img).unsqueeze(0)  # Convert to grayscale and add channel dimension
    elif img.dim() == 2:  # (height, width)
        img_gray = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    else:
        raise ValueError(f'Unexpected img shape: {img.shape}')

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
    
    grad_x = torch.nn.functional.conv2d(img_gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(img_gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0)  # Remove channel dimension
    return edge

def rgb_to_grayscale(img):
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError(f'Expected img with shape (3, H, W), but got {img.shape}')
    r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.unsqueeze(0)

def resize_mask(mask, target):
    b, c, h, w = target.size()
    resized_mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return resized_mask.expand(b, c, h, w)

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation, target_style_representation = target_representations

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        if gram_gt.size() == gram_hat.size():
            style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    mrf_loss = torch.tensor(0., requires_grad=True).to(optimizing_img.device)
    dist_loss = torch.tensor(0., requires_grad=True).to(optimizing_img.device)

    content_loss *= config['content_weight']
    style_loss *= config['style_weight']
    mrf_loss *= config['mrf_weight']
    dist_loss *= config['dist_weight']

    # Add L2 regularization
    l2_reg = torch.tensor(0., requires_grad=True).to(optimizing_img.device)
    for param in neural_net.parameters():
        l2_reg += torch.norm(param)
    
    total_loss = content_loss + style_loss + 0.001*l2_reg

    return total_loss, content_loss, style_loss, mrf_loss, dist_loss

def neural_style_transfer(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dump_path = config['output_img_dir']
    utils.ensure_exists(dump_path)

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    content_img = utils.prepare_img(content_img_path, config['height'], device, config['content_invert'])
    style_img = utils.prepare_img(style_img_path, config['height'], device, config['style_invert'])

    # create distance module path
    cont_dist = utils.dist_cv2(content_img.clone(), device)
    cont_dist = cont_dist**6
    cont_dist[cont_dist>1e3] = 1e3
    cont_dist[cont_dist==float("Inf")] = 1e3
    dist_template = cont_dist*content_img

    cv.imwrite('dist_template.png', np.transpose(dist_template.to('cpu').detach().numpy().squeeze(axis=0), (1, 2, 0)))

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = (target_content_representation, target_style_representation)
    
    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        gaussian_noise_img = torch.from_numpy(gaussian_noise_img).to(device)
        # random_noise_img = torch.randn(content_img.size()).to(device) * 0.256
        optimizing_img = Variable(gaussian_noise_img, requires_grad=True)
    elif config['init_method'] == 'content':
        optimizing_img = Variable(content_img.clone().to(device), requires_grad=True)
    elif config['init_method'] == 'style':
        optimizing_img = Variable(style_img.clone().to(device), requires_grad=True)

    num_of_iterations = {'lbfgs': 1000, 'adam': 2000}

    if config['optimizer'] == 'adam':
        optimizer = Adam([optimizing_img], lr=1e1)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, mrf_loss, dist_loss = build_loss(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)

            total_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, mrf loss={config["mrf_weight"] * mrf_loss.item():12.4f}, dist loss={config["dist_weight"] * dist_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, mrf_loss, dist_loss = build_loss(
                neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, mrf loss={config["mrf_weight"] * mrf_loss.item():12.4f}, dist loss={config["dist_weight"] * dist_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path

if __name__ == '__main__':
    content_images_dir = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
    style_images_dir = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    output_img_dir = os.path.join(os.path.dirname(__file__), 'data', 'output-images')
    img_format = (4, '.jpg')

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, required=True, help="name of the content image")
    parser.add_argument("--style_img_name", type=str, required=True, help="name of the style image")
    parser.add_argument("--height", type=int, help="height of the content and style images, aspect ratio preserved", default=500)
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=3e4)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e5)
    parser.add_argument("--mrf_weight", type=float, help="weight factor for mrf loss", default=1e4)
    parser.add_argument("--dist_weight", type=float, help="weight factor for distance loss", default=1e2)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)

    parser.add_argument("--content_invert", action='store_true')
    parser.add_argument("--style_invert", action='store_true')
    parser.add_argument("--result_invert", action='store_true')
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    results_path = neural_style_transfer(optimization_config)
