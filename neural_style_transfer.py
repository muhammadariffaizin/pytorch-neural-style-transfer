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
    target_content_representation, target_style_representation, target_content_edge_representation, mask = target_representations

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    # check dimension
    print(target_content_representation.shape, current_content_representation.shape)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        print(gram_gt.size(), gram_hat.size())
        if gram_gt.size() == gram_hat.size():
            style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    current_content_edge_representation = edge_detection(optimizing_img)
    print(target_content_edge_representation.shape, current_content_edge_representation.shape)
    edge_loss = torch.nn.MSELoss(reduction='mean')(target_content_edge_representation, current_content_edge_representation)

    content_loss *= config['content_weight']
    style_loss *= config['style_weight']
    edge_loss *= config['edge_weight']

    # Add L2 regularization
    l2_reg = torch.tensor(0., requires_grad=True).to(optimizing_img.device)
    for param in neural_net.parameters():
        l2_reg += torch.norm(param)
    
    total_loss = content_loss + style_loss

    return total_loss, content_loss, style_loss, edge_loss
    
    # ---------------------------
    content_features = neural_net(target_content_representation)
    style_features = neural_net(target_style_representation)

    print(mask.shape)
    print(mask[0][0])
    
    content_loss = 0.0
    style_loss = 0.0
    for i in range(len(current_set_of_feature_maps)):
        gen_feat = current_set_of_feature_maps[i]
        content_feat = content_features[i]
        style_feat = style_features[i]
        
        mask_resized = resize_mask(mask, gen_feat)

        print(mask_resized.shape)
        print(mask_resized[0][0])

        # Content loss (Foreground preservation)
        content_loss += torch.nn.functional.mse_loss(gen_feat * mask_resized, content_feat * mask_resized)

        gen_gram = gram_matrix(gen_feat)
        style_gram = gram_matrix(style_feat)
        mask_gram = gram_matrix(mask_resized)

        # Style loss (Background transfer)
        style_loss += torch.nn.functional.mse_loss(gen_gram * (1 - mask_gram), style_gram * (1 - mask_gram))
    
    content_edges = edge_detection(target_content_representation)
    generated_edges = edge_detection(optimizing_img)
    edge_loss = torch.nn.functional.mse_loss(content_edges, generated_edges)
    
    content_loss *= config['content_weight']
    style_loss *= config['style_weight']
    edge_loss *= config['edge_weight']
    total_loss = content_loss + style_loss + edge_loss

    return total_loss, content_loss, style_loss, edge_loss

def neural_style_transfer(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dump_path = config['output_img_dir']
    utils.ensure_exists(dump_path)

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    mask_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)
    content_edge_img = edge_detection(content_img)
    mask = utils.prepare_img(mask_img_path, config['height'], device)
    mask = rgb_to_grayscale(mask.squeeze(0))

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')
    
    print(content_img.shape)
    print(style_img.shape)
    print(content_edge_img.shape)
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_content_edge_representation = content_edge_img
    target_representations = (target_content_representation, target_style_representation, target_content_edge_representation, mask)
    
    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        # random_noise_img = torch.randn(content_img.size()).to(device) * 0.256
        optimizing_img = Variable(gaussian_noise_img, requires_grad=True)
    elif config['init_method'] == 'content':
        optimizing_img = Variable(content_img.clone().to(device), requires_grad=True)
    elif config['init_method'] == 'style':
        optimizing_img = Variable(style_img.clone().to(device), requires_grad=True)

    num_of_iterations = {'lbfgs': 300, 'adam': 2000}

    if config['optimizer'] == 'adam':
        optimizer = Adam([optimizing_img], lr=1e1)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, edge_loss = build_loss(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)

            total_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, edge loss={config["edge_weight"] * edge_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, edge_loss = build_loss(
                neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, edge loss={config["edge_weight"] * edge_loss.item():12.4f}')
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
    parser.add_argument("--edge_weight", type=float, help="weight factor for edge preservation loss", default=1e4)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    results_path = neural_style_transfer(optimization_config)
