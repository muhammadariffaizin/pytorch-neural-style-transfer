import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import cv2 as cv
import argparse
from utils.loss_fns import *

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(h * w)

def rgb_to_grayscale(img):
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError(f'Expected img with shape (3, H, W), but got {img.shape}')
    r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.unsqueeze(0)

def build_loss(neural_net, optimizing_img, mrf_layers, loss_layers, loss_fns, targets, weights, style_layers, style_patches_lists, weight_list, patch_size, dist_template, inpaint_template, cont_dist, mrf_loss_fn, config, device):
    optimizing_fms = neural_net(optimizing_img, loss_layers)

    content_loss = 0
    style_loss = 0
    for a,A in enumerate(optimizing_fms[len(mrf_layers):]):
        one_layer_loss = weights[a] * loss_fns[a](A, targets[a])
        if a < len(style_layers):
            style_loss += one_layer_loss
        else:
            content_loss += one_layer_loss

    # MRF loss
    mrf_loss = mrf_loss_fn(optimizing_fms[:len(mrf_layers)], style_patches_lists, weight_list, patch_size)
    
    # Distance transform loss
    d_temp = cont_dist * optimizing_img.clone()
    dist_loss = torch.nn.MSELoss().to(device)(d_temp, dist_template)

    # Inpainting loss
    i_temp = utils.inpaint_cv2(optimizing_img.clone(), device)
    color_loss = torch.nn.MSELoss().to(device)(i_temp.mean(dim=[2,3]), inpaint_template.mean(dim=[2,3]))

    content_loss *= config['content_weight']
    style_loss *= config['style_weight']
    mrf_loss *= config['mrf_weight']
    dist_loss *= config['dist_weight']
    color_loss *= config['color_weight']

    # Add L2 regularization
    l2_reg = torch.tensor(0., requires_grad=True).to(optimizing_img.device)
    for param in neural_net.parameters():
        l2_reg += torch.norm(param)
    
    total_loss = content_loss + style_loss + mrf_loss + dist_loss

    return total_loss, content_loss, style_loss, mrf_loss, dist_loss, color_loss

def neural_style_transfer(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    utils.ensure_exists(dump_path)

    content_img = utils.prepare_img(content_img_path, config['height'], device, config['content_invert'])
    style_img = utils.prepare_img(style_img_path, config['height'], device, config['style_invert'])

    # test save image
    utils.save_image_with_unnormalize_img(content_img, 'content.jpg')
    utils.save_image_with_unnormalize_img(style_img, 'style.jpg')

    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        gaussian_noise_img = torch.from_numpy(gaussian_noise_img).to(device)
        # random_noise_img = torch.randn(content_img.size()).to(device) * 0.256
        optimizing_img = Variable(gaussian_noise_img, requires_grad=True)
    elif config['init_method'] == 'content':
        optimizing_img = Variable(content_img.clone().to(device), requires_grad=True)
    elif config['init_method'] == 'style':
        optimizing_img = Variable(style_img.clone().to(device), requires_grad=True)

    # Define style layers
    style_layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
    style_weights = [1e4/n**3 for n in [64,128,256,512,512]]
    # Define mrf layers
    mrf_layers = ['relu3_1', 'relu4_1'] 
    # Define content layers
    content_layers = ['relu4_2']
    content_weights = [1e3]
    # loss layers: layers to be used by opt_img ( style_layers & mrf_layers & content_layers)
    loss_layers = mrf_layers + style_layers + content_layers

    neural_net = utils.prepare_model(config['model'])
    neural_net.load_state_dict(torch.load('vgg_conv.pth'))
    neural_net.to(device)
    print(f'Using {config["model"]} in the optimization procedure.')

    style_variable = Variable(style_img.clone().to(device), requires_grad=False)
    content_variable = Variable(content_img.clone().to(device), requires_grad=False)

    # Feature maps from style images
    mrf_fms = [A.detach() for A in neural_net(style_variable, mrf_layers)]
    # Extract style patches & create conv3d from those patches
    style_patches_lists, weight_list = get_style_patch_weights(mrf_fms, device, k=config['patch_size'])

    # Compute style target
    style_targets = [utils.gram_matrix(A).detach() for A in neural_net(style_variable, style_layers)]
    # Computer content target
    content_targets = [A.detach() for A in neural_net(content_variable, content_layers)]
    # targets
    targets = style_targets + content_targets
    # layers weights
    weights = style_weights + content_weights
    # Opt layers
    loss_fns = [utils.GramMSELoss()] * len(style_layers) + [torch.nn.MSELoss()] * len(content_layers)
    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    # create distance module path
    cont_dist = utils.dist_cv2(content_img.clone(), device)
    cont_dist = cont_dist**6
    cont_dist[cont_dist>1e3] = 1e3
    cont_dist[cont_dist==float("Inf")] = 1e3
    dist_template = cont_dist*content_img

    # create inpaint module path
    inpaint_img = utils.inpaint_cv2(style_img.clone(), device)
    print(f'Input image shape: {inpaint_img.shape}')
    print(f'Content image shape: {content_img.shape}')
    inpaint_template = inpaint_img*(255 - content_img.clone())

    utils.save_image_with_unnormalize_img(dist_template, 'dist_template.png')
    utils.save_image_with_unnormalize_img(inpaint_template, 'inpaint_template.png')

    num_of_iterations = {'lbfgs': 1000, 'adam': 2000}

    if config['optimizer'] == 'adam':
        optimizer = Adam([optimizing_img], lr=1e1)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, mrf_loss, dist_loss, color_loss = build_loss(neural_net, optimizing_img, mrf_layers, loss_layers, loss_fns, targets, weights, style_layers, style_patches_lists, weight_list, config['patch_size'], dist_template, inpaint_template, cont_dist, mrf_loss_fn, config, device)

            total_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, mrf loss={config["mrf_weight"] * mrf_loss.item():12.4f}, dist loss={config["dist_weight"] * dist_loss.item():12.4f}, color loss={config["color_weight"] * color_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS([optimizing_img], max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, mrf_loss, dist_loss, color_loss = build_loss(neural_net, optimizing_img, mrf_layers, loss_layers, loss_fns, targets, weights, style_layers, style_patches_lists, weight_list, config['patch_size'], dist_template, inpaint_template, cont_dist, mrf_loss_fn, config, device)
            if total_loss.requires_grad:
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_([optimizing_img], max_norm=1.0)
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, mrf loss={config["mrf_weight"] * mrf_loss.item():12.4f}, dist loss={config["dist_weight"] * dist_loss.item():12.4f}, color loss={config["color_weight"] * color_loss.item():12.4f}')
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
    parser.add_argument("--mrf_weight", type=float, help="weight factor for mrf loss", default=1e3)
    parser.add_argument("--dist_weight", type=float, help="weight factor for distance loss", default=1e6)
    parser.add_argument("--color_weight", type=float, help="weight factor for color loss", default=1e2)

    parser.add_argument("--patch_size", type=int, help="patch size of the style image", default=5)
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
