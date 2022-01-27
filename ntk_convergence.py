import torch
from models import DeepNN, ParallelDeepNN
from utils import get_grad, get_kernel, set_random_seeds
import time
import argparse
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


def get_ntk_list(x0, pts, model, device, depth, C_L):
    grad_x0 = get_grad(model, x0, device)

    ntk_list = []
    for pt in pts:
        grad_pt = get_grad(model, pt, device)
        ntk_list.append(get_kernel(grad_x0, grad_pt).item() / (depth * (C_L ** 2)))
    return ntk_list


def train(model, optimizer, criterion, x, y, dev_list):
    model.train()
    optimizer.zero_grad()
    x, y = x.to(dev_list[0]), y.to(dev_list[-1])
    loss = criterion(model(x).squeeze(), y)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    set_random_seeds(args.seed)

    output_dir = os.path.join(
        args.output_dir,
        "init_" + args.init,
        "iterations_" + str(args.iterations),
        "dtype_" + args.dtype,
        "date_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )

    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger_fname = 'result.txt'
    logger_dir = os.path.join(output_dir, logger_fname)
    file_handler = logging.FileHandler(logger_dir)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Seed: {args.seed}")

    try:
        dev_list = ['cuda:' + str(dev_num) for dev_num in args.dev_num]
    except:
        dev_list = ['cuda:' + str(args.dev_num)]

    depths = [depth for depth in args.depths]
    lrs = [lr for lr in args.learning_rates]
    scales = [scale for scale in args.scales]
    init_ntk = [[] for _ in range(len(depths))]
    trained_ntk = [[] for _ in range(len(depths))]
    loss_list = [[] for _ in range(len(depths))]
    pred_list = [[] for _ in range(len(depths))]

    assert len(depths) == len(lrs)
    assert len(lrs) == len(scales)

    # Fixed point x0
    angle0 = torch.tensor(np.pi * 0.25)
    x0 = torch.stack([torch.cos(angle0), torch.sin(angle0)])

    # Points on the 1st quadrant of the unit circle (to keep it positive)
    angles = torch.tensor(np.linspace(0, 0.5 * np.pi, 220))[10:210]
    circ_pts = torch.stack([torch.cos(angles), torch.sin(angles)], -1)

    train_angles = torch.tensor(np.linspace(0, 0.5 * np.pi, 220))[10:210]
    x = torch.stack([torch.cos(train_angles), torch.sin(train_angles)], -1)
    y = torch.tensor([x[i][0].item() * x[i][1].item() for i in range(x.shape[0])])

    for i in range(len(depths)):
        depth = depths[i]
        logger.info(f"Depth {depth:6d}")
        scaling_factor = scales[i]
        learning_rate = lrs[i]

        for j in range(args.num_trials):
            logger.info(f"Run {j + 1:2d} / {args.num_trials:2d}")
            if len(dev_list) > 1:
                model = ParallelDeepNN(d_in=2, d_out=1, L=depth, C_L=scaling_factor, activation='relu',
                                       init=args.init, dev_list=dev_list)
            else:
                model = DeepNN(d_in=2, d_out=1, L=depth, C_L=scaling_factor, activation='relu', init=args.init)
                model.to(dev_list[0])

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            criterion = torch.nn.MSELoss()

            #######################################
            # Calculate NTK at the initialization #
            #######################################

            logger.info(f"Calculating NTK at the initialization")
            tick = time.time()

            ntk_list = get_ntk_list(x0, circ_pts, model, dev_list[0], depth, scaling_factor)

            logger.info(f"List of NTK at initialization: {ntk_list}")
            init_ntk[i].append(ntk_list)

            tock = time.time()

            logger.info(f"Elapsed time for calculating NTK: {tock - tick:.4f}")

            #######################################
            # Train the model with random samples #
            #######################################

            train_loss = []

            # train with GD to approximate f((x1, x2)) = x1 * x2
            for iteration in range(args.iterations):
                logger.info(f"Iteration {iteration + 1:5d} / {args.iterations:5d}")

                _train_loss = train(model, optimizer, criterion, x, y, dev_list)

                train_loss.append(_train_loss)

                logger.info(f"Loss: {train_loss[-1]}")

            loss_list[i].append(train_loss)
            logger.info(f"Finished Training\n")

            with torch.no_grad():
                x = x.to(dev_list[0])
                y_pred = model(x)
                pred_list[i].append(y_pred.squeeze().detach().cpu().numpy())


            ################################
            # Calculate NTK after training #
            ################################

            logger.info(f"Calculating NTK after training")
            tick = time.time()

            ntk_list = get_ntk_list(x0, circ_pts, model, dev_list[0], depth, scaling_factor)

            logger.info(f"List of NTK after training: {ntk_list}\n\n")
            trained_ntk[i].append(ntk_list)

            tock = time.time()

            logger.info(f"Elapsed time for calculating NTK: {tock - tick:.4f}\n\n")

    ################
    # Save results #
    ################

    torch.save({'depths': depths,
                'lrs': lrs,
                'scales': scales,
                'init_ntk': init_ntk,
                'trained_ntk': trained_ntk,
                'loss_list': loss_list,
                'pred_list': pred_list
                },
               os.path.join(output_dir, 'result.pt')
               )


    ####################
    # Plot the results #
    ####################
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.figsize': (3200, 2000)})
    plt.rcParams.update({'figure.dpi': 1})

    color_list = ['r', 'g', 'b', 'c', 'm', 'y']

    # NTK at initialization as solid line, and after the training as dashed line
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Kernel value')
    for i, depth in enumerate(depths):
        color = color_list[i]
        solid = color + '-'
        dash = color + '--'
        for j, ntk in enumerate(init_ntk[i]):
            if j == 0:
                plt.plot(angles, ntk, solid, linewidth=0.7, label=f'L = {depth}, t = 0')
            else:
                plt.plot(angles, ntk, solid, linewidth=0.7)
        for j, ntk in enumerate(trained_ntk[i]):
            if j == 0:
                plt.plot(angles, ntk, dash, linewidth=0.7, label=f'L = {depth}, t = {args.iterations}')
            else:
                plt.plot(angles, ntk, dash, linewidth=0.7)

    plt.legend(fontsize='small')
    plt.savefig(os.path.join(output_dir, 'ntk_plot.png'))
    plt.cla()

    # Check how well f^* is approximated
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$f_\theta^L(\cos(\gamma), \sin(\gamma))$')
    plt.plot(train_angles, y.cpu().numpy(), 'k-', linewidth=0.7, label=r'True function $f^\star$')
    for i, depth in enumerate(depths):
        color = color_list[i]
        solid = color + '-'
        for j, preds in enumerate(pred_list[i]):
            if j == 0:
                plt.plot(train_angles, preds, solid, linewidth=0.7, label=f'L = {depth}')
            else:
                plt.plot(train_angles, preds, solid, linewidth=0.7)
    plt.legend(fontsize='small')
    plt.savefig(os.path.join(output_dir, 'approx_plot.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=None, type=str, help="How to Initialize the Model")
    parser.add_argument("--depths", default=100, type=int, nargs="+", help='List of Depths')
    parser.add_argument("--num_trials", default=5, type=int, help="How many independent trials to run")
    parser.add_argument("--iterations", default=500, type=int, help="Number of training iterations")
    parser.add_argument("--dtype", default='float32', type=str, help="Data type of the tensor: float32 (single) or "
                                                                     "float64 (double)")
    parser.add_argument("--scales", default=1, type=int, nargs="+", help="List of Scaling factors")
    parser.add_argument("--learning_rates", default=0.0001, type=float, nargs="+", help='List of learning rates')
    parser.add_argument("--output_dir", default='./results', type=str, help="Output directory")
    parser.add_argument("--dev_num", default=0, type=int, nargs="+", help='Cuda Device Number')
    parser.add_argument("--seed", default=0, type=int, help='Random seed')

    args = parser.parse_args()
    main(args)
