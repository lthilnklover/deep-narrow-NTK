import torch
from models import DeepNN, ParallelDeepNN, DeepCNN, ParallelDeepCNN
from dataset import mnist_subset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import round_to_nearest_int, str2bool, set_random_seeds

import os
import time
import logging
import argparse
from datetime import datetime


def train(model, optimizer, dataloader, dtype, model_type, dev_list, binary=False):
    model.train()
    train_loss = 0.
    count = 0
    for img, label in dataloader:
        if not binary:
            label = F.one_hot(label, num_classes=10)
        if dtype == 'float64':
            label = label.type(torch.float64)
        else:
            label = label.type(torch.float32)
        img = img.to(dev_list[0])
        label = label.to(dev_list[-1])
        if model_type == 'mlp':
            img = torch.nn.Flatten()(img)
        pred = model(img)
        if model_type == 'cnn':
            pred = torch.squeeze(pred)

        loss = torch.nn.functional.mse_loss(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * img.shape[0]
        count += img.shape[0]

    return train_loss / count


def evaluate(model, dataloader, model_type, dev_list, binary):
    model.eval()
    with torch.no_grad():
        accurate = 0
        count = 0
        for img, label in dataloader:
            img = img.to(dev_list[0])
            label = label.to(dev_list[-1])
            if model_type == 'mlp':
                img = torch.nn.Flatten()(img)
            pred = model(img)
            if model_type == 'cnn':
                pred = torch.squeeze(pred)
            if binary:
                pred = round_to_nearest_int(pred)
            else:
                pred = torch.argmax(pred, dim=1)

            accurate += (pred == label).sum().item()
            count += len(label)

        return accurate * 100. / count


def main(args):
    set_random_seeds(args.seed)
    output_prefix = os.path.join(args.output_dir, args.type)
    if args.binary:
        output_prefix = os.path.join(output_prefix, 'binary')
    else:
        output_prefix = os.path.join(output_prefix, '10_class')

    if args.init is None:
        output_dir = os.path.join(
            output_prefix,
            "depth_" + str(args.depth),
            "init_none",
            "dtype_" + args.dtype,
            "learning_rate_" + str(args.learning_rate / (args.scale ** 2)),
            "epochs_" + str(args.epochs),
            "batch_size_" + str(args.batch_size),
            "date_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )
    else:
        output_dir = os.path.join(
            output_prefix,
            "depth_" + str(args.depth),
            "init_" + args.init,
            "dtype_" + args.dtype,
            "scale_" + str(args.scale),
            "learning_rate_" + str(args.learning_rate),
            "epochs_" + str(args.epochs),
            "batch_size_" + str(args.batch_size),
            "date_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

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

    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if args.binary:
        subset = (0, 1)
        num_out = 1
    else:
        subset = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        num_out = 10

    train_ds = mnist_subset(root='./data', train=True, download=True, transform=T.ToTensor(),
                            subset=subset)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    test_ds = mnist_subset(root='./data', train=False, download=True, transform=T.ToTensor(),
                          subset=subset)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    num_epochs = args.epochs

    dev_list = ['cuda:' + str(dev_num) for dev_num in args.dev_num]
    if args.type == 'mlp':
        if len(dev_list) == 1:
            model = DeepNN(d_in=784, d_out=num_out, L=args.depth, C_L=args.scale, activation='relu', init=args.init)
            model.to(dev_list[0])
        else:
            model = ParallelDeepNN(d_in=784, d_out=num_out, L=args.depth, C_L=args.scale, activation='relu',
                                   init=args.init, dev_list=dev_list)
    elif args.type == 'cnn':
        if len(dev_list) == 1:
            model = DeepCNN(input_size=28, c_in=1, c_out=num_out, L=args.depth, C_L=args.scale, activation='relu',
                            init=args.init)
            model = model.to(dev_list[0])
        else:
            model = ParallelDeepCNN(input_size=28, c_in=1, c_out=num_out, L=args.depth, C_L=args.scale,
                                    activation='relu', init=args.init, dev_list=dev_list)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    total_time = 0.
    for epoch in range(num_epochs):
        logger.info(f'Starting Epoch {epoch + 1:4d}')
        tick = time.time()

        train_loss = train(model, optimizer, train_dl, args.dtype, args.type, dev_list, args.binary)

        loss_list.append(train_loss)
        logger.info(f"Epoch {epoch + 1:4d}/{num_epochs} Loss {loss_list[-1]:.6f}")

        if (epoch + 1) % args.eval_every == 0:

            train_acc = evaluate(model, train_dl, args.type, dev_list, args.binary)
            train_acc_list.append(train_acc)

            test_acc = evaluate(model, test_dl, args.type, dev_list, args.binary)
            test_acc_list.append(test_acc)

            logger.info(f"Epoch {epoch+1:4d}/{num_epochs} Train Accuracy {train_acc_list[-1]:.2f} %")
            logger.info(f"Epoch {epoch+1:4d}/{num_epochs} Test Accuracy {test_acc_list[-1]:.2f} %")

            torch.save({
                'model_state_dict': model.state_dict()
            }, os.path.join(output_dir, 'model.pt'))

        tock = time.time()

        logger.info(f"Epoch {epoch + 1:4d}/{num_epochs} Elapsed Time {tock - tick:.2f}\n")
        total_time += tock - tick

    torch.save({
        'train_loss': loss_list,
        'train_acc': train_acc_list,
        'val_acc': test_acc_list
    }, os.path.join(output_dir, 'result.pt'))

    torch.save({
        'model_state_dict': model.state_dict()
    }, os.path.join(output_dir, 'model.pt'))

    logger.info(f"Total Elapsed Time {total_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='mlp', type=str, help="mlp or cnn")
    parser.add_argument("--binary", default=False, type=str2bool, help='Whether it is a binary classification or not')
    parser.add_argument("--depth", default=1000, type=int, help="Depth of the Model")
    parser.add_argument("--init", default=None, type=str, help="How to Initialize the Model (None or custom)")
    parser.add_argument("--scale", default=1, type=int, help="Scaling Factor, C_L")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning Rate")
    parser.add_argument("--epochs", default=500, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch Size")
    parser.add_argument("--eval_every", default=1, type=int, help="How often to evaluate the model")
    parser.add_argument("--output_dir", default='./results', type=str, help="Output Result Prefix")
    parser.add_argument("--dev_num", default=0, type=int, nargs="+", help="Cuda Device Number")
    parser.add_argument("--seed", default=0, type=int, help='Random Seed')
    parser.add_argument("--dtype", default='float32', type=str, help="data type (float32 or float64)")

    args = parser.parse_args()
    main(args)
