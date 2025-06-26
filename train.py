import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import Dataset, prepareH5
from model import SRmodel

parser = argparse.ArgumentParser(description="SRdemo")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--milestone", type=int, default=20)
parser.add_argument("--data_path", type=str, default="X2")
parser.add_argument("--prepareH5", type=bool, default=False)
opt = parser.parse_args()


def main():
    print("Load dataset\n")
    dataset_train = Dataset(train=True, data_path="X2")
    print(len(dataset_train))
    dataset_val = Dataset(train=False, data_path="X2")
    loader_train = DataLoader(dataset=dataset_train, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, shuffle=False)

    model = SRmodel()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # train
        model.train()
        epoch_loss = 0
        for i, (in_img, out_img) in enumerate(loader_train):
            model.zero_grad()
            optimizer.zero_grad()
            in_img = in_img.to(device)
            out_img = out_img.to(device)

            pred_img = model(in_img)

            loss = loss_function(pred_img, out_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # eval
        model.eval()
        val_loss = 0
        total_psnr = 0
        with torch.no_grad():
            for i, (in_img, out_img) in enumerate(loader_val):
                in_img = in_img.to(device)
                out_img = out_img.to(device)
                pred_img = model(in_img)
                loss = loss_function(pred_img, out_img)
                val_loss += loss.item()
                psnr = 10 * torch.log10(1.0 / loss)
                total_psnr += psnr

        avg_psnr = total_psnr / len(loader_val)
        print(
            f"Epoch: {epoch+1}, Train_Loss: {epoch_loss:.4f}, Eval_Loss: {val_loss:.4f}, PSNR: {avg_psnr:.2f} dB"
        )


if __name__ == "__main__":
    if opt.prepareH5:
        prepareH5(opt.data_path)
    main()
