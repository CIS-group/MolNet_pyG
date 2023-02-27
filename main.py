import torch
import math
import numpy as np
from model import Test_MolNet, initialize_weight
from torch_geometric.data import DataLoader
import argparse
import pickle
import pandas as pd
from tensorboardX import SummaryWriter
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score

model_map = {"MolNet": Test_MolNet}
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="GPU index to use")
parser.add_argument('--model', default="MolNet", choices=model_map.keys())
parser.add_argument("--dataset", default='bace_cla', help="dataset to use")
parser.add_argument('--epoch', default=500, help="number of iteration")
parser.add_argument('--patience', default=50, help="number of patience")
parser.add_argument('--batch', default=8, help="number of batch")
parser.add_argument('--lr', default=0.001, help="learning rate")
parser.add_argument('--l2', default=0, help="l2 regularization")
parser.add_argument('--T_0', default=10, help="T0 for consine annealing")
parser.add_argument('--split', default=10, help="number of split for CV")
parser.add_argument('--comment', default="", help="comment for experiment")

args = parser.parse_args()

if __name__ == "__main__":

    def train(loader):
        model.train()
        loss_all = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            bce = torch.nn.BCEWithLogitsLoss()
            output = torch.flatten(model(data))
            loss_1 = bce(output, torch.flatten(data.y))
            loss_l2 = 0
            rf_model_kvpair = model.state_dict()
            for key, value in rf_model_kvpair.items():
                if "scalar_embed" in key or "sc_fc" in key:
                    loss_l2 += sum(p.pow(2.0).sum() for p in value.cpu())
            loss = loss_1 + 0.005*loss_l2
            loss.backward()
            loss_all += loss_1.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        with torch.no_grad():
            error = 0
            total_true = []
            total_predict = []
            for data in loader:
                data = data.to(device)
                predict = torch.flatten(model(data))
                sigmoid = torch.nn.Sigmoid()
                sig_predict = sigmoid(predict).cpu().numpy()
                total_predict += list(sig_predict)
                total_true += list(torch.flatten(data.y).cpu().numpy())
                bce = torch.nn.BCEWithLogitsLoss()
                error += (bce(predict, torch.flatten(data.y)) * data.num_graphs).cpu()# MAE
            test_roc = roc_auc_score(np.array(total_true), np.array(total_predict))
            return error / len(loader.dataset), test_roc
    hyper = {}
    hyper["model"] = args.model
    hyper["epoch"] = args.epoch
    hyper["patience"] = args.patience
    hyper["lr"] = args.lr
    hyper["batch"] = args.batch
    hyper["l2"] = args.l2
    hyper["split"] = args.split
    hyper["dataset"] = args.dataset
    hyper["T_0"] = args.T_0
    hyper["comment"] = args.comment
    results = []

    dataset = torch.load("./data/{}.pt".format(hyper["dataset"]))
    device = torch.device('cuda:{}'.format(args.gpu))

    now = datetime.now()
    date = now.strftime("%y%m%d%H%M")
    os.mkdir("./result/{}".format(date))

    idx_dict = None
    with open("./data/index.pkl", "rb") as f:
        idx_dict = pickle.load(f)

    df =pd.DataFrame(columns=["train_loss", "train_roc", "val_loss", "val_roc"])
    total_train_loss = []
    total_train_roc = []
    total_val_loss = []
    total_val_roc = []
    for i in range(hyper["split"]):
        os.mkdir("./result/{}/fold{}".format(date,i))
        writer = SummaryWriter(log_dir=("./result/{}/fold{}".format(date,i)))
        fold_result = None
        best_val_loss = 99999
        best_val_roc = None
        checker = 0
        train_loss, train_roc = None, None

        model = model_map[args.model]()
        model.apply(initialize_weight)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["l2"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = hyper["T_0"], T_mult=2, eta_min=0)
#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, min_lr=0.0005)

        idx_active, idx_inactive = idx_dict["active"], idx_dict["inactive"]
        len_active, len_inactive = int(len(idx_active)/hyper["split"]),  int(len(idx_inactive)/hyper["split"])
        val_idx = np.append(idx_inactive[len_inactive*i:len_inactive*(i+1)], idx_active[len_active*i:len_active*(i+1)])
        train_idx = np.append(np.append(idx_inactive[:len_inactive*i], idx_inactive[len_inactive*(i+1):]),
                              np.append(idx_active[:len_active*i], idx_active[len_active*(i+1):]))

        train_dataset, val_dataset = [dataset[temp] for temp in train_idx], [dataset[temp] for temp in val_idx]

        val_loader = DataLoader(val_dataset, batch_size=hyper["batch"], shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=hyper["batch"], shuffle=True)

        for epoch in range(1, hyper["epoch"]+ 1):
            lr = scheduler.optimizer.param_groups[0]["lr"]
            train_loss = train(train_loader)
            val_loss, val_roc = test(val_loader)
            _, train_roc = test(train_loader)
            checker += 1
            scheduler.step()
            writer.add_scalars('loss/mse', {'train': train_loss, 'val': val_loss}, i)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_roc = val_roc
                final_train_loss, final_train_roc = test(train_loader)
                checker = 0
                torch.save(model.state_dict(), "./result/{}/fold{}/best_model.pth".format(date,i))
            if checker == hyper["patience"]:
                df = df.append({"train_loss": final_train_loss, "train_roc": final_train_roc, "val_loss": best_val_loss, "val_roc": best_val_roc}, ignore_index=True)
                total_train_loss += [final_train_loss]
                total_train_roc += [final_train_roc]
                total_val_loss += [best_val_loss]
                total_val_roc += [best_val_roc]
                fold_result = {"train_loss": final_train_loss, "train_roc": final_train_roc, "val_loss": best_val_loss, "val_roc": best_val_roc}
                with open("./result/{}/fold{}/fold{}_result.txt".format(date, i, i), "w") as f:
                    print(fold_result, file=f)
                f.close()
                break
            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, ROC: {:.7f}, Val Loss: {:.7f}, Val ROC: {:.7f}'.format(epoch, lr, train_loss, train_roc, val_loss, val_roc))
            if epoch == hyper["epoch"]:
                df = df.append({"train_loss": final_train_loss, "train_roc": final_train_roc, "val_loss": best_val_loss, "val_roc": best_val_roc}, ignore_index=True)
                total_train_loss += [final_train_loss]
                total_train_roc += [final_train_roc]
                total_val_loss += [best_val_loss]
                total_val_roc += [best_val_roc]
                fold_result = {"train_loss": final_train_loss, "train_roc": final_train_roc, "val_loss": best_val_loss, "val_roc": best_val_roc}
                with open("./result/{}/fold{}/fold{}_result.txt".format(date, i, i), "w") as f:
                    print(fold_result, file=f)
                f.close()
    df = df.append({"train_loss": np.mean(np.array(total_train_loss)), "train_roc": np.mean(np.array(total_train_roc)), "val_loss": np.mean(np.array(total_val_loss)), "val_roc": np.mean(np.array(total_val_roc))}, ignore_index=True)
    df.to_csv("./result/{}/result.csv".format(date), index=False)
    with open("./result/{}/hyper.txt".format(date), "w") as f:
        print(hyper, file=f)