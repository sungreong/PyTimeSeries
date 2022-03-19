"""
Library
"""
import os, sys
from pathlib import Path

current_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(current_dir, ".."))
CurDir = Path(current_dir)
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from src.algorithms.time2vec.time2vec import SineActivation
import torch
from tqdm import tqdm

torch.cuda.is_available()
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime

today = datetime.strftime(datetime.today(), "%Y-%m-%d")
import yfinance as yf

# R"""equest historical data for past 5 years
class args:
    lr = 1e-4
    batch_size = 32
    grad_norm = 0.7
    device = "cuda:1"
    time_dim = 16
    num_layers = 1
    dropout = 0.1
    nhead = 8
    n_epoch = 10000
    n_log_interval = 50
    save_folder_name = "time2vec_transformer"
    stock_start_date = "2015-01-01"
    scheduler_step_size = 5
    scheduler_gamma = 0.9
    train_length = 1300
    WINDOW_SIZE = 32


df = yf.download("^GSPC", start=args.stock_start_date, end=today)
df.columns = [i.replace(" ", "_") for i in list(df)]
target_col = "Adj_Close"

"""
Data Preprocesisng
"""
from sklearn.preprocessing import MinMaxScaler

scaled_data = []
for col in list(df):
    min_, max_ = df[col].min(), df[col].max()
    min_value = 0.9 * min_
    max_value = 1.1 * max_
    scaled_data.append(np.array([min_value, max_value]).reshape(-1, 1))
else:
    scaled_info = np.hstack(scaled_data)
    col_order = list(df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(scaled_info)

df[col_order] = scaler.transform(df[col_order].values)
df = df.reset_index(drop=False)
df["date"] = pd.to_datetime(df["Date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["dayofmonth"] = df["date"].dt.days_in_month
df["dayofyear"] = df["date"].dt.dayofyear
df["weekday"] = df["date"].dt.weekday
df["weekofyear"] = df["date"].dt.weekofyear
df.drop(columns=["year", "date", "Date"], inplace=True)
all_data = pd.get_dummies(
    df, columns=["month", "day", "dayofweek", "dayofmonth", "dayofyear", "weekday", "weekofyear"]
)


def create_inout_sequences(input_data, target_data, tw, output_window):
    input_seq = []
    output_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = target_data[i + output_window : i + tw + output_window]
        input_seq.append(train_seq)
        output_seq.append(train_label)
    return np.array(input_seq), np.array(output_seq)[:, :, np.newaxis]


def evaluate(eval_model, data_loader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, samples in enumerate(data_loader):
            x_train, y_train = samples
            output = eval_model(x_train)
            total_loss += criterion(output, y_train).cpu().item()
    return total_loss / batch_idx


def plot_and_loss(eval_model, data_loader, epoch, folder, device):
    eval_model.eval()
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            pred = model(x.to(device))
            _ = y[:, -1].squeeze().cpu()
            truth = torch.cat((truth, _), 0)
            _ = pred[:, -1].squeeze().cpu()
            total_loss += criterion(pred.cpu(), y.cpu()).item()

            test_result = torch.cat((test_result, _), 0)
        else:
            total_loss /= batch_idx
            plt.plot(test_result.detach().numpy(), color="red")
            plt.plot(truth.detach().numpy(), color="blue")
            plt.plot((test_result - truth).detach().numpy(), color="green")
            plt.title(f"Loss : {total_loss:.5f}")
            plt.grid(True, which="both")
            plt.axhline(y=0, color="k")
            fig_path = f"{folder}/epoch{epoch:05d}.png"
            filepath = CurDir.joinpath(fig_path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath)
            plt.close()
            return total_loss


class TransAm(nn.Module):
    def __init__(self, time_dim=16, feature_size=250, num_layers=1, dropout=0.1, nhead=6, device="cpu"):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.time_vec = SineActivation(feature_size, time_dim).to(device)
        output_size = 2 * time_dim + feature_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(output_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        time_src = self.time_vec(src)
        src = torch.cat((src, time_src), axis=-1)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


N_Train_Test_Split = args.train_length
train = all_data[:N_Train_Test_Split]
valid = all_data[N_Train_Test_Split:]
print(train.shape, valid.shape)

x_train_ts, y_train_ts = create_inout_sequences(train.values, train[target_col], tw=args.WINDOW_SIZE, output_window=1)
x_valid_ts, y_valid_ts = create_inout_sequences(valid.values, valid[target_col], tw=args.WINDOW_SIZE, output_window=1)


feature_size = train.values.shape[1]
model = TransAm(
    time_dim=args.time_dim,
    feature_size=feature_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
    nhead=args.nhead,
    device=args.device,
)
model = model.to(args.device)

print(x_train_ts.shape, y_train_ts.shape)


tr_dataset = CustomDataset(x_data=x_train_ts, y_data=y_train_ts)
va_dataset = CustomDataset(x_data=x_valid_ts, y_data=y_valid_ts)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, gamma=args.scheduler_gamma)
tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
va_dataloader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False)


epoch = 1
total_n_batch = len(tr_dataloader)
pbar = tqdm(range(epoch, args.n_epoch), desc="start")
va_loss = np.inf
best_va_loss = np.inf
for epoch in pbar:
    model.train()
    total_loss = 0
    for batch_idx, samples in enumerate(tr_dataloader):
        optimizer.zero_grad()
        x_train, y_train = samples
        pred = model(x_train.to(args.device))
        loss = criterion(y_train.to(args.device), pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        total_loss += loss.item()
        cur_loss = total_loss / (batch_idx + 1)
        percent = (batch_idx / total_n_batch) * 100
        pbar.set_description(f"[{epoch:04d}][{percent:05.2f}%] : {cur_loss:.3f} / validation loss : {va_loss:.5f}")
    else:

        pbar.update(1)
        scheduler.step()

    if epoch % args.n_log_interval == 0 == 0:
        va_loss = plot_and_loss(model, va_dataloader, epoch, args.save_folder_name, device=args.device)
        if best_va_loss > va_loss:
            best_va_loss = va_loss
            save_parms = dict(
                parameter=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                loss=best_va_loss,
            )
            save_path = CurDir.joinpath(f"../model/{args.save_folder_name}/model.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_parms, save_path)
