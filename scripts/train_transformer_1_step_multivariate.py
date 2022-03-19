import torch
import torch.nn as nn
import numpy as np
import time
import math
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(__file__)
CurDir = Path(current_dir)
sys.path.append(os.path.join(current_dir, ".."))
from datetime import datetime
from tqdm import tqdm

today = datetime.strftime(datetime.today(), "%Y-%m-%d")
import yfinance as yf


class args:
    lr = 1e-4
    batch_size = 32
    grad_norm = 0.7
    device = "cuda:1"
    num_layers = 1
    dropout = 0.1
    n_epoch = 10000
    n_log_interval = 100
    save_folder_name = "m_transformer_1_step_multivariate"
    stock_start_date = "2015-01-01"
    scheduler_step_size = 5
    scheduler_gamma = 0.9
    train_length = 1300
    WINDOW_SIZE = 32
    seed = 0
    nhead = 6
    input_window = 30  # number of input steps
    output_window = 1  # number of prediction steps, in this model its fixed to one


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Request historical data for past 5 years
df = yf.download("^GSPC", start=args.stock_start_date, end=today)
df.columns = [i.replace(" ", "_") for i in list(df)]


"""
Function
"""


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + args.output_window : i + tw + args.output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def get_batch(source, i, batch_size, target_index):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i : i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(args.input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(args.input_window, 1))
    input = input.squeeze(dim=2)
    target = target[:, :, :, target_index]
    return input, target


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, nhead=10):
        super(TransAm, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
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

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


def train(train_data):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.0
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, args.batch_size)):
        data, targets = get_batch(train_data, i, args.batch_size)
        optimizer.zero_grad()
        output = model(data.to(args.device))
        loss = criterion(output, targets.to(args.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        total_loss += loss.item()
    else:
        return total_loss


def plot_and_loss(eval_model, data_source, epoch, folder):
    eval_model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data.to(args.device))
            total_loss += criterion(output.cpu(), target.cpu()).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy() -> no need to detach stuff..
    len(test_result)
    plt.plot(test_result, color="red")
    plt.plot(truth[:500], color="blue")
    plt.plot(test_result - truth, color="green")
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    fig_path = f"{folder}/epoch{epoch:05d}.png"
    filepath = CurDir.joinpath(fig_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close()

    return total_loss / i


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data.to(args.device))
            total_loss += len(data[0]) * criterion(output.cpu(), targets).cpu().item()
    return total_loss / len(data_source)


N_Train_Test_Split = args.train_length
train_data = df[:N_Train_Test_Split]
valid_data = df[N_Train_Test_Split:]

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


target_index = list(df).index("Adj_Close")
from functools import partial

get_batch = partial(get_batch, target_index=target_index)

train_data_scaled = scaler.transform(train_data[col_order].values)
valid_data_scaled = scaler.transform(valid_data[col_order].values)


model = TransAm(feature_size=6, num_layers=args.num_layers, dropout=args.dropout, nhead=args.nhead).to(args.device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, gamma=args.scheduler_gamma)

best_val_loss = float("inf")
best_model = None
train_inputs = create_inout_sequences(train_data_scaled, tw=args.input_window)
valid_inputs = create_inout_sequences(valid_data_scaled, tw=args.input_window)
print(train_inputs.shape, valid_inputs.shape)

epoch = 1
pbar = tqdm(range(epoch, args.n_epoch + 1))
va_loss = np.inf
best_va_loss = np.inf
for epoch in pbar:
    epoch_start_time = time.time()
    train(train_inputs)

    if (epoch % args.n_log_interval) == 0:
        va_loss = plot_and_loss(model, valid_inputs, epoch, args.save_folder_name)
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
