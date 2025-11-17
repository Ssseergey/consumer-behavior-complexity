from scipy.linalg import hankel

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

from sklearn.metrics import f1_score

mp.set_start_method("fork", force=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_num_threads(4)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

highlab = ["survival", "socialization", "self_realization"]
device = "cpu"

fwd = 7
split = 90
epochs = 1
batch_size = 1
test_batch_size = 1024
ignore_existing = False
processes = 4
folder_type = "raif"  # "trans_34k"  "transact_10k"

processed_folder = Path(f"processed_data/{folder_type}")
results_folder = Path(f"results/{folder_type}")

results_folder.mkdir(parents=True, exist_ok=True)

existing_result_files = [f for f in results_folder.iterdir() if f.is_file()]
existing_result_files.sort()

files = [
    f
    for f in processed_folder.iterdir()
    if f.is_file()
    and not f.name.endswith("_lzc.csv")
    and not f.name.endswith("_huffman.csv")
]
files.sort()

if ignore_existing:
    exact_names = []
    not_processed_files = []

    for item in existing_result_files:
        exact_names.append(item.name[:3])

    for item in files:
        if item.name[:3] not in exact_names:
            not_processed_files.append(item)

    files = not_processed_files
    print("Files to process:", files)


def MakeSet(ser, lzc, fwd):
    H = hankel(ser)
    X0 = H[: -lzc - fwd + 1, :lzc]
    X = []
    for i in range(X0.shape[0] - fwd - 1):
        X.append(X0[i : i + fwd + 1, :].T)
    X = np.array(X)
    y = H[: -lzc - 2 * fwd, lzc + fwd : lzc + 2 * fwd]
    return X, y


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.tanh(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


def train_model(df, compression, user_id):
    model_metrics = []

    for sssr in highlab:
        trans = df.loc[user_id][sssr].values
        # lzc = compression.loc[user_id][sssr]

        lzc = 30

        X_train, y_train = MakeSet(trans[:split], lzc, fwd)
        X_test, y_test = MakeSet(trans[split:], lzc, fwd)

        if len(trans) <= split:
            print("Skipped ", user_id, sssr)
            continue

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

        model = LSTMModel(
            input_size=X_train_t.shape[2],
            hidden_size=X_train_t.shape[1],
            output_size=y_train_t.shape[1],
        ).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t), batch_size=test_batch_size, shuffle=False
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        f1 = []
        with torch.no_grad():
            for xb, yb in test_loader:
                y_pred = model(xb)
                y_pred_bin = (y_pred > 0.5).int()

                for yp, yt in zip(y_pred_bin.cpu().numpy(), yb.cpu().numpy()):
                    f1_score_val = f1_score(yt, yp, zero_division=1)
                    f1.append(f1_score_val)

        if len(y_test_t) <= 1:
            print("Skipped ", user_id, sssr)
            continue

        model_metrics.append(np.where(np.array(f1) > 0.75)[0].shape[0] / len(f1))

        # del model
        # torch.cuda.empty_cache()

    return model_metrics


def worker(args):
    df, compression, user_id = args
    try:
        result = train_model(df, compression, user_id)
        return user_id, result, None
    except Exception as e:
        return user_id, None, str(e)


def process_client(df, compression):
    n = len(compression.index.to_list())
    res = pd.DataFrame(columns=["id"] + highlab)
    res["id"] = [0] * n

    users = compression.index.to_list()
    args_list = [(df, compression, user) for user in users]
    with mp.Pool(processes=processes) as pool:
        with tqdm(
            total=len(users), desc="Processing users", position=1, leave=False
        ) as pbar:
            for i, (user, result, error) in enumerate(pool.imap(worker, args_list)):
                (
                    res.loc[i, "id"],
                    res.loc[i, highlab[0]],
                    res.loc[i, highlab[1]],
                    res.loc[i, highlab[2]],
                ) = (user, *result)

                if error:
                    print(f"user:{user} error:{error}")
                pbar.update(1)

    return res


def process_file(file):
    df = pd.read_csv(file, index_col=["client", "date"])
    compression = pd.read_csv(file.parent / f"{file.stem}_lzc.csv", index_col="client")

    res = process_client(df, compression)
    res.to_csv(results_folder / f"{file.stem}_results.csv", index=False)


def main():
    results_folder.mkdir(parents=True, exist_ok=True)

    for file in tqdm(files, desc="Processing files", position=0, leave=False):
        process_file(file)


if __name__ == "__main__":
    main()
