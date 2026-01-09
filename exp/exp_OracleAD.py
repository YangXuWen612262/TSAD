# exp/exp_OracleAD.py



import os
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment
import torch
import torch.nn as nn
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider

try:
    from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
except Exception:
    EarlyStopping = None
    adjust_learning_rate = None

    def adjustment(gt, pred):
        gt = gt.astype(int)
        pred = pred.astype(int)
        in_anom = False
        start = 0
        for i in range(len(gt)):
            if gt[i] == 1 and not in_anom:
                in_anom = True
                start = i
            if gt[i] == 0 and in_anom:
                end = i
                if pred[start:end].sum() > 0:
                    pred[start:end] = 1
                in_anom = False
        if in_anom:
            end = len(gt)
            if pred[start:end].sum() > 0:
                pred[start:end] = 1
        return gt, pred

try:
    from sklearn.metrics import precision_recall_fscore_support
except Exception:
    precision_recall_fscore_support = None


class Exp_OracleAD(Exp_Basic):
    """
    OracleAD baseline exp for anomaly_detection
    model forward: recon, pred, c_star, A
    """

    def __init__(self, args):
        super().__init__(args)

        # 复用 run.py 已有参数名（避免 unrecognized）
        self.lambda_recon = float(getattr(args, "lambda_recon", 1.0))
        self.lambda_pred = float(getattr(args, "lambda_dev", 0.1))

        self.criterion = nn.MSELoss(reduction="mean")
        self.score_point = str(getattr(args, "score_point", "last"))  # last/mean（run.py未必有，默认last）

    def _build_model(self):
        model_name = self.args.model
        if model_name not in self.model_dict:
            raise ValueError(f"[Exp_OracleAD] model '{model_name}' not registered in Exp_Basic.model_dict")

        ModelObj = self.model_dict[model_name]
        # 兼容你之前“module or class”的情况
        if hasattr(ModelObj, "OracleAD"):
            ModelCls = getattr(ModelObj, "OracleAD")
        elif hasattr(ModelObj, "Model"):
            ModelCls = getattr(ModelObj, "Model")
        else:
            ModelCls = ModelObj

        num_vars = int(getattr(self.args, "enc_in", getattr(self.args, "c_out", None)))
        window_size = int(getattr(self.args, "seq_len", getattr(self.args, "window_size", None)))
        hidden_dim = int(getattr(self.args, "d_model", getattr(self.args, "hidden_dim", 128)))
        dropout = float(getattr(self.args, "dropout", 0.1))

        model = ModelCls(num_vars=num_vars, window_size=window_size, hidden_dim=hidden_dim, dropout=dropout).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        lr = float(getattr(self.args, "learning_rate", 1e-3))
        wd = float(getattr(self.args, "weight_decay", 0.0))
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    @staticmethod
    def _to_model_input(batch_x):
        # TSLib一般是 [B, L, N] -> 模型吃 [B, N, L]
        return batch_x.permute(0, 2, 1).contiguous()

    def _forward_and_loss(self, batch_x):
        batch_x = batch_x.float().to(self.device)     # [B,L,N]
        x_in = self._to_model_input(batch_x)          # [B,N,L]

        recon, pred, c_star, A = self.model(x_in)

        recon_loss = self.criterion(recon, x_in)

        # 自监督预测目标：窗口最后一点（你也可以换成预测t+1，需要数据provider配）
        target_pred = x_in[:, :, -1]
        pred_loss = self.criterion(pred, target_pred)

        total_loss = self.lambda_recon * recon_loss + self.lambda_pred * pred_loss
        return total_loss, recon_loss, pred_loss, recon, pred, c_star, A

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, *_ in vali_loader:
                total_loss, *_ = self._forward_and_loss(batch_x)
                losses.append(total_loss.item())
        self.model.train()
        return float(np.mean(losses)) if len(losses) else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = None
        if EarlyStopping is not None:
            patience = int(getattr(self.args, "patience", 3))
            early_stopping = EarlyStopping(patience=patience, verbose=True)

        model_optim = self._select_optimizer()

        use_amp = bool(getattr(self.args, "use_amp", False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        train_steps = len(train_loader)

        for epoch in range(int(self.args.train_epochs)):
            self.model.train()
            epoch_losses = []
            t0 = time.time()

            for i, (batch_x, *_) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    total_loss, recon_loss, pred_loss, *_ = self._forward_and_loss(batch_x)

                scaler.scale(total_loss).backward()
                scaler.step(model_optim)
                scaler.update()

                epoch_losses.append(total_loss.item())

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            vali_loss = self.vali(vali_data, vali_loader)

            print(f"Epoch {epoch+1}/{self.args.train_epochs} | "
                  f"train={train_loss:.6f} val={vali_loss:.6f} | time={time.time()-t0:.1f}s")

            if early_stopping is not None:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                torch.save(self.model.state_dict(), os.path.join(path, "checkpoint.pth"))

            if adjust_learning_rate is not None:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _collect_energy(self, loader):
        self.model.eval()
        energies = []
        with torch.no_grad():
            for batch_x, *_ in loader:
                batch_x = batch_x.float().to(self.device)  # [B,L,N]
                x_in = self._to_model_input(batch_x)       # [B,N,L]
                recon, pred, c_star, A = self.model(x_in)

                err = (recon - x_in).pow(2)  # [B,N,L]
                if self.score_point == "mean":
                    e = err.mean(dim=(1, 2))               # [B]
                else:
                    e = err[:, :, -1].mean(dim=1)          # [B]
                energies.append(e.detach().cpu().numpy())
        self.model.train()
        return np.concatenate(energies, axis=0) if energies else np.array([])

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        if test == 1:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        train_energy = self._collect_energy(train_loader)
        val_energy = self._collect_energy(vali_loader)
        energy_ref = np.concatenate([train_energy, val_energy], axis=0)

        anomaly_ratio = float(getattr(self.args, "anomaly_ratio", 1.0))
        threshold = np.percentile(energy_ref, 100.0 - anomaly_ratio)

        test_energy = self._collect_energy(test_loader)

        # 取label（尽量兼容）
        gt = None
        for batch in test_loader:
            if len(batch) >= 2:
                y = batch[1]
                if torch.is_tensor(y):
                    y = y.detach().cpu().numpy()

                if y.ndim == 2:
                    y = (y[:, -1] > 0.5).astype(int)
                elif y.ndim > 2:
                    y = y.reshape(y.shape[0], -1)
                    y = (y[:, -1] > 0.5).astype(int)
                else:
                    y = (y > 0.5).astype(int)

                gt = y if gt is None else np.concatenate([gt, y], axis=0)
            else:
                break

        if gt is None:
            print("[Warning] No GT labels found. Only print threshold/stat.")
            print(f"threshold={threshold:.6f}, test_energy mean={test_energy.mean():.6f}, std={test_energy.std():.6f}")
            return

        pred = (test_energy > threshold).astype(int)
        gt_adj, pred_adj = adjustment(gt, pred)

        if precision_recall_fscore_support is not None:
            p, r, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average="binary", zero_division=0)
            print(f"[Test] ratio={anomaly_ratio:.2f}% thr={threshold:.6f} | P={p:.4f} R={r:.4f} F1={f1:.4f}")
        else:
            tp = int(((gt_adj == 1) & (pred_adj == 1)).sum())
            fp = int(((gt_adj == 0) & (pred_adj == 1)).sum())
            fn = int(((gt_adj == 1) & (pred_adj == 0)).sum())
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 2 * p * r / (p + r + 1e-12)
            print(f"[Test] ratio={anomaly_ratio:.2f}% thr={threshold:.6f} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    
