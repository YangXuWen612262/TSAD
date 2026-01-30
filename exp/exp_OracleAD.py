# exp/exp_OracleAD.py
import os
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils.tools import adjustment
import torch
import torch.nn as nn
from torch import optim
import inspect
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

    def _model_forward(self, batch_x):
        """
        兼容两种模型 forward:
        - forward(self, x)  -> 只吃 batch_x
        - forward(self, x, _, _, _) -> TSLib transformer 风格
        """
        sig = inspect.signature(self.model.forward)
        n_params = len(sig.parameters)  # 包含 self

        if n_params >= 5:
            return self.model(batch_x, None, None, None)
        else:
            return self.model(batch_x)

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
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                                                map_location=self.device))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')  # 和默认exp一致：逐元素loss

        # =========================
        # (1) statistics on train
        # =========================
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self._model_forward(batch_x) if callable(getattr(self.model, "__call__", None)) else self._model_forward(batch_x)
                recon = self._get_recon_only(outputs)

                # 如果你的OracleAD/G_OracleAD内部走的是 [B,N,L]，这里 recon/batch_x 就不是同shape
                # 但你现在已经能跑通训练，通常你model wrapper会返回 [B,L,C]（和batch_x一致）
                # 如果确实不一致，取消下面两行注释做转置对齐：
                # if recon.dim() == 3 and batch_x.dim() == 3 and recon.shape != batch_x.shape:
                #     # recon [B,C,L] -> [B,L,C]
                #     recon = recon.transpose(1, 2).contiguous()

                score = torch.mean(self.anomaly_criterion(batch_x, recon), dim=-1)  # [B,L]
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # =========================
        # (2) find threshold
        # =========================
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self._model_forward(batch_x) if callable(getattr(self.model, "__call__", None)) else self._model_forward(batch_x)
                recon = self._get_recon_only(outputs)

                # 同上：如果shape不一致可转置
                # if recon.dim() == 3 and batch_x.dim() == 3 and recon.shape != batch_x.shape:
                #     recon = recon.transpose(1, 2).contiguous()

                score = torch.mean(self.anomaly_criterion(batch_x, recon), dim=-1)  # [B,L]
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y.detach().cpu().numpy() if torch.is_tensor(batch_y) else batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # =========================
        # (3) evaluation
        # =========================
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n\n')
        f.close()
        return

    def _get_recon_only(self, model_outputs):
        """
        model_outputs can be:
        - recon tensor
        - (recon, pred, c_star, A)
        We only need recon for evaluation.
        """
        if isinstance(model_outputs, (tuple, list)):
            return model_outputs[0]
        return model_outputs

