# exp/exp_G_OracleAD.py
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

# TSLib utilities (存在则用；没有就fallback)
try:
    from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
except Exception:
    EarlyStopping = None
    adjust_learning_rate = None

    # event-level adjustment fallback (简化版)
    def adjustment(gt, pred):
        """
        把连续异常段“整段算对”：只要pred命中异常段内任一点，就把整段pred置1
        gt/pred: np.ndarray shape [T]
        """
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


class Exp_G_OracleAD(Exp_Basic):
    """
    G_OracleAD experiment runner for TSLib anomaly_detection pipeline.
    - model forward should return: recon, pred, c_star, A
      recon: [B, N, L], pred: [B, N], c_star: [B, N, d], A: [B, N, N]
    """

    def __init__(self, args):
        super().__init__(args)

        # ---- hyperparams (如果你没在parser里加，这里给默认值) ----
        self.lambda_pred = float(getattr(args, "lambda_pred", 0.1))
        self.lambda_grad = float(getattr(args, "lambda_grad", 0.05))
        self.beta_grad = float(getattr(args, "beta_grad", 4.0))
        self.grad_eps = float(getattr(args, "grad_eps", 1e-6))
        self.use_grad_soft = int(getattr(args, "use_grad_soft", 1))  # 1=启用
        self.dist_norm = int(getattr(args, "dist_norm", 1))          # 1=对c_star做L2 norm
        self.score_point = str(getattr(args, "score_point", "last")) # last / mean

        self.criterion = nn.MSELoss(reduction="mean")

        self._dbg_reg_print = 0
        self._dbg_g_none = 0


        print("lambda_grad:", getattr(args, "lambda_grad", None),"use_grad_soft:", getattr(args, "use_grad_soft", None))
        print(
                f"[Init] lambda_pred={self.lambda_pred} lambda_grad={self.lambda_grad} beta_grad={self.beta_grad} "
                f"use_grad_soft={self.use_grad_soft} dist_norm={self.dist_norm} score_point={self.score_point} "
                f"grad_eps={self.grad_eps}"
            )


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
        """
        这里尽量兼容 TSLib 常见写法：
        - 你可以在 exp_basic.py 的 model_dict 里把 "G_OracleAD" 映射到你的模型类
        - 你的模型构造函数是: G_OracleAD(num_vars, window_size, hidden_dim, dropout)
        """
        model_name = self.args.model

        if model_name not in self.model_dict:
            raise ValueError(
                f"[Exp_G_OracleAD] args.model='{model_name}' not found in Exp_Basic.model_dict. "
                f"Please register it in exp/exp_basic.py first."
            )

        # ModelCls = self.model_dict[model_name]
        ModelObj = self.model_dict[model_name]

        # 如果注册进来的是“模块”，就取其中的类
        if hasattr(ModelObj, "G_OracleAD"):
            ModelCls = getattr(ModelObj, "G_OracleAD")
        elif hasattr(ModelObj, "Model"):
            ModelCls = getattr(ModelObj, "Model")
        else:
            ModelCls = ModelObj  # 假设它本身就是类

        num_vars = int(getattr(self.args, "enc_in", getattr(self.args, "c_out", None)))
        if num_vars is None:
            raise ValueError("[Exp_G_OracleAD] Cannot infer num_vars. Please set args.enc_in (number of variables).")

        window_size = int(getattr(self.args, "seq_len", getattr(self.args, "window_size", None)))
        if window_size is None:
            raise ValueError("[Exp_G_OracleAD] Cannot infer window_size. Please set args.seq_len (window length).")

        hidden_dim = int(getattr(self.args, "d_model", getattr(self.args, "hidden_dim", 128)))
        dropout = float(getattr(self.args, "dropout", 0.1))

        # model = ModelCls(num_vars=num_vars, window_size=window_size, hidden_dim=hidden_dim, dropout=dropout).float()
        model = ModelCls(
                            num_vars=num_vars,
                            window_size=window_size,
                            hidden_dim=hidden_dim,
                            dropout=dropout
                        ).float()
                        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        lr = float(getattr(self.args, "learning_rate", 1e-3))
        wd = float(getattr(self.args, "weight_decay", 0.0))
        return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    @staticmethod
    def _to_model_input(batch_x):
        """
        TSLib dataloader 通常给 [B, L, N]，你的模型要 [B, N, L]
        """
        if batch_x.dim() != 3:
            raise ValueError(f"batch_x should be 3D [B,L,N], got {tuple(batch_x.shape)}")
        return batch_x.permute(0, 2, 1).contiguous()

    def _grad_soft_regularizer(self, base_loss, c_star, A):
        """
        你说的“梯度软约束 -> 软约束特征距离矩阵”的实现：
        - g = | d(base_loss) / dA |   (detach)
        - w = sigmoid(beta * (norm(g) - 1))
        - D = pairwise_dist(c_star)
        - reg = mean( w * D ) (去掉对角)
        """
        if (A is None) or (c_star is None):
            return base_loss.new_tensor(0.0)

        g = torch.autograd.grad(
            base_loss, A,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]

        # ---- debug: check grad wrt A ----
        if self._dbg_reg_print < 5:  # 只打印前5次，避免刷屏
            if g is None:
                self._dbg_g_none += 1
                print(f"[Reg-Debug] g is None (count={self._dbg_g_none}) -> reg=0")
            else:
                g_abs_mean = g.detach().abs().mean().item()
                g_abs_max = g.detach().abs().max().item()
                A_mean = A.detach().mean().item()
                A_std = A.detach().std().item()
                print(f"[Reg-Debug] |g| mean={g_abs_mean:.3e} max={g_abs_max:.3e} | A mean={A_mean:.4f} std={A_std:.4f}")
            self._dbg_reg_print += 1


        if g is None:
            return base_loss.new_tensor(0.0)

        g = g.detach().abs()  # [B,N,N]

        # row-wise normalize，让每个query维度的梯度可比
        g_row_mean = g.mean(dim=-1, keepdim=True)
        g_norm = g / (g_row_mean + self.grad_eps)

        # soft weight: 大于均值的边权更大
        w = torch.sigmoid(self.beta_grad * (g_norm - 1.0))  # [B,N,N]

        # feature distance
        if self.dist_norm == 1:
            c_star = torch.nn.functional.normalize(c_star, p=2, dim=-1)
        D = torch.cdist(c_star, c_star, p=2)  # [B,N,N]

        # remove diagonal
        B, N, _ = D.shape
        eye = torch.eye(N, device=D.device, dtype=D.dtype).unsqueeze(0)  # [1,N,N]
        mask = 1.0 - eye

        reg = (w * D * mask).sum() / (mask.sum() * B + self.grad_eps)
        return reg

    def _forward_and_loss(self, batch_x):
        """
        返回 total_loss, base_loss, recon_loss, pred_loss, reg, outputs...
        """
        batch_x = batch_x.float().to(self.device)  # [B,L,N]
        x_in = self._to_model_input(batch_x)        # [B,N,L]

        recon, pred, c_star, A = self.model(x_in)

        # recon: [B,N,L] vs x_in: [B,N,L]
        recon_loss = self.criterion(recon, x_in)

        # pred: [B,N]，这里用窗口最后一个点作为“自监督预测目标”
        target_pred = x_in[:, :, -1]
        pred_loss = self.criterion(pred, target_pred)

        base_loss = recon_loss + self.lambda_pred * pred_loss

        if self.use_grad_soft == 1 and self.lambda_grad > 0:
            reg = self._grad_soft_regularizer(base_loss, c_star, A)
        else:
            reg = base_loss.new_tensor(0.0)

        total_loss = base_loss + self.lambda_grad * reg
        return total_loss, base_loss, recon_loss, pred_loss, reg, recon, pred, c_star, A

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        losses = []

        # 临时关掉梯度软约束（只在vali时）
        old_use = self.use_grad_soft
        self.use_grad_soft = 0

        with torch.no_grad():
            for batch_x, *_ in vali_loader:
                total_loss, *_ = self._forward_and_loss(batch_x)
                losses.append(total_loss.item())

        # 恢复
        self.use_grad_soft = old_use
        self.model.train()
        return float(np.mean(losses)) if len(losses) else 0.0


    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = None
        if EarlyStopping is not None:
            patience = int(getattr(self.args, "patience", 3))
            early_stopping = EarlyStopping(patience=patience, verbose=True)

        model_optim = self._select_optimizer()

        # AMP (可选)
        use_amp = bool(getattr(self.args, "use_amp", False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(int(self.args.train_epochs)):
            iter_count = 0
            epoch_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, *_) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    total_loss, base_loss, recon_loss, pred_loss, reg, *_ = self._forward_and_loss(batch_x)

                scaler.scale(total_loss).backward()
                scaler.step(model_optim)
                scaler.update()

                epoch_losses.append(total_loss.item())

                if (i + 1) % int(getattr(self.args, "log_step", 100)) == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (train_steps - i)

                    # -------- extra stats (no grad) --------
                    with torch.no_grad():
                        # 输入统计（在 _forward_and_loss 已经有 x_in，但这里 batch_x 也能快速看）
                        bx_mean = batch_x.float().mean().item()
                        bx_std = batch_x.float().std().item()

                    reg_contrib = (self.lambda_grad * reg).item() if torch.is_tensor(reg) else float(self.lambda_grad) * float(reg)

                    print(
                        f"\titers: {i+1}/{train_steps}, epoch: {epoch+1} | "
                        f"loss={total_loss.item():.6f} "
                        f"(base={base_loss.item():.6f}, recon={recon_loss.item():.6f}, pred={pred_loss.item():.6f}, "
                        f"reg={reg.item():.6f}, lambda_grad*reg={reg_contrib:.6f}) | "
                        f"batch_x(mean/std)={bx_mean:.4f}/{bx_std:.4f} | "
                        f"speed={speed:.4f}s/iter, left={left_time:.1f}s"
                    )

                    iter_count = 0
                    time_now = time.time()
                # if i == 0:
                #     print(f"[Test-Debug][Train] batch_x={tuple(batch_x.shape)} recon={tuple(recon.shape)} "
                #         f"x(mean/std)={batch_x.mean().item():.4f}/{batch_x.std().item():.4f} "
                #         f"recon(mean/std)={recon.mean().item():.4f}/{recon.std().item():.4f}")



            train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else 0.0
            vali_loss = self.vali(vali_data, vali_loader)

            print(
                f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.6f} Vali Loss: {vali_loss:.6f}"
            )

            if early_stopping is not None:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                # 没EarlyStopping就每轮存一次
                torch.save(self.model.state_dict(), os.path.join(path, "checkpoint.pth"))

            if adjust_learning_rate is not None:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # load best
        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _collect_energy(self, loader):
        """
        energy 用于阈值/检测：
        - last: 用每个窗口的最后一个点的重构误差作为该点的energy（常见做法）
        - mean: 用窗口平均重构误差
        返回: np.ndarray shape [T_like]
        """
        self.model.eval()
        energies = []

        with torch.no_grad():
            for batch_x, *_ in loader:
                batch_x = batch_x.float().to(self.device)   # [B,L,N]
                x_in = self._to_model_input(batch_x)         # [B,N,L]
                recon, pred, c_star, A = self.model(x_in)

                # mse per point
                err = (recon - x_in).pow(2)  # [B,N,L]

                if self.score_point == "mean":
                    e = err.mean(dim=(1, 2))                 # [B]
                else:
                    # last point
                    e = err[:, :, -1].mean(dim=1)            # [B]

                energies.append(e.detach().cpu().numpy())

        self.model.train()
        if len(energies) == 0:
            return np.array([])
        return np.concatenate(energies, axis=0)

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

                outputs = self._model_forward(batch_x)

                recon = self._get_recon_only(outputs)

                if i == 0:
                    print(f"[Test-Debug][Test]  batch_x={tuple(batch_x.shape)} recon={tuple(recon.shape)} "
                        f"x(mean/std)={batch_x.mean().item():.4f}/{batch_x.std().item():.4f} "
                        f"recon(mean/std)={recon.mean().item():.4f}/{recon.std().item():.4f}")


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

                outputs = self._model_forward(batch_x)

                recon = self._get_recon_only(outputs)

                if i == 0:
                    print(f"[Test-Debug][Test]  batch_x={tuple(batch_x.shape)} recon={tuple(recon.shape)} "
                        f"x(mean/std)={batch_x.mean().item():.4f}/{batch_x.std().item():.4f} "
                        f"recon(mean/std)={recon.mean().item():.4f}/{recon.std().item():.4f}")

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
