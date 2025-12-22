from exp.exp_basic import Exp_Basic
from models import OracleAD
from utils.tools import adjust_learning_rate
from data_provider.data_factory import data_provider
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score

class Exp_OracleAD(Exp_Basic):
    def __init__(self, args):
        super(Exp_OracleAD, self).__init__(args)
        self.SLS = None 
        self.dist_matrices_buffer = []

    def _build_model(self):
        model = OracleAD.Model(self.args).float()
        if self.args.use_gpu:
            model = model.cuda()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        # =======================================================
        # 【关键修复】在此处初始化优化器和损失函数
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()
        # =======================================================

        train_steps = len(train_loader)
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            self.dist_matrices_buffer = [] 
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                # Forward
                # 模型的 forward 返回: pred(预测值), recon(重建值), dist_matrix(距离矩阵)
                pred, recon, dist_matrix = self.model(batch_x)
                
                # 收集距离矩阵用于后续计算 SLS
                self.dist_matrices_buffer.append(dist_matrix.detach().cpu().numpy())
                
                # --- Loss Calculation ---
                # 1. Prediction Loss (针对最后一个时间点)
                true_last = batch_x[:, -1:, :]
                loss_pred = self.criterion(pred, true_last)
                
                # 2. Reconstruction Loss (针对过去 L-1 个时间点)
                true_past = batch_x[:, :-1, :]
                loss_recon = self.criterion(recon, true_past)
                
                # 3. Deviation Loss (结构偏差)
                loss_dev = 0
                if self.SLS is not None:
                    sls_tensor = torch.tensor(self.SLS).float().to(self.device)
                    # 计算当前距离矩阵与 SLS 的均方误差
                    loss_dev = torch.mean((dist_matrix - sls_tensor) ** 2)
                
                # 获取超参数权重
                lambda_recon = getattr(self.args, 'lambda_recon', 0.1)
                lambda_dev = getattr(self.args, 'lambda_dev', 3.0)
                
                # 总损失
                loss = loss_pred + lambda_recon * loss_recon + lambda_dev * loss_dev
                
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())

            # --- Epoch End: Update SLS ---
            # 计算本轮所有 Batch 的平均距离矩阵
            if len(self.dist_matrices_buffer) > 0:
                all_dists = np.concatenate(self.dist_matrices_buffer, axis=0)
                # axis=0 是 Batch 维度，取平均得到 [N, N] 的 SLS
                self.SLS = np.mean(all_dists, axis=0)
            
            print(f"Epoch: {epoch + 1}, Cost: {np.average(epoch_loss)}")
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # 保存最佳模型（此处简化为保存最后一个 epoch）
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(self.model.state_dict(), best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        
        preds = []
        trues = []
        dist_scores = []
        test_labels = [] 
        
        sls_tensor = torch.tensor(self.SLS).float().to(self.device) if self.SLS is not None else None

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                
                # ================= [关键修改] =================
                # batch_y shape 通常是 [Batch, Seq_Len]
                # 我们只取最后一个时间点的标签，以匹配 OracleAD 对 x^L 的预测
                if batch_y.dim() == 2:
                    true_label = batch_y[:, -1] # [Batch]
                elif batch_y.dim() == 3:
                    true_label = batch_y[:, -1, 0] # 防止有的loader返回 [Batch, Seq_Len, 1]
                else:
                    true_label = batch_y # Fallback
                
                test_labels.append(true_label.cpu().numpy())
                # ============================================
                
                pred, _, dist_matrix = self.model(batch_x)
                
                preds.append(pred.cpu().numpy())
                trues.append(batch_x[:, -1:, :].cpu().numpy())
                
                if sls_tensor is not None:
                    dev_score = torch.norm(dist_matrix - sls_tensor, p='fro', dim=(1, 2))
                    dist_scores.append(dev_score.cpu().numpy())
                else:
                    dist_scores.append(np.zeros(batch_x.shape[0]))

        # Concatenate
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        dist_scores = np.concatenate(dist_scores, axis=0)
        test_labels = np.concatenate(test_labels, axis=0).flatten()
        
        # P_score 计算
        p_score = np.mean(np.abs(preds - trues), axis=(1, 2))
        
        # Final Score
        final_score = p_score * dist_scores

        # 打印形状以进行调试验证 (现在两个应该都是 87832)
        print(f"Labels shape: {test_labels.shape}, Scores shape: {final_score.shape}")

        # --- Metrics Calculation ---
        print("Calculating metrics...")
        
        try:
            auc = roc_auc_score(test_labels, final_score)
            ap = average_precision_score(test_labels, final_score)
            print(f"AUC-ROC: {auc:.4f}, AUC-PR: {ap:.4f}")
        except Exception as e:
            print(f"AUC calculation failed: {e}")

        # Best F1 Score Search
        min_score, max_score = np.min(final_score), np.max(final_score)
        thresholds = np.linspace(min_score, max_score, 100)
        
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.0
        
        for thr in thresholds:
            pred_label = (final_score > thr).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_label, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = thr

        print("-" * 30)
        print(f"Best F1: {best_f1:.4f}")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall: {best_recall:.4f}")
        print(f"Threshold: {best_threshold:.4f}")
        print("-" * 30)

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'anomaly_scores.npy', final_score)
        np.save(folder_path + 'labels.npy', test_labels)
        
        return