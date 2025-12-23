import torch
import torch.nn as nn
import os
import time
import numpy as np
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Exp_GCAD(Exp_Basic):
    def __init__(self, args):
        super(Exp_GCAD, self).__init__(args)
        self.beta = 0.5  
        self.threshold = 0.1 
        self.A_norm = None

    def _build_model(self):
        model = self.model_dict['GCAD'].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # ================= 核心：计算误差梯度 =================
    def get_causality_matrix(self, x_input, y_true, model):
        x_input.requires_grad = True
        
        # 1. Forward
        pred = model(x_input, None, None, None) # [Batch, Channels]
        
        # Ensure y_true has correct shape [Batch, Channels]
        if y_true.ndim == 3:
            y_true = y_true[:, -1, :]

        batch_size, seq_len, channels = x_input.shape
        causality_matrix = torch.zeros(batch_size, channels, channels).to(x_input.device)
        
        # 2. Gradient of Prediction Error
        diff = pred - y_true
        
        for j in range(channels):
            loss_j = (diff[:, j]) ** 2 
            
            # Calculate gradient w.r.t input
            grad = torch.autograd.grad(
                outputs=loss_j, 
                inputs=x_input, 
                grad_outputs=torch.ones_like(loss_j),
                retain_graph=True,
                create_graph=False,
                only_inputs=True
            )[0] 
            
            # Sum absolute gradients over time
            causal_impact = torch.sum(torch.abs(grad), dim=1) 
            causality_matrix[:, :, j] = causal_impact

        return causality_matrix

    def sparsify_graph(self, matrix):
        # Remove bidirectional symmetry
        matrix_t = matrix.transpose(1, 2)
        diff = matrix - matrix_t
        sparsified = torch.relu(diff)
        # Thresholding
        sparsified[sparsified < self.threshold] = 0
        return sparsified

    # ================= 训练 (自监督) =================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_loss = []
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                # Ignore label batch_y, construct feature target
                target = batch_x[:, -1, :] 

                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, target)
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                target = batch_x[:, -1, :]
                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, target)
                total_loss.append(loss.item())
        return np.average(total_loss)

    # ================= 校准 =================
    def calibrate(self):
        print("Calibrating normal causal patterns...")
        train_data, train_loader = self._get_data(flag='train')
        self.model.eval()
        
        causal_matrices = []
        sample_count = 0
        max_samples = 500
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if sample_count >= max_samples: break
            batch_x = batch_x.float().to(self.device)
            
            # Construct feature target
            y_true = batch_x[:, -1, :]
            
            with torch.enable_grad():
                matrix = self.get_causality_matrix(batch_x, y_true, self.model)
                sparsified = self.sparsify_graph(matrix)
                
            causal_matrices.append(sparsified.detach().cpu())
            sample_count += batch_x.shape[0]
            
        all_matrices = torch.cat(causal_matrices, dim=0)
        self.A_norm = torch.mean(all_matrices, dim=0).to(self.device)
        print("Calibration finished.")

    # ================= 辅助函数: Point Adjustment =================
    def anomaly_adjustment(self, pred, gt):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0: break
                    else:
                        if pred[j] == 0: pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0: break
                    else:
                        if pred[j] == 0: pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred

    # ================= 测试 =================
    def test(self, setting, test=0):
        self.calibrate()
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        anomaly_scores = []
        print("Testing...")
        
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_x = batch_x.float().to(self.device)
            y_true_feat = batch_x[:, -1, :] 
            
            batch_x.requires_grad = True
            
            with torch.enable_grad():
                matrix = self.get_causality_matrix(batch_x, y_true_feat, self.model)
                A_test = self.sparsify_graph(matrix)
            
            # Scores
            A_norm_batch = self.A_norm.unsqueeze(0).expand_as(A_test)
            epsilon = 1e-7
            diff = torch.abs(A_test - A_norm_batch)
            denominator = A_norm_batch + epsilon
            
            S_c = torch.sum(diff / denominator, dim=(1, 2))
            
            diag_diff = torch.diagonal(diff, dim1=1, dim2=2)
            diag_denom = torch.diagonal(denominator, dim1=1, dim2=2)
            S_t = torch.sum(diag_diff / diag_denom, dim=1)
            
            # DEBUG PRINT (Placed safely after calculation)
            if i == 0:
                print("\n>>> DEBUG INFO <<<")
                print(f"A_test max: {A_test.max().item():.6f}")
                print(f"S_c sample: {S_c[0].item():.4f}")
                print(f"S_t sample: {S_t[0].item():.4f}")
                print(">>> END DEBUG <<<\n")
            
            batch_score = S_c + self.beta * S_t
            anomaly_scores.append(batch_score.detach().cpu().numpy())

        # Evaluation
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        anomaly_scores = np.nan_to_num(anomaly_scores)
        
        gt_labels = test_data.test_labels
        if len(gt_labels) > len(anomaly_scores):
            gt_labels = gt_labels[-len(anomaly_scores):]
        else:
            anomaly_scores = anomaly_scores[-len(gt_labels):]

        # MinMax Scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        anomaly_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).ravel()

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(gt_labels, anomaly_scores)
            print(f"Test AUC: {auc:.4f}")
        except:
            print("AUC Error")

        # Threshold Search & Point Adjustment
        print("Searching best threshold...")
        best_f1 = 0.0
        best_results = []
        
        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
        for threshold in thresholds:
            pred_labels = (anomaly_scores > threshold).astype(int)
            pred_adjusted = self.anomaly_adjustment(pred_labels.copy(), gt_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, pred_adjusted, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_results = [precision, recall, f1, threshold]

        print(">>>>>>> Final Results <<<<<<<")
        if best_results:
            print(f"Best Threshold: {best_results[3]:.4f}")
            print(f"Precision: {best_results[0]:.4f}")
            print(f"Recall:    {best_results[1]:.4f}")
            print(f"F1-Score:  {best_results[2]:.4f}")
        
        return