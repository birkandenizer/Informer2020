from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, InformerC

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, classification_metric, FocalLoss
from torchvision.ops import sigmoid_focal_loss

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import wandb

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerc':InformerC,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerc' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' or self.args.model=='informerc' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            '4G_mm15':Dataset_Custom,
            '4G_tt7':Dataset_Custom,
            '4G_bus':Dataset_Custom,
            '5G_beyond':Dataset_Custom,
            '5G_berlin':Dataset_Custom,
            '5G_addix':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            scaler=args.scaler,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer=='Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        else: #self.args.optimizer=='AdamW'
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim
    
    def _select_criterion(self):
        if self.args.loss=='mse':
            print('model informer - MSELoss')
            criterion =  nn.MSELoss()

        if self.args.loss=='l1':
            print('model informer - L1Loss')
            criterion = nn.L1Loss()
        
        if self.args.model=='informerc':
            print('model informerc - CrossEntropyLoss')
            criterion = nn.CrossEntropyLoss()
            #criterion = nn.BCELoss() # Balanced Cross-Entropy Loss # Gives error
            #criterion = FocalLoss()
            #criterion = sigmoid_focal_loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            wandb.log({"val_loss": loss})
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        wandb.log({"val_total_loss": total_loss})
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                #print(f'type of pred: {type(pred)}, type of true: {type(true)}')
                #print(f'size of pred: {pred.size()}, size of true: {true.size()}')
                #print(f'dtype of pred: {pred.dtype}, dtype of true: {true.dtype}')
                
                #print(pred)
                #print(true)

                loss = criterion(pred, true)
                wandb.log({"train_loss": loss})
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, run):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        
        print('dtype:', preds.dtype, trues.dtype)
        print('shape:', preds.shape, trues.shape)
        #preds.shape[-1]    
        #preds.shape[-2]
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('reshape dtype:', preds.dtype, trues.dtype)
        print('reshape shape:', preds.shape, trues.shape)
        #print('preds[0] shape and value:', preds[0].shape, preds[0])

        """ if self.args.inverse:
            inverse_preds = []
            inverse_trues = []

            for pred in preds:
                inverse_preds.append(test_data.inverse_transform(pred))

            for true in trues:
                inverse_trues.append(test_data.inverse_transform(true))

            inverse_preds = np.array(inverse_preds)
            inverse_trues = np.array(inverse_trues)

            print('inverse dtype:', inverse_preds.dtype, inverse_trues.dtype)
            print('inverse shape:', inverse_preds.shape, inverse_trues.shape)

            preds = inverse_preds
            trues = inverse_trues """

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        m_n='BandFormer'
        print(f'{m_n} rmse:{rmse}, mae:{mae}, mse:{mse}')
        wandb.log({m_n+" rmse": rmse, m_n+" mae": mae, m_n+" mse": mse})

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        artifact = wandb.Artifact(name="informer-predictions", type="data")
        artifact.add_file(local_path=folder_path+'pred.npy')
        run.log_artifact(artifact)

        artifact = wandb.Artifact(name="informer-labels", type="data")
        artifact.add_file(local_path=folder_path+'true.npy')
        run.log_artifact(artifact)

        if False:#self.args.pred_len == 1:
            shift_by = 1
            span = 8

            df = pd.DataFrame({'trues': trues.flatten(), 'preds': preds.flatten()})
            df['shifted'] = df['trues'].shift(shift_by, fill_value=0)
            #df['sma']     = df['trues'].rolling(span).mean()
            df['ewma8']   = df['trues'].ewm(span=span, adjust=True).mean()

            mae, mse, rmse, mape, mspe = metric(df['shifted'].to_numpy(), df['trues'].to_numpy())
            m_n='Shifted'
            print(f'{m_n} rmse:{rmse}, mae:{mae}, mse:{mse}')
            wandb.log({m_n+" rmse": rmse, m_n+" mae": mae, m_n+" mse": mse})

            """ mae, mse, rmse, mape, mspe = metric(df['sma'].to_numpy(), trues)
            m_n='SMA'
            print(f'{m_n} rmse:{rmse}, mae:{mae}, mse:{mse}')
            wandb.log({m_n+" rmse": rmse, m_n+" mae": mae, m_n+" mse": mse}) """

            mae, mse, rmse, mape, mspe = metric(df['ewma8'].to_numpy(), df['trues'].to_numpy())
            m_n='EWMA8'
            print(f'{m_n} rmse:{rmse}, mae:{mae}, mse:{mse}')
            wandb.log({m_n+" rmse": rmse, m_n+" mae": mae, m_n+" mse": mse})

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        #print(f'batch_x dtype: {batch_x.dtype}, shape: {batch_x.shape}')
        #batch_x dtype: torch.float32, shape: torch.Size([32, 5, 8])
        batch_y = batch_y.float()
        #print(f'batch_y before dtype: {batch_y.dtype}, shape: {batch_y.shape}')
        #batch_y before dtype: torch.float32, shape: torch.Size([32, 6, 8])

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        #print(f'output.size: {outputs.size()}')
        #print(outputs)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            """ outputs_concat = []
            for element in outputs:
                outputs_concat.append(dataset_object.inverse_transform(element))
            outputs = torch.cat(outputs_concat)
            outputs_concat = [] """
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        #print(f'batch_y after dtype: {batch_y.dtype}, shape: {batch_y.shape}')
        #batch_y after dtype: torch.float32, shape: torch.Size([32, 1, 1])

        return outputs, batch_y
