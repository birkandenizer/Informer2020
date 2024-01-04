import argparse
import os
import torch
import wandb

from exp.exp_informer import Exp_Informer

from torchinfo import summary

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--scaler', type=str, default='standard',help='feature scaler')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    '4G_mm15':{'data':'MM15.csv','T':'bandwidth','M':[8,8,8],'S':[1,1,1],'MS':[8,8,1]}, # NYU-METS
    '4G_tt7':{'data':'TT7.csv','T':'bandwidth','M':[8,8,8],'S':[1,1,1],'MS':[8,8,1]}, # NYU-METS
    '4G_bus':{'data':'BUS_LINES.csv','T':'bandwidth','M':[8,8,8],'S':[1,1,1],'MS':[8,8,1]}, # NYU-METS
    #'4G':{'data':'car.csv','T':'DL_bitrate','M':[11,11,11],'S':[1,1,1],'MS':[11,11,1]}, # Beyond4G
    '5G_beyond':{'data':'Download-limited.csv','T':'DL_bitrate','M':[9,9,9],'S':[1,1,1],'MS':[9,9,1]}, #Beyond5G
    '5G_berlin':{'data':'filtered_data_downlink_selected.csv','T':'datarate','M':[17,17,17],'S':[1,1,1],'MS':[17,17,1]}, # BerlinV2X
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

""" for ii in range(args.itr):
    
    wandb.login()
    
    run = wandb.init(project="time-series-informer", config=args)

    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, run)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

    wandb.finish() """

sweep_configuration = {
        'method': 'bayes', # grid, random, bayes
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_total_loss'}, # val_total_loss, val_loss
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3},
        'parameters': {
            'features': {'values': ['MS']},
            'freq': {'values': ['s']},
            'seq_len': {'values': [64]}, # 16, 32, 48, 64
            'label_len': {'values': [32]}, # 4, 8, 16
            'pred_len': {'values': [1]},

            'd_model': {'values': [1024]}, # 256, 512, 1024, 2048
            'n_heads': {'values': [8]}, # 4, 6, 8
            'e_layers': {'values': [3]}, # 1, 2, 3
            'd_layers': {'values': [1]}, # 1, 2, 3
            'd_ff': {'values': [2048]}, #256, 512, 1024, 2048

            'dropout': {'values': [0.05]},
            #'dropout': {'min': 0.01, 'max': 0.1},
            'attn': {'values': ['prob']}, # prob, full
            'embed': {'values': ['timeF']}, # timeF, 'fixed', 'learned'
            'activation': {'values': ['gelu']}, #'gelu', 'relu', 'LeakyReLU'??

            'itr': {'values': [1]},
            'train_epochs': {'values': [20]},
            'batch_size': {'values': [32]},
            'patience': {'values': [3]},
            'learning_rate': {'values': [0.0001]}, 
            #'learning_rate': {'min': 0.0001, 'max': 0.01},
            'loss': {'values': ['mse']}, # l1
            'scaler': {'values': ['standard']}, #'standard', 'minmax'
            'optimizer': {'values': ['Adam']}, # 'Adam', 'AdamW'
            #'lr_scheduler': {'values': ['StepLR', 'ReduceLROnPlateau']},
        },
        'run_cap' : 1000
    }
sweep_id = wandb.sweep(sweep=sweep_configuration, project='time-series-informer')

def wandb_train(config=None):

    for ii in range(args.itr):

        wandb.login()
    
        run = wandb.init(config=args)
        print(f'wandb config {wandb.config}')
        print(f'args {args}')
        #args = wandb.config

        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(wandb.config.model, wandb.config.data, wandb.config.features, 
                    wandb.config.seq_len, wandb.config.label_len, wandb.config.pred_len,
                    wandb.config.d_model, wandb.config.n_heads, wandb.config.e_layers, wandb.config.d_layers, wandb.config.d_ff, wandb.config.attn, wandb.config.factor, 
                    wandb.config.embed, wandb.config.distil, wandb.config.mix, wandb.config.des, ii)
        
        print('wandb.config in experiment:')
        print(wandb.config)

        #exp = Exp(args) # set experiments
        exp = Exp(wandb.config) # set experiments

        summry_model = exp.model
        batch_size = 16
        #summary(summry_model, input_size=(batch_size, 1, 28, 28))
        print(summary(summry_model, verbose=1))

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, run)

        torch.cuda.empty_cache()

        wandb.finish()


wandb.agent(sweep_id, function=wandb_train, count=1)