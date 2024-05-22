import logging
import argparse
import math
import os
import sys
import time
import numpy as np
from time import strftime, localtime
import random
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchview import draw_graph

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

from data_utils_2 import build_tokenizer, build_embedding_matrix, Dataset

from models import AOA, AOA_2, AOA_3, AOA_4 , AOA_5,AOA_6,AOA_7, ABL, SELA, INFERSENT

import optuna
from optuna.trial import TrialState
import torch.optim as optim

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

TPC = ['tpc', 'tpc_2', 'tpc_3', 'mix','tpc_f','tpc_c1','tpc_full','tpc_full_2', 'tpc_f_2']

class Instructor:
    def __init__(self, opt, number=''):
        self.opt = opt
        if opt.dataset in TPC:
            fnames = [opt.dataset_file['train']]
        elif opt.dataset_file['val']==None:
            fnames = [opt.dataset_file['train'], opt.dataset_file['test']]
        else:
            fnames = [opt.dataset_file['train'], opt.dataset_file['val'], opt.dataset_file['test']]
        tokenizer = build_tokenizer(
            fnames,
            max_seq_len=opt.max_seq_len,
            dat_fname='{0}_{1}_tokenizer.dat'.format(opt.dataset, number))
        embedding_matrix = build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{0}_{1}_{2}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset, number))
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print(opt.device)

        self.trainset = Dataset(opt.dataset_file['train'], tokenizer, dat_fname='{0}_2_train.dat'.format(opt.dataset))
        # self.weight_classes =torch.tensor( compute_class_weight('balanced', np.unique([i['polarity'] for i in self.trainset.data]), self.trainset[4]), dtype = torch.float).to(self.opt.device)
        # self.valset = ABSADataset(opt.dataset_file['val'], tokenizer)self.trainset[4]
        if opt.K_Fold!='true':
        
            if opt.dataset not in TPC:
                self.testset = Dataset(opt.dataset_file['test'], tokenizer,dat_fname='{0}_2_test.dat'.format(opt.dataset))
            assert 0 <= opt.valset_ratio < 1
            if opt.valset_ratio > 0:
                valset_len = int(len(self.trainset) * opt.valset_ratio)
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
            else:
                self.valset = Dataset(opt.dataset_file['val'], tokenizer,dat_fname='{0}_2_val.dat'.format(opt.dataset))
        
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, lrshrink, minlr, number = ''):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        epoch = 1
        adam_stop = False
        stop_training = False

        while not stop_training and epoch <= self.opt.num_epoch:
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            ts = time.time()
            for i_batch, sample_batched in enumerate(train_data_loader):
                
                global_step += 1
                # clear gradient accumulators
                self.model.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['class_n'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            
            val_acc, val_f1, report_val = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, report_val['class1']['f1-score']))
            logger.info('> report: {}'.format(report_val['class1']))
            print('timepassed:')
            logger.info('> timepassed: {}'.format((time.time()-ts)))
            
            epoch +=1

            if report_val['class1']['f1-score'] > max_val_acc:
                max_val_acc = report_val['class1']['f1-score']

                if not os.path.exists('state_dict_2/'+self.opt.dataset):
                    os.makedirs('state_dict_2/'+self.opt.dataset)
                path = 'state_dict_2/'+self.opt.dataset+'/{0}_{1}_{2}_val_f1_{3}'.format(number,self.opt.model_name, self.opt.dataset, round(report_val['class1']['f1-score'], 4))
                

                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            else:
                if 'sgd' in self.opt.optimizer_rec:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lrshrink
                    logger.info('Shrinking lr by : {0}. New lr = {1}'
                          .format(lrshrink,
                                  optimizer.param_groups[0]['lr']))

                    if optimizer.param_groups[0]['lr'] < minlr:
                        stop_training = True
                if 'adam' in self.opt.optimizer_rec:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                

        return path
    
    
    
    def _train_cross_val(self, criterion, optimizer, dataset, number):
        
        global_step = 0
        path = None
        
        k=5
        splits=KFold(n_splits=k,shuffle=True,random_state=42)

        pathes = []
        
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            if path is not None:
                self.model.load_state_dict(state_dict = torch.load(path))
            #train_set = torch.utils.data.Subset(dataset, train_idx)
            #val_set = torch.utils.data.Subset(dataset,val_idx)
            train_set = dataset.get_subset(train_idx)
            val_set = dataset.get_subset(val_idx)
            
            temp = [t['class_n'] for t in train_set.data]
            

            class_sample_count = np.array([len(np.where(temp == t)[0]) for t in np.unique(temp)])
            weight_temp = 1. / class_sample_count
            samples_weight = np.array([weight_temp[t] for t in temp])
            
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
 
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([len(train_set.data)/t for t in class_sample_count],dtype=torch.float).to(self.opt.device))
            
            train_loader = DataLoader(train_set, batch_size=self.opt.batch_size, shuffle=False, sampler=sampler)
            val_loader = DataLoader(val_set, batch_size=self.opt.batch_size, shuffle=False)
            
            total_val_acc = []
            total_val_f1 = []
            max_val_acc = 0
            max_val_f1 = 0
            epoch = 1
            adam_stop = False
            stop_training = False
            
            
            while not stop_training and epoch <= self.opt.num_epoch:
                logger.info('>' * 100)
                logger.info('epoch: {}'.format(epoch))
                n_correct, n_total, loss_total = 0, 0, 0
                # switch model to training mode
                self.model.train()
                ts = time.time()
                for i_batch, sample_batched in enumerate(train_loader):

                    global_step += 1
                    # clear gradient accumulators
                    self.model.zero_grad()
                    inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                    outputs = self.model(inputs)
                    targets = sample_batched['class_n'].to(self.opt.device)
                    
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
        
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    loss_total += loss.item() * len(outputs)
                    if global_step % self.opt.log_step == 0:
                        train_acc = n_correct / n_total
                        train_loss = loss_total / n_total
                        logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                
                val_acc, val_f1 = self._evaluate_acc_f1(val_loader)
                total_val_acc.append(val_acc)
                total_val_f1.append(val_f1)
                    
                logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
                #logger.info('> report: {}'.format(report_val['class1']))
                print('timepassed:')
                logger.info('> timepassed: {}'.format((time.time()-ts)))
                
                epoch +=1
                    
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
            
                    if not os.path.exists('state_dict_3/'+self.opt.dataset):
                        os.makedirs('state_dict_3/'+self.opt.dataset)
                    path = 'state_dict_3/'+self.opt.dataset+'/{0}_{1}_fold_{2}_val_f1_{3}'.format(self.opt.model_name, self.opt.dataset+number,str(fold), round(max_val_f1, 4))
                    
                
                    torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))
                """
                ДЛЯ ОТРИСОВКИ МОДЕЛИ В png
                
                
                draw_graph(self.model, input_data=[inputs], expand_nested=True, 
                                         save_graph = True, filename = "graph_aoa_6.png", device = self.opt.device)
                """
            logger.info('> total_val_acc: {:.4f}, total_val_f1: {:.4f}'.format(np.mean(total_val_acc), np.mean(total_val_f1)))
            if path is not None:
                pathes.append(path)
                    

        return pathes
    


    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['class_n'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
        if self.opt.K_Fold!='true':
            
            try:
                report = classification_report(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), target_names=['class0','class1'], output_dict=True, digits = 4)
            except:
                print('Fail classification')
                report = []
            
            # print(report)
            return acc, f1, report
        else:
            return acc, f1
    

    def run(self, number=''):
        
            # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        if self.opt.K_Fold!='true':
            
            if self.opt.valset_ratio==0:
                temp = [t['class_n'] for t in self.trainset.data]
            else:
        	    temp = [t['class_n'] for t in self.trainset.dataset.data]
            class_sample_count = np.array([len(np.where(temp == t)[0]) for t in np.unique(temp)])
            weight_temp = 1. / class_sample_count
            samples_weight = np.array([weight_temp[t] for t in temp])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
            
            if self.opt.valset_ratio==0:
                train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, sampler = sampler)	
                test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
                val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
            else:
                train_data_loader = DataLoader(dataset=self.trainset.dataset, batch_size=self.opt.batch_size, sampler = sampler)
                if self.opt.dataset not in TPC:
                    test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
                val_data_loader = DataLoader(dataset=self.valset.dataset, batch_size=self.opt.batch_size, shuffle=False)
            
            self._reset_params()
            best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, self.opt.lrshrink, self.opt.minlr, number)
            
            logger.info('best_model_path: ' + best_model_path)
            if self.opt.dataset not in TPC:
                try:
                    self.model.load_state_dict(torch.load(best_model_path))
                except:
                    logger.info('no best')
                self.model.eval()
                test_acc, test_f1 , report_test = self._evaluate_acc_f1(test_data_loader)
                logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
                logger.info('> report: {}'.format(report_test))
            
        else:
            self._reset_params()
            best_model_path = self._train_cross_val(criterion, optimizer, self.trainset, number)
            for i in best_model_path:
                print(i)
            
        return best_model_path
        # print(report_test)
        # logger.info('>> report: {:.4f}'.format(report_test))
        

    def objective(self, trial):
        model = self.model
        
        
        global_step = 0
        epoch = 1
        
            # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, model.parameters())
        #optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam","SGD"])
        lr = trial.suggest_float("lr", 1e-10, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(_params, lr=lr, weight_decay = weight_decay)
    
        while epoch <= self.opt.num_epoch:
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            model.train()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                
                global_step += 1
                # clear gradient accumulators
                model.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = model(inputs)
                targets = sample_batched['class_n'].to(self.opt.device)
    
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            
            val_acc, val_f1, report_val = self._evaluate_acc_f1(self.val_data_loader)
            
            trial.report(val_f1, epoch)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            epoch +=1
    
            
                
        return val_f1
    
    
    def run_objective(self, number=''):
        
         
        
        if self.opt.valset_ratio==0:
            temp = [t['class_n'] for t in self.trainset.data]
        else:
    	    temp = [t['class_n'] for t in self.trainset.dataset.data]
        class_sample_count = np.array([len(np.where(temp == t)[0]) for t in np.unique(temp)])
        weight_temp = 1. / class_sample_count
        samples_weight = np.array([weight_temp[t] for t in temp])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
        
        train_data_loader = DataLoader(dataset=self.trainset.dataset, batch_size=self.opt.batch_size, sampler = sampler)
        val_data_loader = DataLoader(dataset=self.valset.dataset, batch_size=self.opt.batch_size, shuffle=False)
        
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        
        self._reset_params()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, timeout=600)
    
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    
        print("Best trial:")
        trial = study.best_trial
     
        print("  Value: ", trial.value)
    
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
    
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aoa_6', type=str, help = 'aoa for AOA, abl for Ablation, sela for selfattention')
    parser.add_argument('--dataset', default='tpc_2', type=str, help='tpc, tpc_2, tpc_3,tpc_f,tpc_c1')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-03, type=float, help='8e-1')
    parser.add_argument('--l2reg', default=1.5e-05, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='default 20')
    parser.add_argument('--batch_size', default=64, type=int, help='default 64')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=240, type=int)
    parser.add_argument('--class_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2345, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--K_Fold', default='true', type=str)
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='default cdm')
    parser.add_argument('--SRD', default=3, type=int, help='default 3')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'aoa_7':AOA_7,
        'aoa_6':AOA_6,
        'aoa_5':AOA_5,
        'aoa_4':AOA_4,
        'aoa_3':AOA_3,
        'aoa_2':AOA_2,
        'aoa': AOA,
        'abl':ABL,
        'sela': SELA,
        'infersent': INFERSENT
    }
    dataset_files = {
        
    'tpc_full':{
            'train':'./datasets/with_feature_extraction_2/train_tpc10.p'
            },
    'tpc_2':{
        'train':'./datasets/omap/train_tpc.p'
        },
    'tpc':{
        'train':'./datasets/omap/TPC-DI_1.p'
        
        },
    'nom':{
        'train': './datasets/omap/train_nominal.p',
        'val': './datasets/omap/val_nominal.p',
           'test': './datasets/omap/test_nominal.p'
        },
    'num':{
        'train': './datasets/omap/train_numeric.p',
        'val': './datasets/omap/val_numeric.p',
           'test': './datasets/omap/test_numeric.p'
        },
    'nom2':{
        'train': './datasets/omap/train_nominal2.p',
        'val': './datasets/omap/val_nominal2.p',
           'test': './datasets/omap/test_nominal2.p'
        },
    'num2':{
        'train': './datasets/omap/train_numeric2.p',
        'val': './datasets/omap/val_numeric2.p',
           'test': './datasets/omap/test_numeric2.p'
        },

    }
    input_colses = {
        'aoa': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'aoa_2': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'aoa_3': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2', 'constraints'],
        'aoa_4': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2', 'constraints'],
        'aoa_5': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2', 'constraints'],
        'aoa_6': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2', 'constraints'],
        'aoa_7': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2', 'constraints'],
        'infersent': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'sela': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
        'abl': ['text_raw_indices1', 'aspect_indices1','text_raw_indices2', 'aspect_indices2'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw':torch.optim.AdamW,
    }
    constr_dim = {
        'num':8,
        'nom':6,
        'num2':21,
        'nom2':33,
        'tpc':28,
        'tpc_2':28,
        'tpc_3':28,
        'mix':28,
        'tpc_f':30,
        'tpc_c1':2,
        'tpc_full':34,
        'tpc_full_2':28,
        'tpc_f_2':34
        }
    opt.constr_dim = constr_dim[opt.dataset]
    
    opt.model_class = model_classes[opt.model_name]
    if opt.dataset!='tpc_f_2' and opt.dataset!='tpc_3' and opt.dataset!='mix' and opt.dataset!='tpc_f' and opt.dataset!='tpc_c1' and opt.dataset!='tpc_full_2':
        opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer_rec = opt.optimizer
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('logs/'+opt.dataset):
    	os.makedirs('logs/'+opt.dataset)   

    log_file ='./logs/'+opt.dataset +'/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file))
    if opt.dataset=='tpc_3':
        pathes=[]
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/omap/train_tpc{}.p'.format(i)}
            ins = Instructor(opt,number)
            pathes.append(ins.run(number))
        with open('pathes_last.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='tpc_f_2':
        pathes=[]
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/with_feature_extraction_2/train_tpc{}.p'.format(i)}
            ins = Instructor(opt,number)
            pathes.append(ins.run(number))
        with open('pathes_tpc_f_2.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='mix':
        pathes=[]
        for i in range(20):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            try:
                opt.dataset_file = {'train':'./datasets/omap/train_mix{}.p'.format(i)}
                ins = Instructor(opt,number)
                pathes.append(ins.run(number))
            except:
                logger.info('No such file')
        with open('pathes_mix.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='tpc_f':
        pathes=[]
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/with_feature_extraction/train_tpc{}.p'.format(i)}
            ins = Instructor(opt,number)
            pathes.append(ins.run(number))
        with open('pathes_with_feature_extraction.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='tpc_c1':
        pathes=[]
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/with_1_constraint/train_tpc{}.p'.format(i)}
            ins = Instructor(opt,number)
            pathes.append(ins.run(number))
        with open('pathes_with_1_constraint.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='tpc_full_2':
        pathes=[]
        for i in range(2, 20):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/full_data/train_tpc{}.p'.format(i)}
            ins = Instructor(opt,number)
            pathes.append(ins.run(number))
        with open('pathes_full_data.p', 'wb') as file:
            pickle.dump(pathes, file)
    elif opt.dataset=='tpc_full':
        ins = Instructor(opt)
        ins.run_objective()
    else:
        ins = Instructor(opt)
        ins.run()


if __name__ == '__main__':
    main()
