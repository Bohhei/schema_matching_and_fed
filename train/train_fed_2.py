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
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
#from torchview import draw_graph

from sklearn import metrics
from sklearn.model_selection import KFold

from data_utils_2 import build_tokenizer, build_embedding_matrix, Dataset

from models import AOA, AOA_2, AOA_3, AOA_4, AOA_5 ,AOA_6, ABL, SELA, INFERSENT

import flwr as fl
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

TPC = ['tpc', 'tpc_2', 'tpc_3', 'mix', 'tpc_f','tpc_c1']
NUM_CLIENTS = 4

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
        self.embedding_matrix = embedding_matrix
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print(opt.device)

        self.trainset = Dataset(opt.dataset_file['train'], tokenizer, dat_fname='{0}_2_train.dat'.format(opt.dataset))
        # разбиение на NUM_CLIENTS частей для федеративного обучения
        self.trainloaders, self.valloaders = split_dataset(self.trainset,self.opt, NUM_CLIENTS)
        
        #self.valset = self.trainset
        #self.trainsets = self.trainsets[:-1]
        
        self.number = number
        
        
        
        if opt.device.type == 'cuda':
            print("device count ", torch.cuda.device_count())
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
            
def split_dataset(full_dataset, opt, num_clients):
    # Split training set into `num_clients` partitions to simulate different local datasets
    
    trainset = full_dataset
    #train_size = int(0.8 * len(full_dataset))
    #test_size = len(full_dataset) - train_size
    #trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] = len(trainset) - partition_size * (num_clients-1)
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        #len_val = len(ds) // 50  # 10 % validation set
        #len_train = len(ds) - len_val
        #lengths = [len_train, len_val]
        #ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds, batch_size=opt.batch_size, shuffle=True))
        valloaders.append(DataLoader(ds, batch_size=opt.batch_size))
    #testloader = DataLoader(testset, batch_size=opt.batch_size)
    return trainloaders, valloaders#, testloader

def reset_params(model, opt):
    for child in model.children():
        for p in child.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)



def train_cross_val(model,criterion, optimizer, opt, number_of_client, dataset, lrshrink, minlr, number, num_epoch=3):
    #max_val_acc = 0
    max_val_f1 = 0
    global_step = 0
    path = None
    epoch = 1
    adam_stop = False
    stop_training = False
    
    k=10
    splits=KFold(n_splits=k,shuffle=True,random_state=42)

    while not stop_training and epoch <= num_epoch:
        total_val_acc = []
        total_val_f1 = []
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            
            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set = torch.utils.data.Subset(dataset,val_idx)
            #train_set = dataset.get_subset(train_idx)
            #val_set = dataset.get_subset(val_idx)
            
            temp = [t['class_n'] for t in train_set]
            
            class_sample_count = np.array([len(np.where(temp == t)[0]) for t in np.unique(temp)])
            weight_temp = 1. / class_sample_count
            samples_weight = np.array([weight_temp[t] for t in temp])
            
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
            
            
            train_loader = DataLoader(train_set, batch_size=opt.batch_size, sampler=sampler)
            val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
            
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            model.train()
            ts = time.time()
            for i_batch, sample_batched in enumerate(train_loader):
                
                global_step += 1
                # clear gradient accumulators
                model.zero_grad()
                inputs = [sample_batched[col].to(opt.device) for col in opt.inputs_cols]
                outputs = model(inputs)
                targets = sample_batched['class_n'].to(opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            
            val_loss, val_acc= evaluate_acc_f1(model, opt, criterion, val_loader)
            total_val_acc.append(val_acc)
            #total_val_f1.append(val_f1)
        
        
        epoch +=1



def evaluate_acc_f1(model, opt, criterion, data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    loss = 0.0
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col].to(opt.device) for col in opt.inputs_cols]
            t_targets = t_sample_batched['class_n'].to(opt.device)
            t_outputs = model(t_inputs)

            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)

            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

    loss = criterion(t_outputs_all, t_targets_all)
    loss /=len(data_loader.dataset)
    acc = n_correct / n_total
    #f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')
   
    return loss, acc
    
def run(opt, number = ""):
    ins = Instructor(opt, number)
    
    #федеративное обучение
    
    params = get_parameters(ins.model)
    
    
    
    print(' initialization --------  ')
    
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
        ):
            """Aggregate model weights using weighted average and store checkpoint"""

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(ins.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

                # Save the model
                if not os.path.exists('state_dict_2/'+opt.dataset):
                    os.makedirs('state_dict_2/'+opt.dataset)
                torch.save(state_dict, f"state_dict_2/{opt.dataset}/model_{number}_round_{server_round}.pth")
            
            

            return aggregated_parameters, aggregated_metrics
        
        def aggregate_evaluate(
            self,
            server_round: int,
            results,
            failures,
        ):
            """Aggregate evaluation accuracy using weighted average."""
        
            if not results:
                return None, {}
        
            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
        
            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        
            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    def fit_config(server_round: int):
        """Return training configuration dict for each round.
    
        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": 1,  
        }
        return config
    # Pass parameters to the Strategy for server-side parameter initialization
    strategy = SaveModelStrategy(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        #evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
    )
    print( ' initialization YES')
    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    if ins.opt.device.type == 'cuda':
        print(ins.opt.device)
        client_resources = {"num_cpus": 20,"num_gpus": 1}
    else:
        client_resources = None
        
    def client_fn(cid):
        model = opt.model_class(ins.embedding_matrix, ins.opt).to(ins.opt.device)
        
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = ins.opt.optimizer(_params, lr=ins.opt.learning_rate, weight_decay=ins.opt.l2reg)
        
        reset_params(model, ins.opt)
        
        valloader = ins.valloaders[int(cid)]
        trainloader = ins.trainloaders[int(cid)]
        return FlowerClient(cid, model, criterion, optimizer, ins.opt, trainloader, valloader, ins.number).to_client()
    
    fl.common.logger.configure(identifier="Experiment", filename=f"log_{number}.txt")
    print('Start simulation')
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),#10
        strategy=strategy,
        client_resources=client_resources,
    )

        
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, criterion, optimizer,opt , trainloader, valloader, number):
        self.cid = cid
        self.model = model
        self.trainset = trainloader.dataset
        self.valloader = valloader
        self.opt = opt
        self.number = number
        self.criterion = criterion
        self.optimizer = optimizer

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client {self.cid}] fit, config: {config}")
        try:
            set_parameters(self.model, parameters)
        except:
            print("FAIL LOAD FIT")
        train_cross_val(self.model,self.criterion, self.optimizer, self.opt, self.cid, self.trainset, self.opt.lrshrink, self.opt.minlr, self.number, num_epoch=local_epochs)
        return get_parameters(self.model), 1, {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        try:
            set_parameters(self.model, parameters)
        except:
            print("FAIL LOAD EVALUATE")
        criterion = nn.CrossEntropyLoss()
        loss, accuracy= evaluate_acc_f1(self.model, self.opt, criterion, self.valloader)
        return float(loss), 1, {"accuracy": float(accuracy)}



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aoa_5', type=str, help = 'aoa for AOA, abl for Ablation, sela for selfattention')
    parser.add_argument('--dataset', default='tpc_2', type=str)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=8e-1, type=float, help='1e-3')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='default 30')
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
        'tpc_c1':2
        }
    opt.constr_dim = constr_dim[opt.dataset]
    
    opt.model_class = model_classes[opt.model_name]
    if opt.dataset!='tpc_3' and opt.dataset!='mix' and opt.dataset!='tpc_f' and opt.dataset!='tpc_c1':
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
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            try:
                opt.dataset_file = {'train':'./datasets/omap/train_tpc{}.p'.format(i)}
                run(opt, number)
            except:
                logger.info('No such file')
    elif opt.dataset=='tpc_f':
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/with_feature_extraction/train_tpc{}.p'.format(i)}
            run(opt, number)
    elif opt.dataset=='tpc_c1':
        for i in range(1,17):
            number = str(i)
            logger.info('ITERATION: {}'.format(i))
            opt.dataset_file = {'train':'./datasets/with_1_constraint/train_tpc{}.p'.format(i)}
            run(opt, number)
    else:
        run(opt)

if __name__ == '__main__':
    main()
