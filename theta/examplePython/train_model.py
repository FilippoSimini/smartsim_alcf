import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import sys
import pandas as pd

from smartredis import Client

# Horovod: import horovod 
import horovod.torch as hvd
from horovod.torch.mpi_ops import Sum
# Horovod: initialize library.
hvd.init()
hrank = hvd.rank()
hsize = hvd.size()

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Training settings
parser = argparse.ArgumentParser(description='train_example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)
# model args
parser.add_argument('--num_db_tensors', type=int, default=1,
                    help='number of tensors loaded onto the db by the simulation ranks (default: 1)')
parser.add_argument('--dim_db_tensors', type=int, default=16,
                    help='first dimension of tensors (default: 16)')
parser.add_argument('--db_tensors_batch_size', type=int, default=1, metavar='N',
                    help='tensors that each ml rank retrieve from the db (default: 1)')
args = parser.parse_args()


### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Connect a SmartRedis client to the Redis database
client = Client(cluster=True)


# dataset
class PhastaRankDataset(torch.utils.data.Dataset):
    """ contains the keys of all tensors uploaded to db by phasta ranks
    """
    def __init__(self,
                 num_db_tensors):

        self.total_data = num_db_tensors

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        return f"data_{idx}"

class MinibDataset(torch.utils.data.Dataset):
    """ dataset of each ML rank in one epoch with the concatenated tensors
    """
    def __init__(self,
                 concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]


# Model 
nNeurons = 20
ndIn = 9
input_dim = ndIn
ndOut = 3

class NeuralNetwork(nn.Module): 
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim, outputDim, numNeurons):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons), 
            nn.LeakyReLU(0.3), 
            nn.Linear(self.nNeurons, self.ndOut),
        )

    def forward(self, x):
        return self.net(x)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def main():    
    # random seeds
    seed = args.seed
    torch.manual_seed(seed + hrank)
    np.random.seed(seed + hrank)
    random.seed(seed + hrank)

    args.cuda = args.device.find("gpu") != -1
    
    if args.device.find("gpu") != -1:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    if hvd.rank() == 0:
        logger.info(f"device: {torch_device}")
     
    if (args.num_threads != 0):
        torch.set_num_threads(args.num_threads)
    
    if hvd.rank() == 0:
        logger.info(args)
        logger.info("Torch Thread setup: ")
        logger.info(f" Number of threads: {torch.get_num_threads()}")

    if args.device.find("gpu") != -1:
        kwargs = {'num_workers': 1,
                  'pin_memory': True}
    else:
        kwargs = {}

    # Define dataset
    train_dataset = PhastaRankDataset(args.num_db_tensors * 2)

    # use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=hsize, rank=hrank, seed=seed, drop_last=False)
    train_tensor_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.db_tensors_batch_size, sampler=train_sampler, **kwargs)
    
    # Instantiate model
    model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons).double()

    if args.device.find("gpu") != -1:
        # Move model to GPU.
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=Sum)


    def train(epoch):
        if hvd.rank() == 0:
            logger.info(f"Starting epoch {epoch} ...")

        step = 0
        while step < 1:
            try:
                step = client.get_tensor("step")[0]
            except RedisReplyError:
                pass
        if hvd.rank() == 0:
            logger.info(f"Started training epoch {epoch} with step {step}")

        model.train()
        running_loss = 0.0
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
    
        loss_fn = nn.functional.mse_loss
        
        for tensor_idx, tensor_keys in enumerate(train_tensor_loader):

            # get all tensors in this batch from db and concatenate
            concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key)) for key in tensor_keys], dim=0)

            mbdata = MinibDataset(concat_tensor)
            train_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=args.batch_size)
            for batch_idx, dbdata in enumerate(train_loader):

                # split into data and target
                data = dbdata[:, :input_dim]
                target = dbdata[:, input_dim:]
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
              
                optimizer.zero_grad()
                output = model.forward(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
                if batch_idx % args.log_interval == 0:
                    logger.info('[{}] Train Epoch: {} [batch {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(hvd.rank(), 
                        epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))
    
        running_loss = running_loss / len(train_loader) / args.batch_size
        loss_avg = metric_average(running_loss, 'running_loss')
    
        if hvd.rank() == 0: 
            logger.info("Training set: Average loss: {:.4f}".format(loss_avg))
    
    
    if hvd.rank() == 0:
        logger.info("Training started... ") 
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te0 = time.time()
        train(epoch)
        te1 = time.time()
        if hvd.rank() == 0:
            logger.info(f"Epoch {epoch}: %s seconds" % (te1 - te0))
    t1 = time.time()
    
    if hvd.rank() == 0:
        print("Total training time: %s seconds" % (t1 - t0))
        logger.info("Total training time: %s seconds" % (t1 - t0))
    

if __name__ == "__main__":
    main()

