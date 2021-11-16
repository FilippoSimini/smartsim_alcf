import sys, time
import numpy as np
from smartredis import Client

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# parse arguments
dim_db_tensors = int(sys.argv[1])
epochs = int(sys.argv[2])

steps = 3
dim_in_out_variables = 12
np.random.seed(rank)

# Connect a SmartRedis client to the Redis database
client = Client(cluster=True)
client.put_tensor(f"step", np.array([0]))

time.sleep(30)
for step in range(1, steps + 1):
    # simulation running
    time.sleep(5)

    # each simulation rank produces two tensors
    data1 = np.random.random(size=(dim_db_tensors, dim_in_out_variables))
    data2 = np.random.random(size=(dim_db_tensors, dim_in_out_variables))

    # put data in db
    client.put_tensor(f"data_{rank}", data1)
    client.put_tensor(f"data_{size + rank}", data2)

    # update step count on db
    if rank == 0:
        client.put_tensor(f"step", np.array([step]))

