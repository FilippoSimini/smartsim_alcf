import os, sys, time
from smartsim import Experiment
from smartsim.settings import AprunSettings
from smartsim.database import CobaltOrchestrator

# Parse command line arguments
ppn = int(sys.argv[1])
nodes = int(sys.argv[2])
allprocs = int(sys.argv[3])
dbnodes = int(sys.argv[4])
simnodes = int(sys.argv[5])
mlnodes = int(sys.argv[6])
batch_size = int(sys.argv[7])
db_tensors_batch_size = int(sys.argv[8]) 

assert (dbnodes + simnodes + mlnodes <= nodes) and (mlnodes >= 0), "The nodes requested are not enough."

num_db_tensors = ppn * simnodes
dim_db_tensors = 16
epochs = 5
PORT = 6780


# Define function to parse node list
def parseNodeList():
    cobStr = os.environ['COBALT_PARTNAME']
    tmp = cobStr.split(',')
    nodelist = []
    for item in tmp:
        if (item.find('-') > 0):
            tmp2 = item.split('-')
            istart = int(tmp2[0])
            iend = int(tmp2[1])
            for i in range(istart,iend+1):
                nodelist.append(str(i))
        else:
            nodelist.append(item)
    nnodes = len(nodelist)
    return nodelist, nnodes

# Get nodes of this allocation (job) and split them between the tasks
nodelist, nnodes = parseNodeList()
print(f"\nRunning on {nnodes} total nodes on Theta")
print(nodelist, "\n")
dbNodes = ','.join(nodelist[0: dbnodes])
simNodes = ','.join(nodelist[dbnodes: dbnodes + simnodes])
mlNodes = ','.join(nodelist[dbnodes + simnodes: dbnodes + simnodes + mlnodes])
print(f"Database running on {dbnodes} nodes:")
print(dbNodes)
print(f"Simulatiom running on {simnodes} nodes:")
print(simNodes)
print(f"ML running on {mlnodes} nodes:")
print(mlNodes, "\n")

# Set up database and start it
exp = Experiment("train-example", launcher="cobalt")
db = CobaltOrchestrator(port=PORT, batch=False, db_nodes=dbnodes, run_args={"node-list": dbNodes})
print("Starting database ...")
exp.start(db)
print("Done\n")

# data producer
print("Loading data ...")
t = time.time()
aprun = AprunSettings("python", 
        exe_args=f"load_data.py {dim_db_tensors} {epochs}", 
        run_args={"node-list": simNodes})
aprun.set_tasks(ppn * simnodes)
aprun.set_tasks_per_node(ppn)
load_data = exp.create_model("load_data", aprun)
exp.start(load_data, summary=False, block=False)
print(f" Done. Loading time: {time.time() - t}")

# data consumer
print("Starting training ...")
t = time.time()
aprun = AprunSettings("python", 
        exe_args=f"train_model.py --batch_size {batch_size} --num_db_tensors {num_db_tensors} --dim_db_tensors {dim_db_tensors} --db_tensors_batch_size {db_tensors_batch_size} --epochs {epochs}", 
        run_args={"node-list": mlNodes})
aprun.set_tasks(ppn * mlnodes)
aprun.set_tasks_per_node(ppn)
ml_model = exp.create_model("train_model", aprun)
exp.start(ml_model, summary=False)
print(f" Done. Training time: {time.time() - t}")

# Stop database
exp.stop(db)
print("Done.")

