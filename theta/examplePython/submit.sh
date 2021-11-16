#!/bin/bash

# change `ChargeAccount` with the charge account for your project
ChargeAccount=youraccount
queue=debug-cache-quad 
runtime=00:30:00

# args:
dbnodes=4
batch_size=4
db_tensors_batch_size=2

ppn=64
nodes=8
allprocs=$(($nodes * $ppn))
simnodes=2
mlnodes=$(($nodes - $dbnodes - $simnodes))

echo number of nodes $nodes 
echo time in minutes $runtime
echo number of processes $allprocs
echo ppn  N $ppn
echo queue $queue
 
qsub -q $queue -n $nodes -t $runtime -A $ChargeAccount run.sh $ppn $nodes $allprocs $dbnodes $simnodes $mlnodes $batch_size $db_tensors_batch_size

