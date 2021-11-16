#!/bin/bash

# change `ChargeAccount` with the charge account for your project
ChargeAccount=cfdml_aesp
queue=debug-cache-quad
runtime=30

# args:
dbnodes=1
simnodes=2
mlnodes=1
nodes=$(($dbnodes + $simnodes + $mlnodes))
ppn=64 # processes per node
simprocs=128
mlprocs=64
allprocs=$(($nodes * $ppn))

echo number of total nodes $nodes 
echo time in minutes $runtime
echo number of total processes $allprocs
echo ppn  N $ppn
echo queue $queue
 
qsub -q $queue -n $nodes -t $runtime -A $ChargeAccount run.sh $ppn $nodes $allprocs $dbnodes $simnodes $mlnodes $simprocs $mlprocs

