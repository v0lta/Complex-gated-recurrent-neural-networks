#block(name=cgRNN_exp2, threads=2, memory=50000, subtasks=1, hours=68, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ./networks/cgRNN.py