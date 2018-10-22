#block(name=mn_cgRNN, threads=2, memory=7500, subtasks=1, hours=96, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ./networks/cgRNN.py