#block(name=cgRNN_CNN_4_8_16_d, threads=2, memory=7500, subtasks=1, hours=80, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ./networks/cgRNN.py
