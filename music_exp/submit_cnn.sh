#block(name=mn_CNN, threads=2, memory=7500, subtasks=1, hours=12, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython ./networks/cnn.py
