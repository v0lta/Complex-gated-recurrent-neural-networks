#block(name=vgg_class_fpb1, threads=10, memory=10000, subtasks=1, hours=48, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py
