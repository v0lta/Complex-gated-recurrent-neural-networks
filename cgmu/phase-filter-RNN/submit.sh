#block(name=paper_sg_rerun, threads=2, memory=7500, subtasks=1, hours=120, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --adding True --memory False --time_steps 250 --stiefel False --real False --non_linearity hirose --model sGRU --GPU 0 --subfolder=testtest --n_units 91