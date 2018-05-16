#block(name=paper_dg_rerun, threads=2, memory=7500, subtasks=1, hours=120, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --model CGRU --memory False --adding True --non_linearity hirose --subfolder extra --n_units 39 --time_steps 250 --GPU 0 --single_gate False