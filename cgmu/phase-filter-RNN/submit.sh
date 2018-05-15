#block(name=paper_dg_parameq, threads=2, memory=7500, subtasks=1, hours=120, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --model CGRU --memory loop --adding loop --time_step -1 --non_linearity loop --subfolder=paper_D --GPU=0 --single_gate False --n_units 39