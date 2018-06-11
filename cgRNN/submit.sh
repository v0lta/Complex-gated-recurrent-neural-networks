#block(name=aaai_rerun, threads=2, memory=7500, subtasks=1, hours=168, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --adding loop --memory loop --time_steps -1 --stiefel True --real False --non_linearity loop --gate_non_linearity mod_sigmoid_sum --model sGRU --GPU 0 --subfolder=aaai_non_lin_stud --n_units 80