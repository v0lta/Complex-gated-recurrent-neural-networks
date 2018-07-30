#block(name=prod_add, threads=2, memory=7500, subtasks=1, hours=336, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --adding loop --memory loop --time_steps -1 --stiefel True --real False --non_linearity loop --gate_non_linearity mod_sigmoid_prod --model sGRU --GPU 0 --subfolder=rebuttal_syn --n_units 128
    # ipython montreal_eval.py -- --adding loop --memory loop --time_steps -1 --stiefel True --real False --non_linearity loop --gate_non_linearity mod_sigmoid_sum --model sGRU --GPU 0 --subfolder=aaai_non_lin_stud --n_units 80
    # ipython bonn_eval.py