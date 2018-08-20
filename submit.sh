#block(name=gate_study_bk_add, threads=2, memory=7500, subtasks=1, hours=24, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython bonn_eval_gate_diff.py
    # ipython montreal_eval.py -- --adding loop --memory loop --time_steps -1 --stiefel True --real False --non_linearity hirose --gate_non_linearity mod_sigmoid_prod --model sGRU --GPU 0 --subfolder=rebuttal_syn_cpx_250 --n_units 250
    # ipython montreal_eval.py -- --adding loop --memory loop --time_steps -1 --stiefel True --real False --non_linearity loop --gate_non_linearity mod_sigmoid_sum --model sGRU --GPU 0 --subfolder=aaai_non_lin_stud --n_units 80
    