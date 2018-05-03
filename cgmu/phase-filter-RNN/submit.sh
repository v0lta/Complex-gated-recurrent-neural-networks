#block(name=gate_init_1_e4, threads=2, memory=7500, subtasks=1, hours=12, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --model GUNN --memory True --adding False --non_linearity hirose --learning_rate=0.001 --batch_size=50