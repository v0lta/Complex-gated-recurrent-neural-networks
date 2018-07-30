#block(name=aaai_rerun_motion, threads=2, memory=7500, subtasks=1, hours=168, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython src/translate.py -- --action walking --seq_length_out 35 --cgru False --size 1024 --learning_rate 0.005