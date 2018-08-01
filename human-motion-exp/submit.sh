#block(name=nips_final_o10, threads=2, memory=7500, subtasks=1, hours=168, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython src/translate.py -- --action walking --seq_length_in 51 --seq_length_out 10 --GPU 0 --fft False --cgru False --window_size 10 --batch_size=8