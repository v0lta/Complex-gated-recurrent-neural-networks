#block(name=cvpr_mods, threads=2, memory=7500, subtasks=1, hours=168, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    # ipython src/translate.py -- --action all --seq_length_in 61 --seq_length_out 60 --GPU 0 --batch_size 16 --cgru True --fft True --window_size 30 --stiefel False --step_size 25
    ipython src/translate.py -- --learning_rate 0.005 --omit_one_hot False --residual_velocities True --cgru True --stiefel False --size 1536