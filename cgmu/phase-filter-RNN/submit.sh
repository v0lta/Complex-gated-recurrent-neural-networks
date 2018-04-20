#block(name=cgmu_GUNN, threads=2, memory=7500, subtasks=1, hours=48, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --model GUNN --memory False --adding True --GPU 0 --learning_rate=0.001 --batch_size=250 --subfolder=gate_init_1_e3
