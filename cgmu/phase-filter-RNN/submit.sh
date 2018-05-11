#block(name=CGRU_paper_Richards, threads=2, memory=7500, subtasks=1, hours=120, gpus=1)
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    ipython montreal_eval.py -- --model CGRU --non_linearity loop --learning_rate=0.001 --batch_size=50 --time_steps=-1 --GPU=0 --memory loop --adding=loop --subfolder=CGRU_paper -n_units=64