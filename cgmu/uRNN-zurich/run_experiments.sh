#!/usr/bin/env bash
#   *------------------------------------------------*
#   |   Attempting to reproduce the results froma    | 
#   |   Unitary Evolution Recurrent Neural Networks  |
#   |   Martin Arjovsky, Amar Shah, Yoshua Bengio    |
#   |   http://arxiv.org/abs/1511.06464              |
#   *------------------------------------------------*

# === shared params (assuming defaults unless specified)
n_iter=20000
n_batch=20      # note they say they did 20 and 50 for adding problem...
learning_rate=0.001

# === the functions (bitta boilerplate)
run_memory () {
    time_steps_set=( 100 200 300 500 )
    input_type="categorical"
    out_every_t="True"
    loss_function="CE"
    n_hidden=$1
    model=$2
    for T in "${time_steps_set[@]}"
    do
        savefile="memory_"$model\_$T
        kwargs="$n_iter $n_batch $n_hidden $T $learning_rate $savefile $model $input_type $out_every_t $loss_function"
        ipython memory_problem.py $kwargs
        mv -v $savefile output/
    done
}

run_adding () {
    time_steps_set=( 100 200 400 750 )
    input_type="real"
    out_every_t="False"
    loss_function="MSE"
    n_hidden=$1
    model=$2
    for T in "${time_steps_set[@]}"
    do
        savefile="adding_"$model\_$T
        kwargs="$n_iter $n_batch $n_hidden $T $learning_rate $savefile $model $input_type $out_every_t $loss_function"
        ipython adding_problem.py $kwargs
        mv -v $savefile output/
    done
}

model=$1
# hardcoded the n_hidden sizes (from paper)
case $model in
    "RNN")
        echo "RNN"
        run_memory 80 "RNN"
        run_adding 128 "RNN"
        ;;
    "IRNN")
        echo "IRNN"
        run_memory 80 "IRNN"
        run_adding 128 "IRNN"
        ;;
    "LSTM")
        echo "LSTM"
        run_memory 40 "LSTM"
        run_adding 128 "LSTM"
        ;;
    "complex_RNN")
        echo "complex_RNN"
        run_memory 128 "complex_RNN"
        run_adding 512 "complex_RNN"
        ;;
    *)
        echo "Unknown model input:" $model
        ;;
esac
