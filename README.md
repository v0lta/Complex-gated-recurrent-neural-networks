Code for our paper on complex gated recurrent neural networks (https://arxiv.org/abs/1806.08267).

To recreate the results in table 1 run bonn_eval_gate_diff.py, once for the adding and one
more time for the memory problem. Adjust ./eval/eval.py with the proper log-directories and it will
do the evaluation for you.

Use the montreal_eval.py file to recreate our experiments on the memory and adding problem.


For the human motion prediction experiemnts run: 

'''
cd human-motion-prediction
python src/translate.py --action walking --seq_length_out 25 --iterations 20000
'''

