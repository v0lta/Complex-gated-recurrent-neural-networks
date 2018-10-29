Modified version of the code at https://github.com/una-dinosauria/human-motion-prediction.git
for instructions on how to get started please take a look at the original readme.

The recreate the experiments in our paper please run:
`ipython src/translate.py -- --learning_rate 0.005 --omit_one_hot False --residual_velocities True --cgru False --size 1024`
and 
`ipython src/translate.py -- --learning_rate 0.005 --omit_one_hot False --residual_velocities True --cgru True --size 1024`
