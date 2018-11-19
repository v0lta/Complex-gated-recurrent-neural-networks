Code for the paper on complex gated recurrent neural networks (https://arxiv.org/pdf/1806.08267v2.pdf).
This project was developed using python 3.6 and Tensorflow 1.10.0

To recreate the results in table 1 run `bonn_eval_gate_diff.py`, once for the adding and one
more time for the memory problem. Adjust `./eval/eval.py` with the proper log-directories and it will
do the evaluation for you.

In order to re-run the human-motion prediction and music transcription experiments in the 
paper take a look at the human_motion_exp and music_exp directories.

Use the 'montreal_eval.py' file to recreate our experiments on the memory and adding problem shown
in figures 2 and 3 of the paper.

This repository contains tensorflow ports of the Theano code at:
https://github.com/amarshah/complex_RNN
and https://github.com/stwisdom/urnn

The custom optimizers class contains the Stiefel-Manifold optimizer proposed in "Full-Capacity Unitary Recurrent Neural Networks"
by Wisdom et al. (https://arxiv.org/abs/1611.00035) this is the default.
In order to work with the basis proposed by Arjovski and Shah et al in "Unitary Evolution Recurrent Neural Networks" (https://arxiv.org/abs/1511.06464) you can set `arjovski_basis=True`, for the complex cells implemented in `custom_cells.py`. 

You don't have to work in the complex domain. To create real valued cells simply
set the `real` argument in the constructor to `True` and choose a real valued
activation such as the relu. The Stiefel manifold optimizer will also work in the 
real domain.

If you find the code in this repository useful please consider citing:
```
@inproceedings{wolter-2018-nips,
     author = {Wolter, Moritz and Yao, Angela},
      title = {Complex Gated Recurrent Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems 31},
       year = {2018},
   abstract = {Complex numbers have long been favoured for digital signal processing, yet
               complex representations rarely appear in deep learning architectures. RNNs, widely
               used to process time series and sequence information, could greatly benefit from
               complex representations. We present a novel complex gated recurrent cell, which
               is a hybrid cell combining complex-valued and norm-preserving state transitions
               with a gating mechanism. The resulting RNN exhibits excellent stability and
               convergence properties and performs competitively on the synthetic memory and
               adding task, as well as on the real-world tasks of human motion prediction.}
}
```


