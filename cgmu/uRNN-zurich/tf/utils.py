#!/usr/bin/env ipython
#
# Utility functions.
#
# ------------------------------------------
# author:       Stephanie Hyland (@corcra)
# date:         20/3/16
# ------------------------------------------

import tensorflow as tf

class ExperimentOptions(dict):
    """
    For storing hyperparameters and other training options.
    """
    def __init__(self, cfg_path=None):
        # TODO: choose better defaults
        # model-intrinsic things
        self.model = 'trivial'
        self.experiment = 'memory'
        self.T = 100                # 'time' length for memory/seq length
        self.n_input = 10
        self.n_hidden = 10
        self.n_output = 10
        # training-specific
        self.n_iter = 20000
        self.batch_size = 20
        # RMSProp
        self.learning_rate = 0.001
        self.decay = 0.5
        self.momentum = 0.9
        self.epsilon = 1e-3
        # gradient clipping
        self.clipping = False
        if not cfg_path is None:
            self.load(cfg_path)
        self.experiment_presets()
    def experiment_presets(self):
        if self.experiment == 'adding':
            self.loss_type = 'MSE'
            self.out_every_t = True
            self.n_input = 2
            self.n_ouptut = 1
        elif self.experiment == 'memory':
            self.loss_type = 'CE'
            self.out_every_t = False
            self.n_input = 10
            self.n_output = 9
        else:
            raise NotImplementedError
    def load(self, cfg_path):
        print 'Loaded options from', cfg_path
        return False
    def save(self, cfg_path):
        print 'Saved options to', cfg_path
        return False
