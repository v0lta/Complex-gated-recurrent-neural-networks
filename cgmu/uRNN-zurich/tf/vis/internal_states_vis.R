#!/usr/bin/env bash
# visualse the norm of the difference between the final vector...
library(ggplot2)

#args<-commandArgs(TRUE)
base_dir <- "/home/hyland/git/complex_RNN/tf/output/adding/"
#args<-"gradnorms_test_tanhRNN_T100_n3.hidden_states.txt"
#args<-"gradnorms_test_LSTM_T100_n3.hidden_states.txt"
args<-"gradnorms_test_complex_RNN_T100_n3.hidden_states.txt"

# --- IRNN --- #
IRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_LSTM_T100_n3.hidden_states.txt"), header=TRUE)

# --- LSTM --- #
LSTM_grads<-read.table(paste0(base_dir, "gradnorms_test_LSTM_T100_n3.hidden_states.txt"), header=TRUE)

# --- tanhRNN --- #
tanhRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_tanhRNN_T100_n3.hidden_states.txt"), header=TRUE)

# --- complex_RNN --- #
complex_RNN_grads<-read.table(paste0(base_dir, "gradnorms_test_complex_RNN_T100_n3.hidden_states.txt"), header=TRUE)

# --- uRNN --- #
uRNN_grads<-read.table(paste0(base_dir, "gradnorms_test_uRNN_T100_n3.hidden_states.txt"), header=TRUE)


ggplot(da, aes(x=k, y=value, colour=what, group=what)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("norm of difference between internal state h_k and final state") + xlab("k") + ylab("|h_k|")
ggsave(gsub(".txt", ".png", states_path), width=4.5, height=3)

