#!/usr/bin/env R

library(ggplot2)
#base_dir <- "/home/hyland/git/complex_RNN/tf/output/adding/"
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"

da<-data.frame()

# --- IRNN --- #
IRNN_grads<-read.table(paste0(base_dir, "gradtest_IRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("IRNN (relu)", nrow(IRNN_grads))
da<-rbind(data.frame(IRNN_grads, which))

IRNN_grads<-read.table(paste0(base_dir, "gradtest-tanh_IRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("IRNN (tanh)", nrow(IRNN_grads))
#da<-rbind(da, data.frame(IRNN_grads, which))

# --- LSTM --- #
LSTM_grads<-read.table(paste0(base_dir, "gradtest_LSTM_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("LSTM", nrow(LSTM_grads))
#da<-rbind(da, data.frame(LSTM_grads, which))

# --- tanhRNN --- #
tanhRNN_grads<-read.table(paste0(base_dir, "gradtest_tanhRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("tanhRNN", nrow(tanhRNN_grads))
#da<-rbind(da, data.frame(tanhRNN_grads, which))

# --- complex_RNN --- #
complex_RNN_grads<-read.table(paste0(base_dir, "gradtest_complex_RNN_T100_n128.hidden_gradients.txt"), header=TRUE)
which<-rep("complex_RNN", nrow(complex_RNN_grads))
#da<-rbind(da, data.frame(complex_RNN_grads, which))

# --- uRNN --- #
uRNN_grads<-read.table(paste0(base_dir, "relu-gradtest_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (relu)", nrow(uRNN_grads))
da<-rbind(da, data.frame(uRNN_grads, which))

uRNN_grads<-read.table(paste0(base_dir, "gradtest-relu-l0_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (relu, l0)", nrow(uRNN_grads))
da<-rbind(da, data.frame(uRNN_grads, which))


uRNN_grads<-read.table(paste0(base_dir, "relumod-gradtest_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (relumod)", nrow(uRNN_grads))
da<-rbind(da, data.frame(uRNN_grads, which))

uRNN_grads<-read.table(paste0(base_dir, "tanh-gradtest_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (tanh)", nrow(uRNN_grads))
#da<-rbind(da, data.frame(uRNN_grads, which))

uRNN_grads<-read.table(paste0(base_dir, "tanhmod-gradtest_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (tanhmod)", nrow(uRNN_grads))
#da<-rbind(da, data.frame(uRNN_grads, which))

uRNN_grads<-read.table(paste0(base_dir, "leakyrelu-gradtest_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
which<-rep("uRNN (leakyrelu)", nrow(uRNN_grads))
da<-rbind(da, data.frame(uRNN_grads, which))

# --- now for plot --- #
ggplot(da, aes(x=k, y=norm, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("cost gradient wrt hidden state h_k") + xlab("k") + ylab("|dC/dh_k|") + scale_y_log10()

ggsave(paste0(base_dir, "internal_gradients_all.png"), width=4.5, height=3)
