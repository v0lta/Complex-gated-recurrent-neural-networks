#!/usr/bin/env R
# visualisation for mnist task!
library(ggplot2)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/mnist"

# --- IRNN --- #
IRNN_trace<-read.table(paste0(base_dir, "/IRNN_T100_n100.vali_acc.txt"), header=TRUE)
batch_size<-16
batch_skip<-150
num_updates <- batch_skip * seq(nrow(IRNN_trace))
num_examples<- batch_size * num_updates
acc <- IRNN_trace$vali_acc_cost
which <- rep("IRNN", nrow(IRNN_trace))

da<-data.frame(num_updates, num_examples, acc, which)

# --- LSTM --- #
LSTM_trace<-read.table(paste0(base_dir, "/LSTM_T100_n128.extracted_trace.txt"), header=TRUE)
which<-rep("LSTM", nrow(LSTM_trace))
dtemp<-data.frame(LSTM_trace, which)

da<-rbind(da, dtemp)

# --- tanhRNN --- #
tanhRNN_trace<-read.table(paste0(base_dir, "/tanhRNN_T100_n100.vali_acc.txt"), header=TRUE)
batch_size<-20
batch_skip<-150
num_updates <- batch_skip * seq(nrow(tanhRNN_trace))
num_examples<- batch_size * num_updates
acc <- tanhRNN_trace$vali_acc_cost
which <- rep("tanhRNN", nrow(tanhRNN_trace))
dtemp<-data.frame(num_updates, num_examples, acc, which)

da<-rbind(da, dtemp)

# --- NOW FOR PLOT --- #
ggplot(da, aes(x=num_updates, y=acc, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2)
ggsave(paste0(base_dir, "/mnist_all.png"), width=4.5, height=3)
