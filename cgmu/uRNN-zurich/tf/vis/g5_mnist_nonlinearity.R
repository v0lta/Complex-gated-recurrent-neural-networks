#!/usr/bin/env R
# compare uRNN on mnist with different nonlinearities
library(ggplot2)

base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/mnist/"
#base_dir<-"/home/hyland/git/complex_RNN/tf/output/mnist/"

relu<-read.table(paste0(base_dir, 'relu_uRNN_T100_n40.vali_acc.txt'), header=T)
relumod<-read.table(paste0(base_dir, 'relumod_uRNN_T100_n40.vali_acc.txt'), header=T)
tanh<-read.table(paste0(base_dir, 'tanh_uRNN_T100_n40.vali_acc.txt'), header=T)

batch_size <- 20
batch_skip <- 150

nonlinearity <- rep('relu', nrow(relu))
nonlinearity <- c(nonlinearity, rep('relumod', nrow(relumod)))
nonlinearity <- c(nonlinearity, rep('tanh', nrow(tanh)))
num_updates <- seq(nrow(relu))
num_updates <- c(num_updates, seq(nrow(relumod)))
num_updates <- c(num_updates, seq(nrow(tanh)))
num_updates <- batch_skip * num_updates

da<-rbind(relu, relumod)
da<-rbind(da, tanh)

da<-data.frame(da, nonlinearity, num_updates)
da$nonlinearity <- factor(da$nonlinearity)

ggplot(da, aes(x=num_updates, y=vali_acc_cost, group=nonlinearity, colour=nonlinearity)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 45)) + geom_hline(yintercept=10, linetype='dotted') + xlab("number of batches (batch size: 20)") + ylab("validation set accuracy (%)") + ggtitle("effect of nonlinearity on uRNN (task: mnist)")
ggsave(paste0(base_dir, "g5_mnist_nonlinearity.png"), width=5, height=3)
