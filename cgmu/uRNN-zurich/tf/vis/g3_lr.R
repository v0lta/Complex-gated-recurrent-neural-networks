#!/usr/bin/env R
# compare influence of learning rate
library(ggplot2)

base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"
#base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

lr3<-read.table(paste0(base_dir, 'relu-fbias_uRNN_T100_n30.vali.txt'), header=T)
lr2<-read.table(paste0(base_dir, 'relu-fbias-lr1e-2_uRNN_T100_n30.vali.txt'), header=T)

batch_size <- 20
batch_skip <- 150

learning_rate <- rep('1e-3', nrow(lr3))
learning_rate <- c(learning_rate, rep('1e-2', nrow(lr2)))
num_updates <- seq(nrow(lr3))
num_updates <- c(num_updates, seq(nrow(lr2)))
num_updates <- batch_skip * num_updates

da<-rbind(lr3, lr2)

da<-data.frame(da, learning_rate, num_updates)
da$learning_rate <- factor(da$learning_rate)

ggplot(da, aes(x=num_updates, y=vali_cost, group=learning_rate, colour=learning_rate)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.2)) + geom_hline(yintercept=0.167, linetype='dotted') + xlab("number of batches (batch size: 20)") + ylab("validation set MSE") + ggtitle("learning rate? uRNN (adding, T=100)")
ggsave(paste0(base_dir, "g3_lr.png"), width=5, height=3)
