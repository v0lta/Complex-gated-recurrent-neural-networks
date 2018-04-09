#!/usr/bin/env R
# compare complex and uRNN with the same hidden state size
library(ggplot2)

base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"
#base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

uRNN<-read.table(paste0(base_dir, 'relu_uRNN_T100_n30.vali.txt'), header=T)
complex_RNN<-read.table(paste0(base_dir, 'complex_RNN_T100_n30.vali.txt'), header=T)

batch_size <- 20
batch_skip <- 150

model <- rep('uRNN', nrow(uRNN))
model <- c(model, rep('complex_RNN', nrow(complex_RNN)))
num_updates <- seq(nrow(uRNN))
num_updates <- c(num_updates, seq(nrow(complex_RNN)))
num_updates <- batch_skip * num_updates

da<-rbind(uRNN, complex_RNN)

da<-data.frame(da, model, num_updates)
da$model <- factor(da$model)

ggplot(da, aes(x=num_updates, y=vali_cost, group=model, colour=model)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.2)) + geom_hline(yintercept=0.167, linetype='dotted') + xlab("number of batches (batch size: 20)") + ylab("validation set MSE") + ggtitle("uRNN v. complexRNN, n=30 (adding, T=100)")
ggsave(paste0(base_dir, "g6_samestatesize.png"), width=5, height=3)
