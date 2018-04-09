#!/usr/bin/env R
# compare initialisation of lambda
library(ggplot2)

base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"
#base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

lambda0<-read.table(paste0(base_dir, 'relu-fbias-l0_uRNN_T100_n30.vali.txt'), header=T)
lambda_not0<-read.table(paste0(base_dir, 'relu-fbias_uRNN_T100_n30.vali.txt'), header=T)

batch_size <- 20
batch_skip <- 150

lambda_init <- rep('init 0', nrow(lambda0))
lambda_init <- c(lambda_init, rep('init normal', nrow(lambda_not0)))
num_updates <- seq(nrow(lambda0))
num_updates <- c(num_updates, seq(nrow(lambda_not0)))
num_updates <- batch_skip * num_updates

da<-rbind(lambda0, lambda_not0)

da<-data.frame(da, lambda_init, num_updates)
da$lambda_init <- factor(da$lambda_init)

ggplot(da, aes(x=num_updates, y=vali_cost, group=lambda_init, colour=lambda_init)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.2)) + geom_hline(yintercept=0.167, linetype='dotted') + xlab("number of batches (batch size: 20)") + ylab("validation set MSE") + ggtitle("initialising lambda to 0? uRNN (adding, T=100)")
ggsave(paste0(base_dir, "g2_lambda0.png"), width=5, height=3)
