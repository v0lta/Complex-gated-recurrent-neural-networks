#!/usr/bin/env R
library(ggplot2)
# still using ggplot for graphs, oh well

base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

ALL<-FALSE
BATCHES<-FALSE

if (ALL){
    da<-read.table(paste0(base_dir, 'all_urNN.vali.txt'), header=TRUE)
} else if (BATCHES){ 
    da<-read.table(paste0(base_dir, 'batches_urNN.vali.txt'), header=TRUE) 
    da$batchsize <- factor(da$batchsize)
} else{
    da<-read.table(paste0(base_dir, 'n_urNN.vali.txt'), header=TRUE) 
    da$n <- factor(da$n)
}

num_batches<-da$updates*150
da<-data.frame(da, num_batches)
num_examples<-da$num_batches*20
da<-data.frame(da, num_examples)
multiplier<-rep(1, nrow(da))
multiplier[da$which=="batch1000_uRNN_T100_n30.vali.txt"] <- 50
multiplier[da$which=="batch50_uRNN_T100_n30.vali.txt"] <- 50/20
multiplier[da$which=="batch250_uRNN_T100_n10.vali.tx"] <- 250/20
multiplier[da$which=="batch250_uRNN_T100_n30.vali.txt"] <- 250/20
da$num_examples <- da$num_examples * multiplier

if (ALL){
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + ylim(0.146, 0.175)
    ggsave(paste0(base_dir,'uRNN_all.png'), width=8, height=3)
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + xlim(0, 2e6)
    ggsave(paste0(base_dir,'uRNN_all.zoom.png'), width=8, height=3)

    ggplot(subset(da, which%in%c("batch250_uRNN_T100_n30.vali.txt", "")), aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + xlim(0, 2e6)
    ggsave(paste0(base_dir,'uRNN_all.zoom.png'), width=8, height=3)
} else if (BATCHES){
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=batchsize)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_batches.png'), width=8, height=3)
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=batchsize)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + xlim(0, 2e6) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_batches.zoom.png'), width=8, height=3)
} else{
    # n #
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=n)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_n.png'), width=8, height=3)
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_n_which.png'), width=8, height=3)
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=n)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + xlim(0, 2e6) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_n.zoom.png'), width=8, height=3)
    ggplot(da, aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + xlim(0, 2e6) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_n_which.zoom.png'), width=8, height=3)
    ggplot(subset(da, n%in%c(20, 40)), aes(x=num_examples, y=vali_cost, group=which, colour=n)) + geom_point(cex=0.3) + ylim(0.146, 0.175) + geom_line(alpha=0.2)
    ggsave(paste0(base_dir,'uRNN_n_20v40.png'), width=8, height=3)
    ggplot(subset(da, n==20), aes(x=num_examples, y=vali_cost, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + ylim(0.145, 0.175)
    ggsave(paste0(base_dir,'uRNN_n20.png'), width=8, height=3)
}
