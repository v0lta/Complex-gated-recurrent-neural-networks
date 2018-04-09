#!/usr/bin/env R
library(ggplot2)
# still using ggplot for graphs, oh well

args <- commandArgs(TRUE)
#args <-"adding/batch500_uRNN_T100_n20_lambdas.txt"
#args <-"adding/batch1000_ntrain1e7_v2_uRNN_T100_n20_lambdas.txt"
#args<-"adding/batch1000_uRNN_T100_n30_lambdas.txt"
#args<-"adding/batch250_lr5e-3_uRNN_T100_n30_lambdas.txt"
#args<-"adding/lambdas_uRNN_T100_n30.vali.txt"

base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"
lambda_path <- paste0(base_dir, args[1])
trace_path <- gsub("_lambdas", ".vali", lambda_path)

# --- load, prep --- #
da<-read.table(lambda_path, header=TRUE)
lambda_abs_means <- rowMeans(abs(da[, 2:ncol(da)]))
lambda_min <- apply(da[, 2:ncol(da)], 1, min)
lambda_max <- apply(da[, 2:ncol(da)], 1, max)

da_trace <- read.table(trace_path, header=TRUE)
da_trace<- da_trace[2:nrow(da_trace),]
vali_cost<-da_trace$vali_cost

N <- nrow(da)
what<-c(rep("mean |lamba|", N), rep("min lamba", N), rep("max lambda", N), rep("vali cost", nrow(da_trace)))
num_examples<-c(rep(seq(0, 100000*da_trace[nrow(da_trace),]$epoch, length=nrow(da)),3), seq(1, 100000*da_trace[nrow(da_trace),]$epoch, length=nrow(da_trace)))
value<-c(lambda_abs_means, lambda_min, lambda_max, vali_cost)
dat<-data.frame(num_examples, what, value)
dat$what <- factor(dat$what)

ggplot(dat, aes(x=num_examples, y=value, group=what, colour=what)) + geom_point(cex=0.3)  + facet_grid(what~., scales="free")
ggsave(gsub('.txt', '.lambdas.png', trace_path), width=4.5, height=3)
