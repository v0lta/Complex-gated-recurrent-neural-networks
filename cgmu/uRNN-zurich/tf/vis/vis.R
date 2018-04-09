#!/usr/bin/env R
library(ggplot2)
# still using ggplot for graphs, oh well

args <- commandArgs(TRUE)

base_dir<-"/home/hyland/git/complex_RNN/tf/output/"
trace_path <- paste0(base_dir, args[1])

# --- load, prep --- #
da<-read.table(trace_path, header=TRUE)
# convert from epoch, batch to straight up batches
batch_increment <- da$batch[2] - da$batch[1]
batch_cum <- seq(nrow(da))*batch_increment
da<-data.frame(da, batch_cum)

ggplot(da, aes(x=batch_cum, y=vali_cost)) + geom_point(cex=0.3) + ylim(0.15, 0.21)
ggsave(gsub('.txt', '.png', trace_path), width=4.5, height=3)
