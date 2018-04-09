#!/usr/bin/env R
# compare no-bias uRNN at n = 20, 30, 40
library(ggplot2)

base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/memory/"
#base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding/"

twenty<-read.table(paste0(base_dir, 'tanh-withbias_uRNN_T100_n20.vali.txt'), header=T)
thirty<-read.table(paste0(base_dir, 'withbias_tanh_uRNN_T100_n30.vali.txt'), header=T)
forty<-read.table(paste0(base_dir, 'tanhbias_uRNN_T100_n40.vali.txt'), header=T)
sixty<-read.table(paste0(base_dir, 'tanh_uRNN_T100_n60.vali.txt'), header=T)
onetwentyeight<-read.table(paste0(base_dir, 'uRNN_T100_n128.vali.txt'), header=T)
arjovsky<-read.table(paste0(base_dir, 'T100/v2_complex_RNN_T100_n128.vali.txt'), header=T)

batch_size <- 20
batch_skip <- 150

state_size <- rep(20, nrow(twenty))
state_size <- c(state_size, rep(30, nrow(thirty)))
state_size <- c(state_size, rep(40, nrow(forty)))
state_size <- c(state_size, rep(60, nrow(sixty)))
state_size <- c(state_size, rep(128, nrow(onetwentyeight)))
state_size <- c(state_size, rep('arjovsky', nrow(arjovsky)))
num_updates <- seq(nrow(twenty))
num_updates <- c(num_updates, seq(nrow(thirty)))
num_updates <- c(num_updates, seq(nrow(forty)))
num_updates <- c(num_updates, seq(nrow(sixty)))
num_updates <- c(num_updates, seq(nrow(onetwentyeight)))
num_updates <- c(num_updates, seq(nrow(arjovsky)))
num_updates <- batch_skip * num_updates

da<-rbind(twenty, thirty)
da<-rbind(da, forty)
da<-rbind(da, sixty)
da<-rbind(da, onetwentyeight)
da<-rbind(da, arjovsky)

da<-data.frame(da, state_size, num_updates)
da$state_size <- factor(da$state_size, levels=c(20, 30, 40, 60, 128, 'arjovsky'))

ggplot(da, aes(x=num_updates, y=vali_cost, group=state_size, colour=state_size)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.2)) + geom_hline(yintercept=0.1732, linetype='dotted') + xlab("number of batches (batch size: 20)") + ylab("validation set CE") + ggtitle("state size analysis: uRNN (memory, T=100)")
ggsave(paste0(base_dir, "g4_statesize.png"), width=5, height=3)
