#!/usr/bin/env R

library(ggplot2)
library(tidyr)

GET_EPOCHS<-FALSE
#GET_EPOCHS<-TRUE

#base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/memory/"
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"

# mem
#base_name <- 'gradtestlong_tanh_uRNN_T100_n30'
#base_name <- 'longgrads_complex_RNN_T100_n128'
#base_name <- 'longgrads_LSTM_T100_n128'
#base_name <- 'longgrads_2_LSTM_T100_n40'

# add
#base_name <- 'gradtestlong_relu_uRNN_T100_n30'
base_name <- 'longgrads-relu-2_uRNN_T100_n30'
#base_name <- 'longgrads_complex_RNN_T100_n512'
#base_name <- 'longgrads_LSTM_T100_n40'
#base_name <- 'leakyrelu-gradtest_uRNN_T100_n30'


#base_name <- 'longgrads_IRNN_T100_n80'
#base_name <- 'longgrads_2_IRNN_T100_n80'

fname<-paste0(base_dir, base_name)
#fname<-paste0(base_dir, 'longgrads_complex_RNN_T100_n512')

# --- grads --- #
grads<-paste0(fname, ".hidden_gradients.txt")
da_grads<-read.table(grads, header=T)

# extract just one k value
#k_pick<-100
# let's put a STAT on this
#da_grads<-subset(da_grads, k==k_pick)

# because in this version, I forgot to include the epoch, we have to deduce it
batches_per_epoch = length(unique(da_grads$batch))
# get epochs
if (GET_EPOCHS){
    epoch<-c()
    epoch_counter<-0
    prev_batch<-0
    for (batch in da_grads$batch){
        if (batch < prev_batch){ epoch_counter <- epoch_counter + 1 } 
        prev_batch <- batch
        epoch<-c(epoch, epoch_counter)
    }
    updates<-(epoch*5000) + da_grads$batch
    da_grads<-data.frame(updates, epoch, da_grads)
} else{
    updates<-(da_grads$epoch*5000) + da_grads$batch
    da_grads<-data.frame(updates, da_grads)
}

# --- now get the vali --- #
vali<-paste0(fname, ".vali.txt")
da_vali<-read.table(vali, header=T)
updates<-da_vali$batch + (5000*da_vali$epoch)
da_vali<-data.frame(updates, da_vali)
# delete the first row
da_vali <- da_vali[2:nrow(da_vali), ]

# combine them further
which<-c(rep("grad", nrow(da_grads)), rep("vali cost", nrow(da_vali)))
da_1<-data.frame(da_grads$updates, da_grads$norm)
names(da_1)<-c("update", "val")
# normalise (arbitrary)
#da_1$val<-da_1$val/da_1$val[0.25*nrow(da_1)]
# first, average over k
#dd <- aggregate(da_grads, by=list(da_grads$updates), FUN=mean)
#da_1$val<-(da_1$val-min(da_1$val))/(max(da_1$val) - min(da_1$val))

da_2<-data.frame(da_vali$updates, da_vali$vali_cost)
names(da_2)<-c("update", "val")
# normalise
#da_2$val<-da_2$val/da_2$val[0.1*nrow(da_2)]
#da_2$val<-(da_2$val-min(da_2$val))/(max(da_2$val) - min(da_2$val))
#da_2$val <- (da_2$val - min(dd$norm))/(max(da_2$val) - min(da_2$val))
#da_2$val<-(da_2$val-min(da_2$val))/(max(da_2$val) - min(da_2$val))

da<-rbind(da_1, da_2)
da<-data.frame(da, which)

# --- now for plot --- #

# this is the averaging one
#ggplot(da, aes(x=update, y=val, group=which, colour=which)) + stat_summary(geom="smooth", fun.data="mean_cl_boot") + xlab("number of training updates") + theme_bw()  + facet_grid(which~., scale="free") + ggtitle(base_name)
#+ coord_cartesian(ylim=c(0, 1))
#ggsave(paste0(fname, ".longgrads.png"), width=4.5, height=3)

# now for the difference one
# want to get difference between k = 0 and k = max
# kill cols 2 and 3
d_s <- da_grads[, c(1, 4, 5)]
d_s <- spread(d_s, k, norm)
names(d_s) <- lapply(names(d_s), function(x) paste0("col", x))
diff <- d_s$col99 - d_s$col1
daa <- data.frame(d_s$colupdates, diff)
names(daa) <-c("update", "val")
daa_2<-rbind(daa, da_2)
which<-c(rep("grad", nrow(daa)), rep("vali", nrow(da_2)))
daa_2<-data.frame(daa_2, which)

#ggplot(daa_2, aes(x=update, y=val, group=which, colour=which)) + geom_point(cex=0.7) + geom_line(alpha=0.2) + xlab("number of training updates") + theme_bw()  + ggtitle(base_name) + facet_grid(which~., scales="free") + ylab("grad: diff btw 1 & 119 state grad")
#+ coord_cartesian(ylim=c(-0.5, 0.5))
#ggsave(paste0(fname, ".diff.longgrads.png"), width=4.5, height=3)


# --- instead of difference, let's get the slope of the log plot --- #
# just need updates, k, and norm
da_slope <- da_grads[, c(1, 4, 5)]
# uh < - lm(log(norm) ~ k, data=dd)
# summary(uh)$coefficients[1,1]
# summary(uh)$adj.r.squared
fits <- by(da_slope, da_slope$updates, function(x) lm(log(norm) ~ k, data=x))
slopes <- sapply(fits, function(x) summary(x)$coefficients[1, 1])
r_sq <- sapply(fits, function(x) summary(x)$adj.r.squared)

updates<-c(rep(as.numeric(names(fits)), 2), da_vali$updates, da_grads$updates)
what<-c(rep("slope", length(fits)), rep("rsq", length(fits)), rep("vali", nrow(da_vali)), rep("grad norm", nrow(da_grads)))
val<-c(slopes, r_sq, da_vali$vali_cost, da_grads$norm)
plot_da <- data.frame(updates, what, val)

# prune absurd values
#IRNN
# on vali
plot_da <- subset(plot_da, (what!="vali")|((what=="vali")&(val<0.3)))


ggplot(plot_da, aes(x=updates, y=val, group=what, colour=what)) + geom_point(cex=0.5) + geom_line(alpha=0.8) + facet_grid(what~., scales="free") + ggtitle(base_name) + theme_bw()
ggsave(paste0(fname, ".all.longgrads.png"), width=5, height=5)
