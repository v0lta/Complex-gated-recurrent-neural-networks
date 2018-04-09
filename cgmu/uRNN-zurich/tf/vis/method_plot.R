#!/usr/bin/env R
library(ggplot2)
library(scales)

#for (n_pick in c(3, 6, 8, 14, 20, 100)){
for (n_pick in c(100)){
    dbox<-read.table('../output/simple/l2/derived/boxplot_compile.txt', header=T)

    if (n_pick == 100){
        # doing some weird subsetting
        rand_dd <- subset(dbox, experiment=="random_unitary")
        means<-aggregate(rand_dd$loss, by=list(rand_dd$n), FUN=mean)
        rescale<-means$x
        names(rescale)<-means$Group.1
        print(rescale)
    }

    #if (n_pick %in% c(3, 6, 8, 100)){
    if (n_pick %in% c(3, 6, 8)){
        dd<-subset(dbox, experiment %in% c('complex_RNN', 'general_unitary'))
        dd$experiment<-factor(dd$experiment, labels=c('arjovsky', 'u(n)'))
    } else{
        dd<-subset(dbox, experiment %in% c('general_unitary', 'general_unitary_restricted'))
        dd$experiment<-factor(dd$experiment, labels=c('u(n)', 'u(n) (restricted)'))
    }

    dd$experiment<-factor(dd$experiment)
    print(levels(dd$experiment))
    dd$method<-factor(dd$method, labels=c('composition', 'Lie algebra', 'QR'))

    if (n_pick < 100){
        p<-ggplot(subset(dd, n==n_pick), aes(experiment, loss)) + theme_bw() + ylab("test set loss") + ggtitle(paste0('n = ', n_pick)) + xlab("approach")
    } else{
        # we need to do a lot of weird scaling now and i'm not so good at that
        for (n in levels(factor(dd$n))){
            print(rescale[[n]])
            dd[dd$n == n, ]$loss <- dd[dd$n == n, ]$loss/rescale[[n]]
            # dangerzone
			dbox[(dbox$experiment=="true")&(dbox$n==n), ]$loss <- dbox[(dbox$experiment=="true")&(dbox$n==n), ]$loss/rescale[[n]]
        }
        p<-ggplot(dd, aes(experiment, loss)) + theme_bw() + ylab("test loss (fraction of random loss)") + xlab("approach") + geom_hline(yintercept=mean(subset(dbox, experiment=="true")$loss), linetype='dashed')
    }
    dodge <- position_dodge(width=0.9)
    p <- p + stat_summary(fun.data = "mean_cl_boot", geom="bar", aes(colour=method, fill=method), position=dodge, alpha=0.2)
    p <- p + stat_summary(fun.data = "mean_cl_boot", geom="errorbar", aes(colour=method, fill=method), position=dodge, alpha=0.7, width=0.5)

    ggsave(paste0('../output/simple/l2/derived/boxplot', n_pick, '.png'), width=4.5, height=3)
    ggsave(paste0('../output/simple/l2/derived/boxplot', n_pick, '.pdf'), width=4.5, height=3)
}
