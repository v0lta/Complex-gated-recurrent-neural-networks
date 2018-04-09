library(ggplot2)
library(boot)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
args<-commandArgs(TRUE)
PLOT_TRAIN<-F

d<-args[1]
#d<-20
#identifier<-args[2]
noise<-0.01

#fname_base<-paste0('../output/simple/l2/d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/hazan_3_d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/projection_test_d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/aaai/aaai_d', d, '_noise', noise, '_bn20_nb50000_')
fname_base<-paste0('../output/simple/restrict_fix_d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/d', d, '_noise', noise, '_bn20_nb5000_')
#fname_base<-paste0('../output/simple/d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/lr_d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/nips/d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/nips/random_projections_d', d, '_noise', noise, '_bn20_nb50000_')
print(fname_base)

# --- vali --- #
fname<-paste0(fname_base, 'vali.txt')
data<-read.table(fname, header=T)
print(levels(factor(data$rep)))
data['rep']<-NULL
data['method']<-NULL
data['t']<-NULL
data$experiment<-factor(data$experiment)
print(levels(data$experiment))
#data$experiment<-factor(data$experiment, labels=c("lie (basis) 0.0001", "lie (basis) 0.0002", "lie (basis) 0.0005", "lie 0.0001", "lie 0.0002", "lie 0.0005", "lie 0.001"))
#data$experiment<-factor(data$experiment, labels=c("composition", "lie algebra", "lie algebra (basis)", "lie algebra (restricted)", "projection"))
#data$experiment<-factor(data$experiment, labels=c("composition", "lie algebra", "lie algebra (basis)", "projection"))

### testing for now
data <- subset(data, experiment %in% c("complex_RNN", "general_unitary", "general_unitary_restricted"))
ggplot(data, aes(x=training_examples, y=loss, colour=experiment, group=experiment, fill=experiment)) +  ggtitle(paste0("validation set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + theme(legend.position="right") + coord_cartesian(ylim=c(0, 5), xlim=c(0, 1000000))
#+ ylim(0, 25)
#+ ylim(0, 30) + xlim(0, 2e06)
#+scale_colour_manual(values=gg_color_hue(5)[c(1, 2, 5)]) + scale_fill_manual(values=gg_color_hue(5)[c(1,2 , 5)])
#$+ ylim(0, 5) + xlim(0, 1e06) + 
ggsave(gsub(".txt", ".2.png", fname), width=6, height=4)
# --- train --- # (copy pasta)
if (PLOT_TRAIN){
    fname<-paste0(fname_base, 'train.txt')
    data<-read.table(fname, header=T)
    data['rep']<-NULL

    ggplot(data, aes(x=training_examples, y=loss, colour=experiment, group=experiment, fill=experiment)) +  ggtitle(paste0("training set loss (d=", d, ")")) + xlab("# training examples seen") + theme_bw() + stat_summary(fun.data = "mean_se", geom = "smooth")  + theme(legend.position="bottom")
    ggsave(gsub(".txt", ".2.png", fname))
    ggsave(gsub(".txt", ".2.pdf", fname))
}

# --- print summary statistics about test --- #
fname<-paste0(fname_base, 'test.txt')
#fname<-paste0(fname_base, 'comb.txt')
dtest<-read.table(fname, header=T)

#if ('where' %in% names(dtest)){
#    data['where']<-NULL
#}
my_means <- function(data, indices){
    exps<-levels(factor(data$experiment))
    print(exps)
    exper<-exps[indices]
    dd <-subset(data, experiment==exper)$loss
#    print(dd[10,])
    return(mean(dd))
}

means<-aggregate(dtest$loss, by=list(dtest$experiment), FUN=mean)
print(means)

# testing bootstrap approach
results <- boot(data=dtest, statistic=my_means, R=5)
print(results)
