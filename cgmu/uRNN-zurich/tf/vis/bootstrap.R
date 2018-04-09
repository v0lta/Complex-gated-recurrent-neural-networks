library(ggplot2)
library(boot)

args<-commandArgs(TRUE)

d<-args[1]
noise<-0.01

#fname_base<-paste0('../output/simple/l2/d', d, '_noise', noise, '_bn20_nb50000_')
#fname_base<-paste0('../output/simple/projection_test_d', d, '_noise', noise, '_bn20_nb50000_')
fname_base<-paste0('../output/simple/aaai_d', d, '_noise', noise, '_bn20_nb50000_')
print(fname_base)

# --- print summary statistics about test --- #
#fname<-paste0(fname_base, 'comb.txt')
fname<-paste0(fname_base, 'test.txt')

dtest<-read.table(fname, header=T)

my_mean<-function(data, indices){
    return(mean(data[indices, ]$loss))
}

means<-aggregate(dtest$loss, by=list(dtest$experiment), FUN=mean)

# testing bootstrap approach
for (exper in levels(factor(dtest$experiment))){
    dd <-subset(dtest, experiment==exper)
    # get bootstrap means
    boot_results <- boot(data=dd, statistic=my_mean, R=1000)
    boot_results.ci <- boot.ci(boot_results, type="basic")
    lower<-boot_results.ci$basic[4]
    upper<-boot_results.ci$basic[5]
    width<-(upper-lower)/2
    cat(mean(boot_results$t), "\t", sd(boot_results$t), "\t", width, "\t", exper, "\n")
}
    
#results <- boot(data=dtest, statistic=my_means, R=3)

#names(means)<-c("experiment", "mean")
#sems<-aggregate(dtest$loss, by=list(dtest$experiment), FUN=function(x) sd(x)/sqrt(length(x)))
#names(sems)<-c("experiment", "standard error")

#test_results<-merge(means, sems)
#print(test_results)
