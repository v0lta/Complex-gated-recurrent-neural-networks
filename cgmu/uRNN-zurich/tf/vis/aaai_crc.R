library(ggplot2)

#switch <- 'MEM'
switch <- 'ADD'

xmax<-50000

if (switch == 'MEM'){
    base_dir <- "/Users/stephanie/PhD/git/uRNN/tf/output/memory/"
    costname<-'CE'
    ymax<-0.25
    xmax<-31000
    baseline<-0.1732
    title <- 'memory task, T = 100'
    irnn_name <- "aaai_crc_IRNN_T100_n30"
    tanh_name <- 'aaai_crc_tanhRNN_T100_n30'
    uRNN_name <- 'aaai_crc_tanh_uRNN_T100_n30'
    uRNN_tanh_beta_name <- 'beta1.05_tanh_3_uRNN_T100_n30'
    uRNN_tanh_name <- 'gradtestlong_tanh_uRNN_T100_n30'
    complex_RNN_name <- 'aaai_crc_complex_RNN_T100_n128'
    lstm_name <- 'aaai_crc_LSTM_T100_n30'
    fnames <- c(irnn_name, complex_RNN_name, lstm_name, uRNN_tanh_name, uRNN_tanh_beta_name)
    names <- c('IRNN', 'uRNN', 'LSTM', 'guRNN', 'guRNN+')
    colvals=c("darkgoldenrod2", "blue", "chartreuse3", "grey", "maroon3")
} else if (switch=='ADD') {
    base_dir <- "/Users/stephanie/PhD/git/uRNN/tf/output/adding/"
    costname<-'MSE'
    ymax<-0.2
    baseline<-0.167
    title <- 'adding task, T = 100'
    irnn_name <- 'aaai_crc_IRNN_T100_n30'
    lstm_name <- 'aaai_crc_LSTM_T100_n30'
    complex_RNN_name <- 'aaai_crc_complex_RNN_T100_n512'
    uRNN_name_relu <- 'aaai_crc_relu_uRNN_T100_n30'
    uRNN_name <- 'beta1.40_relu_4_uRNN_T100_n30'
    tanh_name <- 'aaai_crc_tanhRNN_T100_n30'
    uRNN_modrelu_name <- 'aaai_crc_uRNN_T100_n30'
    fnames <- c(irnn_name, lstm_name, complex_RNN_name, uRNN_name, uRNN_name_relu)
    names<-c("IRNN", "LSTM", "uRNN", "guRNN+", "guRNN")
    colvals=c("darkgoldenrod2", "chartreuse3", "blue", "maroon3", "grey")
} 

da_all <- data.frame()
for (i in seq(length(fnames))){
    base_name<-fnames[i]
    name<-names[i]
    fname<-paste0(base_dir, base_name)
    vali<-paste0(fname, ".vali.txt")
    da<-read.table(vali, header=T)
    updates<-150*seq(nrow(da))
    which<-rep(name, nrow(da))
    da_vali <- data.frame(da, updates, which)
    da_all <- rbind(da_all, da_vali)
}


p <- ggplot(da_all, aes(x=updates, y=vali_cost, group=which, colour=which)) + ggtitle(title) + theme_bw() + ylab(costname) + coord_cartesian(ylim=c(0, ymax), xlim=c(0, xmax))+ scale_colour_manual(values=colvals) + geom_hline(yintercept=baseline, linetype='dashed', alpha=0.8)
if (grepl('ADD', switch)){
    p <- p + geom_point(size=0.3, alpha=1.0) + geom_line(alpha=0.6, size=0.6)
} else {
    p <- p + geom_point(cex=0.5) + geom_line(alpha=0.5, size=1.0)
}
ggsave(paste0(base_dir, switch, "_aaai_crc.vali_trace.png"), width=4.5, height=3)
ggsave(paste0(base_dir, switch, "_aaai_crc.vali_trace.pdf"), width=4.5, height=3)
