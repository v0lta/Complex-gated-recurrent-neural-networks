#!/usr/bin/env R
# look at the gradients for different models

library(ggplot2)

#base_dir <- "/home/hyland/git/complex_RNN/tf/output/adding/"
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/adding/"

LSTM<-read.table(paste0(base_dir, "gradtest_LSTM_T100_n30.hidden_gradients.txt"), header=TRUE)
complex_RNN<-read.table(paste0(base_dir, "gradtest_complex_RNN_T100_n30.hidden_gradients.txt"), header=TRUE)
IRNN<-read.table(paste0(base_dir, "gradtest_IRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
tanh_IRNN<-read.table(paste0(base_dir, "gradtest-tanh_IRNN_T100_n30.hidden_gradients.txt"), header=TRUE)

which<-rep("LSTM", nrow(LSTM))
which<-c(which, rep("complex_RNN", nrow(complex_RNN)))
which<-c(which, rep("IRNN (relu)", nrow(IRNN)))
which<-c(which, rep("IRNN (tanh)", nrow(tanh_IRNN)))

da<-rbind(LSTM, complex_RNN)
da<-rbind(da, IRNN)
da<-rbind(da, tanh_IRNN)

da<-data.frame(da, which)

# --- now for plot --- #
ggplot(da, aes(x=k, y=norm, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("cost gradient wrt hidden state h_k") + xlab("k") + ylab("|dC/dh_k|") + scale_y_log10() + geom_hline(yintercept=1, linetype='dotted') + theme_bw()

ggsave(paste0(base_dir, "g8_grads_models_adding.png"), width=4.5, height=3)
