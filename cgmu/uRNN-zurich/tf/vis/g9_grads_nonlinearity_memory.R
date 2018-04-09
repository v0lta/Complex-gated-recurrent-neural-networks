#!/usr/bin/env R

library(ggplot2)

#base_dir <- "/home/hyland/git/complex_RNN/tf/output/memory/"
base_dir <- "/Users/stephanie/PhD/git/complex_RNN/tf/output/memory/"

#complex_RNN <- read.table(paste0(base_dir, "gradtest-complex_RNN_T100_n30.hidden_gradients.txt"), header=T)
relu<-read.table(paste0(base_dir, "gradtest-relu_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
leakyrelu<-read.table(paste0(base_dir, "gradtest-leakyrelu_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
relumod<-read.table(paste0(base_dir, "gradtest-relumod_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
tanh<-read.table(paste0(base_dir, "gradtest-tanh_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
sigmoid<-read.table(paste0(base_dir, "gradtest_sigmoid_uRNN_T100_n30.hidden_gradients.txt"), header=TRUE)
complex_RNN<-read.table(paste0(base_dir, "gradtest_complex_RNN_T100_n30.hidden_gradients.txt"), header=TRUE)

which<-rep("relu", nrow(relu))
which<-c(which, rep("relumod", nrow(relumod)))
which<-c(which, rep("leakyrelu", nrow(leakyrelu)))
which<-c(which, rep("tanh", nrow(tanh)))
which<-c(which, rep("sigmoid", nrow(sigmoid)))
which<-c(which, rep("complex_RNN", nrow(complex_RNN)))

da<-rbind(relu, relumod)
da<-rbind(da, leakyrelu)
da<-rbind(da, tanh)
da<-rbind(da, sigmoid)
da<-rbind(da, complex_RNN)

da<-data.frame(da, which)

# --- now for plot --- #
ggplot(da, aes(x=k, y=norm, group=which, colour=which)) + geom_point(cex=0.3) + geom_line(alpha=0.2) + facet_grid(batch~.) + ggtitle("cost gradient wrt hidden state h_k") + xlab("k") + ylab("|dC/dh_k|") + scale_y_log10() + theme_bw() + scale_colour_manual(values=c("blue", "darkgray", "black", "turquoise", "green", "deeppink3"))
#+ geom_hline(yintercept=1, linetype='dotted')

ggsave(paste0(base_dir, "g9_grads_nonlinearity_memory.png"), width=5, height=4)
