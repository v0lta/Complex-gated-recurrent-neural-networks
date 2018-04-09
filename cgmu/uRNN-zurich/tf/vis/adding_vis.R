# /usr/local/bin/R
# visualisation for adding task!
library(ggplot2)

#base_dir<-"/home/hyland/git/complex_RNN/tf/output/adding"
base_dir<-"/Users/stephanie/PhD/git/complex_RNN/tf/output/adding"

args<-commandArgs(TRUE)
T_val<-args[1]

# --- constants --- #
batch_size<-20
batch_skip<-150

# --- IRNN --- #
IRNN_fname<-ifelse(T_val==750, "IRNN_T750_n80.vali.txt", paste0("lr1e-4_IRNN_T", T_val, "_n80.vali.txt"))
IRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/", IRNN_fname), header=T)
num_updates <- batch_skip * seq(nrow(IRNN_trace))
num_examples<- batch_size * num_updates
cost <- IRNN_trace$vali_cost
which <- rep("IRNN", nrow(IRNN_trace))

da<-data.frame(num_updates, num_examples, cost, which)

# --- LSTM --- #
LSTM_trace<-read.table(paste0(base_dir, "/T", T_val, "/LSTM_T", T_val, "_n40.vali.txt"), header=TRUE)
num_updates <- batch_skip * seq(nrow(LSTM_trace))
num_examples<- batch_size * num_updates
cost <- LSTM_trace$vali_cost
which <- rep("LSTM", nrow(LSTM_trace))
dtemp <- data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- tanhRNN --- #
tanhRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/tanhRNN_T", T_val, "_n80.vali.txt"), header=TRUE)
num_updates <- batch_skip * seq(nrow(tanhRNN_trace))
num_examples<- batch_size * num_updates
cost <- tanhRNN_trace$vali_cost
which <- rep("tanhRNN", nrow(tanhRNN_trace))
dtemp<-data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- uRNN --- #
#uRNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/relu_uRNN_T", T_val, "_n20.vali.txt"), header=TRUE)
#uRNN_trace<-read.table(paste0(base_dir, "/relu-fbias-l0_uRNN_T", T_val, "_n30.vali.txt"), header=TRUE)
#num_updates <- batch_skip * seq(nrow(uRNN_trace))
#num_examples<- batch_size * num_updates
#cost <- uRNN_trace$vali_cost
#which <- rep("uRNN", nrow(uRNN_trace))
#dtemp<-data.frame(num_updates, num_examples, cost, which)

#da<-rbind(da, dtemp)

# --- complexRNN --- #
complex_RNN_trace<-read.table(paste0(base_dir, "/T", T_val, "/v2_complex_RNN_T", T_val, "_n512.vali.txt"), header=TRUE)
batch_skip <- 50            # NOTE DIFFERENT
num_updates <- batch_skip * seq(nrow(complex_RNN_trace))
num_examples<- batch_size * num_updates
cost <- complex_RNN_trace$vali_cost
which <- rep("complex_RNN", nrow(complex_RNN_trace))
dtemp<-data.frame(num_updates, num_examples, cost, which)

da<-rbind(da, dtemp)

# --- NOW FOR PLOT --- #
#ggplot(da, aes(x=num_updates, y=cost, group=which, colour=which)) + geom_point(cex=0.3) +  geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 0.21)) + ggtitle(paste0("adding, T = ", T_val)) + geom_hline(yintercept=0.167, color="black", linetype="dashed", alpha=0.5)
ggplot(da, aes(x=num_updates/100, y=cost, group=which, colour=which)) + geom_point(cex=0.05) +  geom_line(alpha=0.1) + coord_cartesian(ylim=c(0, 0.5), xlim=c(0, 300)) + ggtitle(paste0("adding, T = ", T_val)) + geom_hline(yintercept=0.167, color="black", linetype="dashed", alpha=0.5) + scale_colour_manual(values=c("darkgoldenrod2", "chartreuse3", "red", "blue")) + theme_bw() + xlab("training steps (hundreds)") + ggtitle("Sequence length = 400") + ylab("MSE")
#ggplot(da, aes(x=num_updates/100, y=cost, group=which, colour=which)) + geom_point(cex=0.05) +  geom_line(alpha=0.2) + coord_cartesian(ylim=c(0, 1.0), xlim=c(0, 300)) + ggtitle(paste0("adding, T = ", T_val)) + geom_hline(yintercept=0.167, color="black", linetype="dashed", alpha=0.5) + scale_colour_manual(values=c("darkgoldenrod2", "chartreuse3", "red", "pink", "blue")) + theme_bw() + xlab("training steps (hundreds)") + ggtitle("Sequence length = 100") + ylab("MSE")
#ggsave(paste0(base_dir, "/adding_T", T_val, ".png"), width=4.5, height=3)
ggsave(paste0(base_dir, "/adding_T", T_val, ".pres.png"), width=5.5, height=3)
