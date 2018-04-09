library(ggplot2)

gg_color_hue <- function(n) {
    hues = seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
}

d<-12
noise<-0.01

fname_base<-paste0('../output/simple/random_d', d, '_noise', noise, '_bn20_nb50000_')
print(fname_base)

# --- get data --- #
fname<-paste0(fname_base, 'vali.txt')
data<-read.table(fname, header=T)
print(levels(factor(data$rep)))
data['method']<-NULL
print(levels(factor(data$experiment)))
if (d==6){
    data['experiment']<-factor(data$experiment, labels=c('full', 'J=16', 'J=25', 'J=36', 'J=4', 'J=9'))
    data$experiment<-factor(data$experiment, levels=c('J=4', 'J=9', 'J=16', 'J=25', 'J=36', 'full'))
} else if (d==12){
    data['experiment']<-factor(data$experiment, labels=c('full', 'J=144', 'J=25', 'J=49', 'J=81', 'J=9'))
    data$experiment<-factor(data$experiment, levels=c('J=9', 'J=25', 'J=49', 'J=81', 'J=144', 'full'))
    # cull rep 5 (didn't finish for most experiments)
    data <- subset(data, rep != 5)
}
print(levels(data$experiment))

# --- get an average time column --- #
ave_time<-rep(0, nrow(data))
data_aug<-data.frame(data, ave_time)
for (lev in levels(data$experiment)){
    data_subset <- subset(data, experiment==lev)[, c("t", "training_examples")]
    ave_t <- aggregate(data_subset, by=list(data_subset$training_examples), FUN=mean)$t
    ave_t_rep = rep(ave_t, length(unique(data$rep)))
    indices<-data_aug[, "experiment"] == lev
    data_aug[indices, "ave_time"] <- ave_t_rep
}
data_aug['t']<-NULL
data_aug['rep']<-NULL

# --- as function of time --- #
print("as a function of time...")

ggplot(data_aug, aes(x=ave_time, y=loss, group=experiment, color=experiment, fill=experiment)) + theme_bw() + stat_summary(fun.data = "mean_cl_boot", geom="smooth", alpha=0.07) + ylab("validation set loss") + xlab("time (s) (average)") 
#+ coord_cartesian(ylim=c(0, 5.5), xlim=c(0, 1500))             # d=6
ggsave(gsub(".txt", "_time.png", fname), width=4.5, height=3)
ggsave(gsub(".txt", "_time.pdf", fname), width=4.5, height=3)

print("... now zoomed in")
ggplot(data_aug, aes(x=ave_time, y=loss, group=experiment, color=experiment, fill=experiment)) + theme_bw() + stat_summary(fun.data = "mean_cl_boot", geom="smooth", alpha=0.07) + ylab("validation set loss") + xlab("time (s) (average)")  + coord_cartesian(ylim=c(0, 20.0), xlim=c(0, 5000))        # d = 12
#+ coord_cartesian(ylim=c(0, 2), xlim=c(0, 500))                # d = 6
ggsave(gsub(".txt", "_time.zoom.png", fname), width=4.5, height=3)
ggsave(gsub(".txt", "_time.zoom.pdf", fname), width=4.5, height=3)


# --- as a function of training examples --- #
print("as a function of traninig examples...")
data['rep']<-NULL

ggplot(data, aes(x=training_examples/1e6, y=loss, colour=experiment, group=experiment, fill=experiment)) +  xlab("training examples seen (millions)") + ylab("validation set loss") + theme_bw() + stat_summary(fun.data = "mean_cl_boot", geom = "smooth", alpha=0.1)
#+ coord_cartesian(ylim=c(0, 7), xlim=c(-0.005, 2))             # d = 6
ggsave(gsub(".txt", ".png", fname), width=4.5, height=3)
ggsave(gsub(".txt", ".pdf", fname), width=4.5, height=3)
# zoom version
print("... now zoomed in")
ggplot(data, aes(x=training_examples/1e6, y=loss, colour=experiment, group=experiment, fill=experiment)) +  xlab("training examples seen (millions)") + ylab("validation set loss") + theme_bw() + stat_summary(fun.data = "mean_cl_boot", geom = "smooth", alpha=0.2) + coord_cartesian(xlim=c(0, 1), ylim=c(0, 20))
#+ coord_cartesian(xlim=c(0, 0.3), ylim=c(0, 7))                # d = 6
ggsave(gsub(".txt", ".zoom.png", fname), width=4.5, height=3)
ggsave(gsub(".txt", ".zoom.pdf", fname), width=4.5, height=3)


# --- actually get the final score of J=4
# --- then find out when all the other experiments got to that level
# --- then plot bar chart (with error bars yo?)
# -- blehargh why am I doing this in R
LAST_AVE_J4 <- FALSE
if (LAST_AVE_J4){
    last_ave_J4 <- mean(subset(data, (experiment=="J=4")&(training_examples==max(data$training_examples)))$loss)
    col_time_to_loss <- c()
    col_experiment <- c()
    col_rep <- c()      # i am very sleep deprived
    # I AM SURE THIS IS AWFUL
    # but it works and that's all that matters here
    for (e in levels(data$experiment)){
        d_s <- subset(data, experiment==e)[, c("loss", "rep", "t")]
        # now for each rep, eugh
        if (e == "J=4"){
            next
        }
        for (r in levels(factor(d_s$rep))){
            d_s_s <- subset(d_s, rep==r)
            # now get the time to loss
            time <- min(d_s_s[d_s_s$loss <= last_ave_J4, ]$t)
            if (is.infinite(time)){
                # sorry god, sorry mom
                time<-99999
                col_time_to_loss <- c(col_time_to_loss, time)
                col_experiment <- c(col_experiment, e)
                col_rep <- c(col_rep, r)
                time<--999999
                col_time_to_loss <- c(col_time_to_loss, time)
                col_experiment <- c(col_experiment, e)
                col_rep <- c(col_rep, r)
            }
            else{
                # saveeee
                col_time_to_loss <- c(col_time_to_loss, time)
                col_experiment <- c(col_experiment, e)
                col_rep <- c(col_rep, r)
            }
        }
    }

    #jaysus
    time_to_loss <- data.frame(col_experiment, col_time_to_loss)
    names(time_to_loss) <- c("experiment", "t")
    time_to_loss$experiment <- factor(time_to_loss$experiment, levels<-c("J=4", "J=9", "J=16", "J=25", "J=36", "full"))
    print(time_to_loss)
    # plot er
    ggplot(time_to_loss, aes(x=experiment, y=t)) + stat_summary(fun.data = "mean_cl_boot", aes(colour=experiment), alpha=0.8) + geom_point() + theme_bw() + ylab("time to average J=4 final loss") + coord_cartesian(ylim=c(0, 700))+ scale_colour_manual(values=gg_color_hue(6)[c(2, 3, 4, 5, 6)])
    ggsave(gsub(".txt", ".bar.png", fname), width=5.5, height=4)
}
