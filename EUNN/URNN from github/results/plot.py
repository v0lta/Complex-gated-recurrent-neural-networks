import matplotlib.pyplot as plt

def deserialize(path):
	loss = [float(l) for l in open(path, 'r').read().split('\n')[:-1]]
	return loss

def plot_cmp_100():
	cmp_simplernn100 = deserialize('cmp_simple_rnn120')
	cmp_lstm100 = deserialize('cmp_lstm120')

	plt.plot(cmp_simplernn100, 'r-')
	plt.plot(cmp_lstm100, 'b-')
	plt.show()

def plot_ap_100():
	ap_simplernn100 = deserialize('ap_simplernn100')
	ap_lstm100 = deserialize('ap_lstm100')

	plt.axis([0, 6000, 0, 2])

	plt.plot(ap_simplernn100, 'r-')
	plt.plot(ap_lstm100, 'b-')
	plt.show()

def plot_ap_200():
	ap_simplernn200 = deserialize('ap_simplernn200')
	ap_lstm200 = deserialize('ap_lstm200')

	plt.axis([0, 6000, 0, 2])

	plt.plot(ap_simplernn200, 'r-')
	plt.plot(ap_lstm200, 'b-')
	plt.show()


plot_cmp_100()
plot_ap_100()
plot_ap_200()

