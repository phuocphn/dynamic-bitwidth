import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('tkagg')
import matplotlib.gridspec as gridspec

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="std_stepsize.pkl", type=str, help="Input file")
parser.add_argument('--output_dir', default="output", type=str,)
args = parser.parse_args()


with open(args.logfile,'br') as f:
    data = pickle.load(f)

# print (data.keys())
# from IPython import embed; embed()



for layer, logs in data.items():
	if ".quan_w" in layer:
		continue
		
	print ("Layer: ", layer)
	actin_std = []
	actout_std = []
	gradin_std = [] 
	mse = []
	sqnr = []
	stepsize = []
	for timestamp in logs:
		actin_std.append(timestamp['actin_std'])
		actout_std.append(timestamp['actout_std'])
		gradin_std.append(timestamp['gradin_std'])
		mse.append(timestamp['mse'])
		sqnr.append(timestamp['sqnr'])
		stepsize.append(timestamp['stepsize'])



	fig = plt.figure(constrained_layout=True, figsize=(15,6))
	spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
	ax1 = fig.add_subplot(spec[0, 0])
	ax2 = fig.add_subplot(spec[0, 1])
	ax3 = fig.add_subplot(spec[0, 2])
	ax4 = fig.add_subplot(spec[1, 0])
	ax5 = fig.add_subplot(spec[1, 1])
	ax6 = fig.add_subplot(spec[1, 2])




	ax1.plot(list(range(len(sqnr))), sqnr,  color='red' ,label='sqnr')
	ax2.plot(list(range(len(actin_std))), actin_std, color='green', label='actin_std')
	ax3.plot(list(range(len(mse))), mse,  color='blue', label='mse')
	ax4.plot(list(range(len(stepsize))), stepsize, color='purple', label='stepsize')
	ax5.plot(list(range(len(actout_std))), actout_std, color='orange', label='actout_std')
	ax6.plot(list(range(len(gradin_std))), gradin_std, color='brown', label='gradin_std')

	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax5.legend()
	ax6.legend()

	plt.savefig(args.output_dir +  layer +".png")
	# plt.show()
