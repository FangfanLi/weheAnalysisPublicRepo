import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess, os, sys, numpy, random
import matplotlib.pyplot as plt
import glob
import matplotlib
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import matplotlib.gridspec as gridspec
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 14})



def plot_stacked_amazon(histogram_data,ax):
	N = 18
	list_of_tuples = []
	list_of_tuples_labels = []
	# resolutions = ['v5','v6','v7','v8','v9','v12']
	resolutions = ['v5','v8','v12']
	tuple_v5 = ()
	# tuple_v6 = ()
	# tuple_v7 = ()
	tuple_v8 = ()
	# tuple_v9 = ()
	tuple_v12 = ()
	# tuples = [tuple_v5,tuple_v6,tuple_v7,tuple_v8,tuple_v9,tuple_v12]
	tuples = [tuple_v5,tuple_v8,tuple_v12]


	with open(histogram_data, 'r') as fp:
		histogram_data = json.load(fp)

	for entry in list(histogram_data.keys())[0:18]:
		cnt = 0
		for i in resolutions:
			
			if i in histogram_data[entry].keys():
				tuples[cnt] =tuples[cnt] + (float(histogram_data[entry][i]*100)/5,)
			else:
				tuples[cnt] = tuples[cnt] + (float(0.0),)
			cnt += 1

		if entry.split('_')[2] == 'VPN':
			list_of_tuples_labels.append(entry.split('_')[1]+" VPN")
		else:
			list_of_tuples_labels.append(entry.split('_')[1])
			
	ind = np.arange(N)    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence
	p1 = ax.bar(ind, tuples[0], width,color='#08519c',label='LD',edgecolor='black',)
	p2 = ax.bar(ind, tuples[1], width,color='#3182bd',label='SD low',edgecolor='black',
	             bottom=tuples[0])
	p3 = ax.bar(ind, tuples[2], width, color='#bdd7e7',label='HD low',edgecolor='black',
	             bottom=tuple(map(sum,zip(tuples[0],tuples[1]))))


	ax.set_ylabel('Percentage of Time')
	ax.axvline(x=3.5,color='black',linestyle='--')
	ax.axvline(x=7.5, color='black',linestyle='--')
	ax.axvline(x=11.5, color='black',linestyle='--')
	ax.axvline(x=15.5, color='black',linestyle='--')
	ax.set_xticks(ind, ('ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon', 'ATT', 'Sprint','TMobile','Verizon','WiFi','WiFi VPN'))
	ax.set_yticklabels(np.arange(0, 110, 25))
	ax.set_xticks(ind)
	ax.set_xticklabels(['ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon', 'ATT', 'Sprint','TMobile','Verizon','WiFi','WiFi VPN'],rotation=45,ha="center")
	ax.annotate('Exposed', xy=(270, 400), xycoords='figure pixels')
	ax.annotate('Default', xy=(210, 370), xycoords='figure pixels')
	ax.annotate('Max Data', xy=(340, 370), xycoords='figure pixels')
	ax.annotate('VPN', xy=(580, 400), xycoords='figure pixels')
	ax.annotate('Default',      xy=(500, 370), xycoords='figure pixels')
	ax.annotate('Max Data', xy=(610, 370), xycoords='figure pixels')
	ax.annotate('WiFi', xy=(760, 370), xycoords='figure pixels')
	ax.set_xlabel('(a) Amazon',fontsize=18)
	return ax

def plot_stacked_net(histogram_data,ax):
	N = 18
	list_of_tuples = []
	list_of_tuples_labels = []
	# resolutions = ['384x216','480x270','608x342','640x480','768x432','720x480','960x540','1280x720']
	resolutions = ['384x216','608x342','640x480','960x540','1280x720']
	tuple_384x216 = ()
	# tuple_480x270 = ()
	tuple_608x342 = ()
	tuple_640x480 = ()
	# tuple_768x432 = ()
	# tuple_720x480 = ()
	tuple_960x540 = ()
	tuple_1280x720 = ()
	# tuple_1080 = ()
	# tuples = [tuple_384x216, tuple_480x270,tuple_608x342,tuple_640x480,tuple_768x432,tuple_720x480,tuple_960x540,tuple_1280x720]
	tuples = [tuple_384x216,tuple_608x342,tuple_640x480,tuple_960x540,tuple_1280x720]

	with open(histogram_data, 'r') as fp:
		histogram_data = json.load(fp)

	for entry in list(histogram_data.keys())[0:18]:
		cnt = 0
		for i in resolutions:
			
			if i in histogram_data[entry].keys():
				tuples[cnt] =tuples[cnt] + (float(histogram_data[entry][i]*100)/5,)
			else:
				tuples[cnt] = tuples[cnt] + (float(0.0),)
			cnt += 1
		if entry.split('_')[2] == 'VPN':
			list_of_tuples_labels.append(entry.split('_')[1]+"VPN")
		else:
			list_of_tuples_labels.append(entry.split('_')[1])
			

	ind = np.arange(N)    # the x locations for the groups
	width = 0.35       # the width of the bars: can also be len(x) sequence

	p1 = ax.bar(ind, tuples[0], width,color='#08519c',label='LD',edgecolor='black',)
	p2 = ax.bar(ind, tuples[1], width,color='#3182bd',label='SD low',edgecolor='black',
	             bottom=tuples[0])
	p3 = ax.bar(ind, tuples[2], width, color='#6baed6',label='SD',edgecolor='black',
	             bottom=tuple(map(sum,zip(tuples[0],tuples[1]))))
	p4 = ax.bar(ind, tuples[3], width, color='#bdd7e7',label='HD low',edgecolor='black',
			bottom=tuple(map(sum,zip(tuples[0],tuples[1],tuples[2]))))
	p5 = ax.bar(ind, tuples[4], width, color='#eff3ff',label='HD',edgecolor='black',
			bottom=tuple(map(sum,zip(tuples[0],tuples[1],tuples[2],tuples[3]))))


	# ax.ylabel('Percentage of Time')
	ax.axvline(x=3.5, color='black',linestyle='--')
	ax.axvline(x=7.5, color='black',linestyle='--')
	ax.axvline(x=11.5, color='black',linestyle='--')
	ax.axvline(x=15.5, color='black',linestyle='--')
	ax.set_xticks(ind)
	ax.set_xticklabels(['ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon', 'ATT', 'Sprint','TMobile','Verizon','WiFi','WiFi VPN'],rotation=45,ha="center")

	# plt.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), (resolutions[0], resolutions[1], resolutions[2],resolutions[3],resolutions[4]))
	
	# plt.legend(bbox_to_anchor=(0,1.12,1,0.2),mode="expand" , ncol=4, loc="lower left")
	ax.set_yticklabels(np.arange(0, 110, 25))
	ax.annotate('Exposed', xy=(1010, 400), xycoords='figure pixels')
	ax.annotate('Default', xy=(950, 370), xycoords='figure pixels')
	ax.annotate('Max Data', xy=(1080, 370), xycoords='figure pixels')
	ax.annotate('VPN', xy=(1320, 400), xycoords='figure pixels')
	ax.annotate('Default',      xy=(1240, 370), xycoords='figure pixels')
	ax.annotate('Max Data', xy=(1370, 370), xycoords='figure pixels')
	ax.annotate('WiFi', xy=(1510, 370), xycoords='figure pixels')
	ax.set_xlabel('(b) Netflix',fontsize=18)
	return ax


def plot_stacked(histogram_data, ax):
	N = 10
	list_of_tuples = []
	list_of_tuples_labels = []
	resolutions = ['240p','360p','480p','720p','1080p']
	tuple_240p = ()
	tuple_360p = ()
	tuple_480p = ()
	tuple_720p = ()
	tuple_1080p = ()
	tuples = [tuple_240p, tuple_360p,tuple_480p,tuple_720p,tuple_1080p]
	with open(histogram_data, 'r') as fp:
		histogram_data = json.load(fp)

	for entry in list(histogram_data.keys())[0:10]:
		cnt = 0

		for i in resolutions:
			
			if i in histogram_data[entry].keys():
				tuples[cnt] =tuples[cnt] + (float(histogram_data[entry][i]*100)/5,)
			else:
				tuples[cnt] = tuples[cnt] + (float(0.0),)
			cnt += 1

		if entry.split('_')[2] == 'VPN':
			list_of_tuples_labels.append(entry.split('_')[1]+" VPN")
		else:
			list_of_tuples_labels.append(entry.split('_')[1])



	ind = np.arange(N)    # the x locations for the groups
	width = 0.03      # the width of the bars: can also be len(x) sequence
	ind = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	p1 = ax.bar(ind, tuples[0], width,color='#08519c', label ='LD',edgecolor='black')
	p2 = ax.bar(ind, tuples[1], width,color='#3182bd',label='SD low',edgecolor='black',
	             bottom=tuples[0])
	p3 = ax.bar(ind, tuples[2], width, color='#6baed6',label='SD',edgecolor='black',
	             bottom=tuple(map(sum,zip(tuples[0],tuples[1]))))
	p4 = ax.bar(ind, tuples[3], width, color='#eff3ff',label='HD',edgecolor='black',
	             bottom=tuple(map(sum,zip(tuples[0],tuples[1],tuples[2]))))

	p5 = ax.bar(ind, tuples[4], width, color='orchid',
	             bottom=tuple(map(sum,zip(tuples[0],tuples[1],tuples[2],tuples[3]))))

	
	ax.axvline(x=0.35,color='black',linestyle='--')
	# # plt.axvline(x=3.5,color='black')
	# # plt.axvline(x=7.52058956,color='black')
	ax.axvline(x=0.75, color='black',linestyle='--')
	
	# ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
	ax.set_xticks(ind)
	ax.set_xticklabels(['ATT','Sprint','TMobile','Verizon','ATT','Sprint','TMobile','Verizon','WiFi','WiFi\nVPN'],rotation=45,minor=False,ha="center")
	ax.set_yticklabels(np.arange(0, 110, 25))
	# # plt.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), (resolutions[0], resolutions[1], resolutions[2],resolutions[3],resolutions[4]))
	ax.set_xlabel('(c) Youtube',fontsize=18)
	
	ax.annotate('Exposed', xy=(1680, 370), xycoords='figure pixels')
	ax.annotate('VPN',      xy=(1870, 370), xycoords='figure pixels')
	ax.annotate('WiFi', xy=(2000, 370), xycoords='figure pixels')


	return ax





try:
	json_file_youtube = sys.argv[1]
	json_file_netflix = sys.argv[2]
	json_file_amazon = sys.argv[3]
except:
    print('\r\n Please provide the three JSON files:'
      '[Youtube JSON file 1] [Netflix JSON file 2] [Amazon JSON file 3] ... ')
    sys.exit()


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28,3))
fig3 = plt.figure(figsize=(28,3))
spec3 = gridspec.GridSpec(ncols=9, nrows=1)

ax1 = fig3.add_subplot(spec3[0, 0:3])
ax2 = fig3.add_subplot(spec3[0, 3:6])
ax3 = fig3.add_subplot(spec3[0, 6:8])



plt.subplots_adjust(wspace=0.3, hspace=4)

plt1 = plot_stacked_amazon(json_file_amazon,ax1)
plt2 = plot_stacked(json_file_youtube,ax3)
plt3 = plot_stacked_net(json_file_netflix,ax2)
# plt3 = plot_stacked_net(json_file_netflix,ax2)
# plt2 = plot_stacked(json_file_youtube,ax3)
# fig3.tight_layout()


# # The data
# x =  [1, 2, 3]
# y1 = [1, 2, 3]
# y2 = [3, 1, 3]
# y3 = [1, 3, 1]
# y4 = [2, 2, 3]

# # Labels to use in the legend for each line
# line_labels = ["Line A", "Line B", "Line C", "Line D"]


# # Create the legend
plt3.legend(bbox_to_anchor=(-.05,1.25,1,0.2),mode="expand" , ncol=5, loc="lower left")

# # Adjust the scaling factor to fit your legend text completely outside the plot
# # (smaller value results in more space being made for the legend)
# plt.subplots_adjust(right=0.85)

# # plt.show()
plt.savefig('a.png',bbox_inches="tight")
