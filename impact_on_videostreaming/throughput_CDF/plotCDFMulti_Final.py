'''
This will plot CDF plots of throughputs for all pcaps
'''

import subprocess, os, sys, numpy, random
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
from scipy.stats import ks_2samp
import glob
import matplotlib
from collections import OrderedDict
import json

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 16})

def list2CDF(myList):
    myList = sorted(myList)
    x   = [0]
    y   = [0]
    for i in range(len(myList)):
        x.append(myList[i])
        y.append(float(i+1)/len(myList))
    
    return x, y


def main():
  fig, ax = plt.subplots(figsize=(8, 3.6))
  service = 'Amazon_and_VPN'
  throughputs = {}
  
  for file in sorted(glob.glob('CDF_Data/'+service+'/*.txt')):
      x_y = {}
      xputs = []
      if file not in throughputs:
        throughputs[file] = {}
      with open(file, 'r') as pcap:
  	    for line in pcap: 
  	        line = line.strip() #or some other preprocessing
  	        xputs.append(float(line)) #storing everything in memory!
      x , y =list2CDF(xputs)
      x_y['x'] = x
      x_y['y'] = y
      throughputs[file] = x_y

      with open(service+'.json', 'w') as fp:
        json.dump(throughputs, fp)

      isvpn = 0
      if len(file.split('/')[2].split('_')) > 2:
        isvpn = 1

      carrier = file.split('/')[2].split('_')[1].replace('.txt','')
      if carrier == 'Sprint':
        if isvpn:
            plt.plot(x, y, 'k-', color='darkorchid', linestyle='dashed', label=carrier+" VPN")
        else:
            plt.plot(x, y, 'k-', color='orchid', label=carrier)
      if carrier == 'ATT':
      	if isvpn:
          	plt.plot(x, y, 'k-', color='darkblue', linestyle='dashed', label=carrier+" VPN")
      	else:
          	plt.plot(x, y, 'k-', color='blue', label=carrier)
      if carrier == 'TMobile':
      	if isvpn:
          	plt.plot(x, y, 'k-', color='darkgreen', linestyle='dashed', label=carrier+" VPN")
      	else:
          	plt.plot(x, y, 'k-', color='green', label=carrier)
      if carrier == 'Verizon':
        if isvpn:
          	plt.plot(x, y, 'k-', color='darkred', linestyle='dashed', label=carrier+" VPN")
        else:
			plt.plot(x, y, 'k-', color='red', label=carrier)

      # if carrier == 'WiFi':
      #     plt.plot(x, y, 'k-', color='green',  linestyle='solid', label=carrier)
  # plt.legend(loc='best', prop={'size':16})
  
  plt.grid()
  plt.xlabel('Throughput (Mbits/sec)')
  plt.ylabel('CDF')
  plt.ylim((0, 1.0))
  plt.xlim((-1, 20))
  plt.tight_layout()
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  
  # manually reordering the legend
  plt.legend(bbox_to_anchor=(-0.03,1.02,1.05,0.2),mode="expand",prop={'size': 14} ,ncol=4, loc="lower left")
  fig .tight_layout()

  # plt.show()
  plt.savefig('CDF_Data/'+service+'_xputCDF.png',bbox_inches="tight")



if __name__ == "__main__":
    main()


