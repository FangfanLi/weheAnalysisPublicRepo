'''
This is for analyzing packet traces collected on the client
Input : one pcap file, one
Output : the bytes/time graph for this recorded trace,
blue dots for the correctly received ones,
red Xs for the retransmitted ones,
yellow dots for the out-of-order ones,
green dots for the acknowledgement sent to the server
The background colors reflect the video quality being played
'''

import sys, subprocess, os, numpy
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Pac(object):
    def __init__(self, ts, psize, retrans):
        self.ts = float(ts)
        self.psize = int(psize)
        self.retrans = retrans

    def update(self, ts, psize, retrans):
        self.ts = float(ts)
        self.psize = int(psize)
        self.retrans = retrans


def GetPacLists(pcapFile, serverPort=None):
    if (serverPort is None):
        print 'Please provide server Port'
        sys.exit()

    src_port = 'tcp.srcport'
    dst_port = 'tcp.dstport'
    if 'QUIC' in pcapFile:
        src_port = 'udp.srcport'
        dst_port = 'udp.dstport'

    cmd = ['tshark', '-r', pcapFile, '-T', 'fields', '-E', 'separator=/t', '-e', src_port,
           '-e', dst_port, '-e', 'frame.time_relative', '-e', 'frame.len', '-e', 'tcp.analysis.retransmission']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    pacLists = {}

    for sl in output.splitlines():
        l = sl.split('\t')
        src_port = l[0]
        dst_port = l[1]
        # Get the info of this packet
        try:
            # time relative to the beginning of the connection
            time = l[2]
            # the size of this packet
            psize = l[3]
            retrans = l[4]
        except:
            continue
        # only cares about server -> client packets
        if src_port != serverPort:
            continue

        pac = Pac(time, psize, retrans)

        key_1 = (src_port, dst_port)
        key_2 = (dst_port, src_port)
        if key_1 in pacLists:
            pacLists[key_1].append(pac)
        elif key_2 in pacLists:
            pacLists[key_2].append(pac)
        else:
            pacLists[key_1] = [pac]

    return pacLists


def plotting_bytes(ax1,pacLists,pcapFile,serverPort,contentprovider):
    in_Colors = ['#fcbba1', '#a6bddb', '#a1d99b', '#d4b9da']
    re_Colors = ['#99000d', '#034e7b', '#005a32', '#91003f']
    retrans_ts = []
    retrans_byte = []
    firstt_ts = []
    firstt_byte = []
    all_pacs_list = []

    for stream in pacLists:
        print(stream, len(pacLists[stream]))
        for pac in pacLists[stream]:
            all_pacs_list.append(pac)

    all_pacs_list.sort(key=lambda x: x.ts)
    # now all_pacs_list has all the packets from multiple connections combines
    # next step is to update the second value in the tuple from psize to bytes_received
    bytes_received = 0
    for pac in all_pacs_list:
        if pac.retrans:
            retrans_ts.append(pac.ts)
            retrans_byte.append(bytes_received/1000000)
        else:
            bytes_received += pac.psize
            firstt_ts.append(pac.ts)
            firstt_byte.append(bytes_received/1000000)

    plotting_dic = {}
    plotting_dic['firstt_byte'] = firstt_byte
    plotting_dic['firstt_ts'] = firstt_ts
    plotting_dic['retrans_ts'] = retrans_ts
    plotting_dic['retrans_byte'] = retrans_byte
    plotting_dic['server_port'] = serverPort
    plotting_dic['contentprovider'] = contentprovider
    plotting_dic['all_pacs_list'] = len(all_pacs_list)
    plotting_dic['fileName'] = pcapFile.split('.')[0].split('/')[-1]


    fileName = pcapFile.split('.')[0].split('/')[-1]

    with open(fileName+'.json', 'w') as fp:
        json.dump(plotting_dic, fp)

    if contentprovider == 'Netflix':
        color,re_color = in_Colors[1],re_Colors[1]
    else:
        color,re_color = in_Colors[0],re_Colors[0]
    ax1.plot(firstt_ts, firstt_byte, 'o', markerfacecolor='none', markeredgewidth=3, markersize=15,
             markeredgecolor=color, label=contentprovider+' First Arrival')
    ax1.plot(retrans_ts, retrans_byte, 'x', markerfacecolor='none', markeredgewidth=3, markersize=15,
             markeredgecolor=re_color, label=contentprovider+' Retrans')
    
    ax1.set_xlim(0,25)
    ax1.set_ylim(0,10)

    # print('\r\n goodput rate', float(firstt_byte[-1]) / float(firstt_ts[-1]) * 8 / 10 ** 6, len(firstt_ts), len(retrans_ts))
    retransmission_rate = float(len(retrans_ts)) / float(len(all_pacs_list))
    # print('\r\n retransmission rate ', retransmission_rate)
    ax1.legend(loc='best', markerscale=2, fontsize=26,ncol=2)
    ax1.set_ylabel('Megabytes')

    return ax1

def main():
    # three inputs
    # pcapFile
    # serverPort try 443

    try:
        pcapFile = sys.argv[1]
        pcapFile2 = sys.argv[2]
        serverPort = sys.argv[3]
        contentprovider = sys.argv[4]
        contentprovider2 = sys.argv[5]
    except:
        print('\r\n Please provide the pcap files, the server port, and the video streaming service as inputs: '
              '[pcapFile1] [pcapFile2] [serverPort] [Videostreaming service 1] [Videostreaming service 2]... ')
        sys.exit()



    # Step 1, plot the video quality changes

    matplotlib.rcParams.update({'font.size': 36})
    # fig, ax = plt.subplots(figsize=(22, 8))
    fig, (ax1) = plt.subplots(1, 1, figsize=(22,7))
    # plt.subplots_adjust(wspace=0.1, hspace=9)



    # Step 2, plot the packet traces

    # get one packet list for each connection (defined by 4 tuples)
    # Since srcIP and dstIP should already be filtered and unique
    # each connection can be defined by dst and src port
    # pacLists = {(tcp.src1, tcp.dst1) : packet_list1,
    # (tcp.src2, tcp.dst2) : packet_list2}
    # packet_list = [Pac1, Pac2], each Pac object has three values
    # Where ts is the timestamp, psize is the packet size,
    # retrans is the boolean shows whether this packet is a retransmission
    pacLists = GetPacLists(pcapFile, serverPort=serverPort)
    pacLists2 = GetPacLists(pcapFile2, serverPort=serverPort)
    ax1_out = plotting_bytes(ax1,pacLists,pcapFile,serverPort,contentprovider)
    ax1.set_xlabel('Time (s)')
    
    ax2_out = plotting_bytes(ax1,pacLists2,pcapFile2,serverPort,contentprovider2)
    # combine packets from all lists
    # outcome should be two lists for first transmit and retransmission
    # each item in the lists is (ts, bytes_received)
    # where bytes_received is the total number of bytes received (retransmission excluded) at each point


    # plt.savefig('{}_Byte_Retrans_{}_Goodput_{}.png'.format(fileName,retransmission_rate,float(firstt_byte[-1]) / float(firstt_ts[-1]) * 8 / 10 ** 6), bbox_inches='tight')
    # plt.show()
    plt.savefig('a.png',bbox_inches="tight")

if __name__ == "__main__":
    main()