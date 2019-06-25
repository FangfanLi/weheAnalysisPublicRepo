# weheAnalysisPublicRepo/impact on videostreaming

To create the bytes over time plot, run the bytes_over_time2.py script with the following inputs:<br/>
python bytes_over_time2.py Netflix_Tmobile_El2_5_pin.pcap Youtube_TMobile_Fire2_4_pin.pcap 443 Netflix Youtube

In order to create the stacked historgram of video streaming qualities, run the stacked_resolution.py script as follows:<br/>
python stacked_resolution.py stacked_youtube.json stacked_netflix.json stacked_amazon.json

Finally, to generate the CDF of throughputs plot, first speficy the video streaming service variable in the script:<br/>
service = 'Amazon_and_VPN'
Then you can run the following script.
python plotCDFMulti_Final.py



