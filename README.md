# weheAnalysisPublicRepo
The scripts are for analyzing Wehe data.

WeheDataParser.py aggregates Wehe tests. It separates tests into directories based on ISPs tests, and then stores tests into json files based on ISP-app pair tested. For example, there are YouTube and Netflix tests from carrier A and carrier B. The script first creates two directories for A and B, and then creates two json files within each directory (e.g., carrierA_Youtube.json) and each json file contains all tests for that ISP-app pair. You can download most recent Wehe tests here[soon].

weheAnalysis.py does the analysis for each ISP-app pair. At a very high level, it first uses KS test to check whether the throughput distributions of original and bit-inverted replays are different. Second, it runs KDE to check whether the throughputs of original replays aggregate around certain value, which is a clear sign of fixed-rate throttling. For more details, please refer to the paper.

ExampleWeheDataset.zip contains 300 YouTube tests from AT&T users.

A sample run:

1. unzip ExampleWeheDataset.zip to some /directory/
2. python3 weheAnalysis.py /directory/ExampleWeheDataset/ 
3. Detection results will be in weheDiffStat.json generated in the scripts' directory, which contains metadata such as detected throttling rate.
4. All plots such as CDF of throughputs, locations of tests ... can be found in /directory/ExampleWeheDataset/SubsetsATT (cellular)/


The scripts for analyzing the impact on video streaming are in the impact_on_videostreaming directory.
