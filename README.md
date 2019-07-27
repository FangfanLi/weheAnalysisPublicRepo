# weheAnalysisPublicRepo
The two scripts are used for analyzing Wehe data.

WeheDataParser.py aggregates Wehe tests. It separates tests into directories based on ISP, and then stores tests into json files based on ISP-app pair tested. For example, there are YouTube and Netflix tests from carrier A and carrier B. The script first creates two directories for A and B, and then creates two json files within each directory (e.g., carrierA_Youtube.json) and each json file contains all tests for that ISP-app pair. You can download Wehe tests separated by date at https://wehe-data.ccs.neu.edu/.

1. After downloading wehe tests, uncompress them in some directory /weheDirectory/.
2. You can have multiple sub directories (tests from multiple dates) like /weheDirectory/201X-XX-XX.
3. Run ```python WeheDataParser.py /weheDirectory/ /weheSeparateByISP/```
4. /weheSeparateByISP/ contains all tests separated by ISP-app pair, and can be used for next step.


weheAnalysis.py does the analysis for each ISP-app pair. At a very high level, it first uses KS test to check whether the throughput distributions of original and bit-inverted replays are different. Second, it runs KDE to check whether the throughputs of original replays aggregate around certain value, which is a clear sign of fixed-rate throttling. For more details, please refer to the paper.

1. Run ```python3 weheAnalysis.py /weheSeparateByISP/```
2. Results will be stored in weheDiffStat.json, and the plots can be found in each subdirectories in /weheSeparateByISP/

ExampleWeheDataset.zip contains 300 YouTube tests from AT&T users.

A sample run with only ExampleWeheDataset:

1. Unzip ExampleWeheDataset.zip to some /directory/
2. Run ```python3 weheAnalysis.py /directory/ExampleWeheDataset/ ```
3. Detection results will be in weheDiffStat.json generated in the scripts' directory, which contains metadata such as detected throttling rate.
4. All plots such as CDF of throughputs, locations of tests ... can be found in /directory/ExampleWeheDataset/SubsetsATT (cellular)/

A sample run with /weheSeparateByISP/ created in the previous step:
1. Run ```python3 weheAnalysis.py /weheSeparateByISP/  ```

The scripts for analyzing the impact on video streaming are in the impact_on_videostreaming directory.
