#  sdn_tc_fs

#### 下載資料集  
https://www.unb.ca/cic/datasets/vpn.html  
  
## 分類器
  
#### 安裝python3 

https://www.python.org/downloads/  
PATH記得打勾  
PATH記得打勾  
PATH記得打勾  

#### 安裝python所需的模組  
sklearn  
numpy  
pandas  
matplotlib  
seaborn  
pydotplus  
joblib  

#### 預處理(輸入pcap, 輸出parquet)  
classifier/preprocessing_pcap.py  
修改輸入輸出路徑  
大約需要8小時，放心如果記憶體不足中斷,處理完成的不會重複處理  

#### 訓練model(輸入parquet, 輸出model)  
classifier/train_test_sklearn.py  
修改輸入輸出路徑  
修改訓練類型  
輸入大約需要15分鐘，可以在最底下寫每次要的參數選項自動執行多次  

#### 將model儲存到流量排程的資料夾(安裝控制器ryu的位置)
ryu/ryu/app/ryu_customapp/models/


## 流量排程

#### 安裝python3
<code>
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-pip
</code>

#### 安裝所需的模組  
scapy

#### 安裝mininet
http://mininet.org/download/

#### 安裝ryu  
https://ryu.readthedocs.io/en/latest/getting_started.html

#### 設定重播pcap參數  
exp_config/exp_config.py  
修改設定  
修改產生封包的開始時間：timestring  
修改pcap路徑：PKT_FILE_LIST  
修改pcap對應的傳送間隔：PKT_FILE_MAP  
修改select函數：SCHEDULER_TYPE  
修改其他設定如執行時間及頻寬相關參數  
  
#### 執行mininet  
<code>
sudo python3 mininet/custom/custom_example_7to7.py  
</code>  

#### 執行ryu  
<code>
ryu-manager ryu/ryu/app/simple_switch_13_slice.py  
</code>  
  
#### 在mininet輸入  
<code>
mininet> xterm h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14  
</code>  
  
#### 在xterm執行 
##### client
在h1執行  
<code>python3 pktgen/cs/client1.py</code>  
在h2執行  
<code>python3 pktgen/cs/client2.py</code>  
在h3執行  
<code>python3 pktgen/cs/client3.py</code>  
...  
在h7執行  
<code>python3 pktgen/cs/client7.py</code>  
##### server
在h8執行  
<code>python3 pktgen/cs/server8.py</code>  
在h9執行  
<code>python3 pktgen/cs/server9.py</code>  
在h10執行  
<code>python3 pktgen/cs/server10.py</code>  
...  
在h14執行  
<code>python3 pktgen/cs/server14.py</code>  
  
#### 等待
等待讀取pcap直到出現ready  
等待到設定的開始時間  
  
#### 執行中  
依照設定的時間每個間隔儲存檔案  

#### 執行完成  
最後輸出client和server的收發時間儲存檔案  
注意，時間過長可能會耗盡記憶體，需要修改儲存方式  
