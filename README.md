### Knowledge-Tracing at Khan Academy

This repo contains the basic code required to train DKT and SAKT for knowledge tracing. DKT and SAKT folders contain the model class script, data prep script and train/test loop script.
In developing the code for these models we relied on publicly available code in other similar repos (eg. https://github.com/THUwangcy/HawkesKT). 
However we made substantive updates to data processing (leveraging torch dataset/dataloader capabilities) as well as updates to the model code (eg. attention and padding masks in SAKT, added features). 
We also added various metrics to the train and test loop, as well as the code for the callibration plot. 

The results of the training of these models on Khan Academy data are posted on Khan Academy research blog and the detailed report can be found here. 

This work is made possible in part thanks to Schmidt Futures. 
