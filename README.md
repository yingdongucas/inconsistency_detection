# VIEM

VIEM (short for Vulnerability Information Extraction Model) is a tool used for automatically extracting vulnerable software names and versions from unstructured reports. It combines a Named Entity Recognition (NER) model and a Relation Extraction (RE) model. The goal is to enable the possibility to continuously monitor different vulnerability reporting websites and periodically generate a **diff** from the CVE/NVD entries. 

More details can be found in our paper:
```
Towards the Detection of Inconsistencies in Public Security Vulnerability Reports
Ying Dong, Wenbo Guo, Yueqi Chen, Xinyu Xing, Yuqing Zhang, Gang Wang
USENIX Security 2019
```

## Dataset

Our manually labeled dataset contains the CVE summaries and unstructured vulnerability reports of 5,193 CVE IDs, including [ExploitDB](https://www.exploit-db.com/), [SecurityFocus Forum](https://www.securityfocus.com/archive/1) and [Openwall](http://www.openwall.com/). The dataset covers all of the 13 categories of vulnerabilities provided in [cvedetails](https://www.cvedetails.com/vulnerabilities-by-types.php). Both NER data and RE data are included. 

### NER data format
For the NER dataset, each word in a report is assigned one of the three labels: vulnerable software name (`SN`), vulnerable software version (`SV`), or others (`O`). Each line is a word followed by its label. Sentences are seperated with a new line.
```
This O 
PoC O 
has O 
been O 
tested O 
on O 
Apple SN
Watch SN 
3 SV
running O 
WatchOS SN
4.0.1 SV
. O 
```

### RE data format
For the RE dataset, we pair SN and SV entities by examining all the possible pairs within the same sentence. In above example, there are 4 entities indicated by 2 software (`Apple Watch` and `WatchOS`) and 2 versions (`3` and `4.0.1`). They can be combined in 4 different ways. By treating the combinations as 4 different pairs of entities, we assign a binary label to each of the combinations. The label can be `Y` and `N`. Each line contains with the index of the head entity and tail entity, the label, and the sentence. 
```
6 7 Y This PoC has been tested on Apple_Watch 3 runing WatchOS 4.0.1 .
6 10 N This PoC has been tested on Apple_Watch 3 runing WatchOS 4.0.1 .
9 10 Y This PoC has been tested on Apple_Watch 3 runing WatchOS 4.0.1 .
7 9 N This PoC has been tested on Apple_Watch 3 runing WatchOS 4.0.1 .
```

### Data structure
The data is organized by vulnerability category. It contains a large amount of data (3,448 CVE IDs) from one primary category (Memory Corruption) and a smaller amount of data (145 CVE IDs) from the other 12 categories to evaluate model transferability. 

Among Memory Corruption data, `memc_test_dup.txt` and `memc_full_dup.txt` are used for measurement purposes, where the corresponding CVE ID and report type/link of each sentence is marked, and duplicate sentences might exist. The structure of RE data is similar to that of NER data.
```
/dataset/ner_data/memc_train.txt
/dataset/ner_data/memc_test.txt
/dataset/ner_data/memc_valid.txt
/dataset/ner_data/memc_test_dup.txt
/dataset/ner_data/memc_full_dup.txt
```


## Requirements

The tool is implemented in Python 3. To install needed packages use:
```
pip3 install -r requirements.txt
```

## Named Entity Recognition Model

VIEM utilizes the state-of-the-art Named Entity Recognition (NER) model to identify the entities of our interest, i.e., the name and versions of the vulnerable software, those of vulnerable components and those of underlying software systems that vulnerable software depends upon.

### Training

`train_NER.py` is used for training the NER model, and accepts the following inputs:
```
python3 ner_model/train_NER.py --category <category> 
```
The flag `category` is one of the 13 vulnerability categories in the list:
```
['memc', 'bypass', 'csrf', 'dirtra', 'dos', 'execution', 'fileinc', 'gainpre', 'httprs', 'infor', 'overflow', 'sqli', 'xss']
```
When the category is not 'memc', the script will load the pre-trained model for Memory Corruption and transfer the model to the new category.


### Testing

`test_NER.py` is used for testing the NER model, and accepts the following inputs:
```
python3 ner_model/test_NER.py --category <category> --input_type <input_type> --transfer <bool> --gaze <bool>
```
The flag `input_type` specifies the type of the input test data, and could be `test`, `test_dup` and `full_dup`, corresponding to `category_test.txt`, `category_test_dup.txt` and `category_full_dup.txt` respectively. `transfer` indicates whether to use the transferred model or not. `gaze` means whether to use gazetteer or not.


## Relation Extraction Model

With the extracted entities, the next task of VIEM is to pair identified entities accordingly. VIEM first goes through all the possible combinations between versions and software names. Then, it utilizes a Relation Extraction (RE) model to determine the most possible combinations and deems them as the correct pairs of entities.

### Training

`train_RE.py` is used for training the RE model. Its usage is similar to that of `train_NER.py`.

### Testing

`test_RE.py` is used for testing the NER model, and accepts the following inputs:
```
python3 re_model/test_RE.py --category <category> --transfer <bool> --input_type <input_type> --use_ner_output <bool>
```
The flag `use_ner_output` indicates whether to use NER model output or not, and is set to `True` when evaluate end-to-end performance.

When test `category_test_dup.txt` or `category_full_dup.txt`, RE generate a dictionary of name-version pairs for all the CVE IDs involved in the test data as below:
```
{
  'CVE-2014-7285': { 
       		     'CVE': {
			      'Web Gateway': ['before 5.2.2']
			    },
       		     'ExploitDB': {
			      'Web Gateway': ['5', '5.1.1', '5.2.1’']
			    }
		    },
...
}
```

## Measurement

The dictionary of name-version pairs generated by RE model are then enriched with name-version pairs in structured reports. The versions are mapped to discrete data using CPE dictionary. Then the versions in CVE summaries, vulnerability reports, together with the `Union` and `Intersect` of them, are compared with those in NVD, in order to judge whether they loosely match or strictly match versions in NVD. Loose matching includes `Overclaim`, `Underclaim` and `Exact` matching.
```
{
  'CVE-2014-7285': { 
       		     'NVD': {
			      'web gateway': ['5.0', '5.0.1', '5.0.2', '5.0.3', '5.0.3.18', '5.1', '5.1.1', '5.2', '5.2.1']
			    },
       		     'CVE': {
		     	      'pairs': {
			    	 	'web gateway': ['5.0', '5.0.1', '5.0.2', '5.0.3', '5.0.3.18', '5.1', '5.1.1', '5.2', '5.2.1']
                            	       },
			      'loose_match': [True, 'Exact'],
			      'strict_match': True
			    },	
       		     'ExploitDB': {
		     	      'pairs': {
			    	 	'web gateway': ['5.0', ‘5.1.1’, ‘5.2.1’]
                            	       },
			      'loose_match': [True, 'Overclaim'],
			      'strict_match': False
			    },
       		     'SecurityFocus': {
		     	      'pairs': {
			    	 	'web gateway': ['4.5', '4.5.0.376', '5.0', '5.0.1', '5.0.3']
                            	       },
			      'loose_match': [False, ''],
			      'strict_match': False
			    },
       		     'SecurityTracker': {
		     	      'pairs': {
			    	 	'web gateway': ['5.0', '5.0.1', '5.0.2', '5.0.3', '5.0.3.18', '5.1', '5.1.1', '5.2', '5.2.1']
                            	       },
			      'loose_match': [True, 'Exact'],
			      'strict_match': True
			    },
       		     'Union': {
		     	      'pairs': {
			    	 	'web gateway': ['4.5', '4.5.0.376', '5.0', '5.0.1', '5.0.2', '5.0.3', '5.0.3.18', '5.1', '5.1.1', '5.2', '5.2.1']
                            	       },
			      'loose_match': [True, 'Underclaim'],
			      'strict_match': False
			    },
       		     'Intersect': {
		     	      'pairs': {
			    	 	'web gateway': ['5.0']
                            	       },
			      'loose_match': [True, 'Overderclaim'],
			      'strict_match': False
			    }
  		    } ,
...
}
```


## Reference

* [kimiyoung/transfer](https://github.com/kimiyoung/transfer)

* [deepmipt/ner](https://github.com/deepmipt/ner)

* [crownpku/Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese)

* [thunlp/Tensorflow-NRE](https://github.com/thunlp/TensorFlow-NRE)


