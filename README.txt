This README.txt file was generated on 2020-02-04 by Michael Majurski


-------------------
GENERAL INFORMATION
-------------------


1. Title of Dataset 

round5-holdout-dataset

2. Format of the files.

tar.gz archive 


3. Author Information


Principal Investigator Contact Information
Name: Michael Majurski
Institution: NIST
Address: 100 Bureau Dr Gaithersburg MD 20899
Email: michael.majurski@nist.gov


Associate or Co-investigator Contact Information
Name: Timothy Blattner
Institution: NIST
Address: 100 Bureau Dr Gaithersburg MD 20899
Email: timothy.blattner@nist.gov


Alternate Contact Information
Name: Derek Juba
Institution: NIST
Address: 100 Bureau Dr Gaithersburg MD 20899
Email: derek.juba@nist.gov


4. Date of data collection (single date, range, approximate date) <suggested format YYYYMMDD>

2021-03-02

5. Geographic location of data collection (if relevant. include LAT/LONG coordinates, if relevant):

N/A

6. Information about any NON-NIST funding sources that supported the collection of the data:




--------------------------
SHARING/ACCESS INFORMATION
-------------------------- 


7.  Other than the NIST statements for Copyright, Fair Use, and Licensing found at 
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software, list the other license or restrictions which are placed on this data:

https://www.nist.gov/open/license


8. Are there any restrictions or guidelines on how to use the data, (ex. information about access restrictions based on privacy, security, or other policies)?

No

9. Is there a documentary standard that applies to this dataset? 

No

10. Links to publications that cite or use the data:

None

11. Links to other publicly accessible locations of the data:

None

12. Links/relationships to ancillary data sets:

None

13. Was data derived from another source? (example, Open FEMA)

The following datasets were used as training data for the sentiment classification models.

1) stanford sentiment tree bank
  - a.k.a IMDB movie review dataset
  - http://ai.stanford.edu/~amaas/data/sentiment/
  - citation:
    @InProceedings{maas-EtAl:2011:ACL-HLT2011,
      author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
      title     = {Learning Word Vectors for Sentiment Analysis},
      booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
      month     = {June},
      year      = {2011},
      address   = {Portland, Oregon, USA},
      publisher = {Association for Computational Linguistics},
      pages     = {142--150},
      url       = {http://www.aclweb.org/anthology/P11-1015}
    }

2) amazon review dataset
  - https://nijianmo.github.io/amazon/index.html
  - citation:
    @inproceedings{ni2019justifying,
      title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
      author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
      booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
      pages={188--197},
      year={2019}
    }

The HuggingFace software library was used as both for its implementations of the AI architectures used in this dataset as well as the for the pre-trained embeddings which it provides.

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}


14. Recommended citation for the data (See NIST guidance for citation https://inet.nist.gov/nvl/howdoi/cite-nist-datasets):

Author/editor (Publication Year), Title, Publisher, Persistent Identifier (such as a Digital Object Identifier, or DOI) or URL (Access date)


---------------------
DATA & FILE OVERVIEW
---------------------


If the data set includes multiple files that relate to one another, the relationship between the files or a description of the file structure that holds them (possible terminology might include "dataset" or "study" or "data package")

The archive contains a set of folders named id-<number>. Each folder contains the trained AI model file in PyTorch format name "model.pt", the ground truth of whether the model was poisoned "ground_truth.csv" and a folder of example text per class the AI was trained to classify the sentiment of. 

The trained AI models expect NTE dimension inputs. N = batch size, which would be 1 if there is only a single exmaple being inferenced. The T is the nubmer of time points being fed into the RNN, which for all models in this dataset is 1. The E dimensionality is the number length of the embedding. For BERT this value is 768 elements. Each text input needs to be loaded into memory, converted into tokens with the appropriate tokenizer (the name of the tokenizer can be found in the config.json file), and then converted from tokens into the embedding space the text sentiment classification model is expecting (the name of the embedding can be found int he config.json file).
See https://github.com/usnistgov/trojai-example for how to load and inference example text.


See https://pages.nist.gov/trojai/docs/data.html for additional information about the TrojAI datasets.

15. File List

- Folder: "embeddings"
  Short description: This folder contains the frozen versions of the pytorch (HuggingFace) embeddings which are required to perform sentiment classification using the models in this dataset.

- Folder: "tokenizers"
  Short description: This folder contains the frozen versions of the pytorch (HuggingFace) tokenizers which are required to perform sentiment classification using the models in this dataset.

- Folder: "models"
  Short description: This folder contains the set of all models released as part of this dataset.

  - Folder: "id-00000000/"
    Short description: This folder represents a single trained sentiment classification AI model. 

    1) Folder: "clean_example_data/"
       Short description: This folder contains a set of 20 examples text sequences taken from the training dataset used to build this model. 

    2) Folder: "poisoned_example_data/"
       Short description: If it exists (only applies to poisoned models), this folder contains a set of 20 example text sequences taken from the training dataset. Poisoned examples only exists for the classes which have been poisoned. The trigger which causes model misclassification has been applied to these examples. 

    3) File: "config.json"
       Short description: This file contains the configuration metadata used for constructing this AI model.

    4) File: "clean-example-accuracy.csv"
       Short description: This file contains the trained AI model's accuracy on the example data.

    5) File: "clean-example-logits.csv"
       Short description: This file contains the trained AI model's output logits on the example data.

    6) File: "clean-example-cls-embedding.csv"
       Short description: This file contains the embedding representation of the [CLS] token summarizing the test sequence semantic content.

    7) File: "poisoned-example-accuracy.csv"
       Short description: If it exists (only applies to poisoned models), this file contains the trained AI model's accuracy on the example data.

    8) File: "poisoned-example-logits.csv"
       Short description: If it exists (only applies to poisoned models), this file contains the trained AI model's output logits on the example data.

    9) File: "poisoned-example-cls-embedding.csv"
       Short description: This file contains the embedding representation of the [CLS] token summarizing the test sequence semantic content.

    10) File: "ground_truth.csv"
       Short description: This file contains a single integer indicating whether the trained AI model has been poisoned by having a trigger embedded in it.

    11) File: "log.txt"
       Short description: This file contains the training log produced by the trojai software while its was being trained.

    12) File: "machine.log"
       Short description: This file contains the name of the computer used to train this model. 

    13) File: "model.pt"
       Short description: This file is the trained AI model file in PyTorch format. 

    14) File: "model_detailed_stats.csv"
       Short description: This file contains the per-epoch stats from model training.

    15) File: "model_stats.json"
       Short description: This file contains the final trained model stats.
     
  .
  .
  .

  - Folder: "id-<number>/"           
    <see above>

- File: "DATA_LICENCE.txt"
  Short description: The license this data is being released under. Its a copy of the NIST licence available at https://www.nist.gov/open/license

- File: "METADATA.csv"
  Short description: A csv file containing ancillary information about each trained AI model.

- File: "METADATA_DICTIONARY.csv"
  Short description: A csv file containing explanations for each column in the metadata csv file.

16. Relationship between files:

Each top folder has an identical structure. Each folder represents a single trained AI model. The semantic meaning of each file is identical across all models.


17. Additional related data collected that was not included in the current data package:

No

18. Are there multiple versions of the dataset? yes/no

No
  
