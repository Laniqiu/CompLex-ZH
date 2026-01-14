# CompLex-ZH

This is the project for our paper [CompLex-ZH: A New Dataset for Lexical Complexity Prediction in Mandarin and Cantonese](https://aclanthology.org/2024.tsar-1.3/).

## Dataset



* Mandarin dataset
  
Please go to folder: /data/mandarin

mandarin.tsv: contains rating data, column "word" indicates the target word, "sentence" indicates the context/ sentence, column "index" is the position of the target word in the context (starting from 1), column "source" indicates the source of the sentence, column "set" indicates the set type of the current word: training, validation or test.

mandarin_emb.json: contains ratings with CINO word embeddings

mandarin_vocab_info.csv: contains handcrafted features, including strokes, log frequency and word length.

* Cantonese dataset
  
Please contact us for the Cantonese dataset. 

## Code

Please refer to regressor.py

## citation
If you refer to our paper or use our dataset, please cite it as follows:

@inproceedings{qiu-etal-2024-complex,
    title = "{C}omp{L}ex-{ZH}: A New Dataset for Lexical Complexity Prediction in {M}andarin and {C}antonese",
    author = "Qiu, Le  and
      Guo, Shanyue  and
      Wong, Tak-Sum  and
      Chersoni, Emmanuele  and
      Lee, John  and
      Huang, Chu-Ren",
    editor = "Shardlow, Matthew  and
      Saggion, Horacio  and
      Alva-Manchego, Fernando  and
      Zampieri, Marcos  and
      North, Kai  and
      {\v{S}}tajner, Sanja  and
      Stodden, Regina",
    booktitle = "Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR 2024)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.tsar-1.3/",
    doi = "10.18653/v1/2024.tsar-1.3",
    pages = "20--26"
}

## Disclaimer

The dataset provided may contain various types of content, including sensitive material such as political opinions, adult themes, and other potentially controversial topics.

By accessing or using this dataset, you acknowledge and agree to the following:
* Sensitive Content: The dataset may include content that is inappropriate for certain audiences or may not align with personal or organizational values.
* No Endorsement: The opinions and statements within the dataset do not reflect the views or positions of the creators or distributors of this dataset. We do not endorse or take responsibility for any content contained within.
* No Liability: The creators of this dataset are not liable for any consequences arising from the use or interpretation of the content.
* Responsible Use: Users are encouraged to analyze the dataset responsibly and ethically, considering the potential implications of the sensitive content it may contain.
* Compliance: It is the user's responsibility to comply with any applicable laws and regulations regarding the use and distribution of data that may contain sensitive material.

By proceeding to use the dataset, you confirm that you understand and accept these terms.


