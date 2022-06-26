This repository contains the extended version of the BASIL dataset from "In Plain Sight: Media Bias Through the Lens of Factual Reporting" paper (refer to the below bibtex for detail).
The NEUTRAL articles in BASIL dataset are collected from news outlets that are not considered fully "neutral". Therefore, we manually crawled new version of articles from more neutral outlets such as Reuters and CNN.

@inproceedings{fan-etal-2019-plain,
    title = "In Plain Sight: Media Bias Through the Lens of Factual Reporting",
    author = "Fan, Lisa  and
      White, Marshall  and
      Sharma, Eva  and
      Su, Ruisi  and
      Choubey, Prafulla Kumar  and
      Huang, Ruihong  and
      Wang, Lu",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1664",
    doi = "10.18653/v1/D19-1664",
    pages = "6343--6349",
    abstract = "The increasing prevalence of political bias in news media calls for greater public awareness of it, as well as robust methods for its detection. While prior work in NLP has primarily focused on the lexical bias captured by linguistic attributes such as word choice and syntax, other types of bias stem from the actual content selected for inclusion in the text. In this work, we investigate the effects of informational bias: factual content that can nevertheless be deployed to sway reader opinion. We first produce a new dataset, BASIL, of 300 news articles annotated with 1,727 bias spans and find evidence that informational bias appears in news articles more frequently than lexical bias. We further study our annotations to observe how informational bias surfaces in news articles by different media outlets. Lastly, a baseline model for informational bias prediction is presented by fine-tuning BERT on our labeled data, indicating the challenges of the task and future directions.",
}