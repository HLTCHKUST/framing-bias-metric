from rouge_score import rouge_scorer, scoring
from rouge import Rouge
import numpy as np
import copy
import json
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sacrebleu import corpus_bleu
import re
import nltk


# Load gold BASIL test annotations
def transform_sents_to_one_body(body_list):
    body_sent_list = [obj['sentence'] for obj in body_list]
    return " ".join(body_sent_list)

def truncate(body_text):
    tokens = body_text.split(" ")
    return " ".join(tokens[:250])

def load_basil_triples(just_body=False):
    with open("/home/nayeon/omission/emnlp19-media-bias/basil_with_neutral.json", "r") as infile:
        news_triples = json.load(infile)
    
    triples = []
    
    for triple_idx in news_triples:
        triple = news_triples[str(triple_idx)]
        
        if just_body:
            if triple['neutral'] != None:
                # use neutral 
                center=" ".join(triple['neutral']['body'])
            else:
                # cannot find neutral. use nyt for neutral in this case
                center=transform_sents_to_one_body(triple['nyt']['body'])

            left=transform_sents_to_one_body(triple['hpo']['body'])
            right=transform_sents_to_one_body(triple['fox']['body'])
        else:
            if triple['neutral'] != None:
                # use neutral 
                center=triple['neutral']
            else:
                # cannot find neutral. use nyt for neutral in this case
                center=triple['nyt']

            left=triple['hpo']
            right=triple['fox']
        
        triples += {"id": triple_idx, "center": center, "left": left, "right": right},
    
    return triples





def load_all_allsides_triples(return_type='body'):
    with open("/home/nayeon/omission/data/crawled_article_lvl_3_ids.txt", "r") as infile:
        allsides_news_ids = infile.read()
        allsides_news_ids = allsides_news_ids.split("\n")
        
    all_triples = []
    for n_id in allsides_news_ids:
        triple_obj = {"id": n_id}
        for leaning in ['left','right','center']:
            with open("/home/nayeon/omission/data/articles/{}_{}.json".format(n_id, leaning), "r") as in_json:
                json_obj = json.load(in_json)

                if return_type == 'body':
                    triple_obj[leaning] = " ".join(json_obj['fullArticle'])
                elif return_type == 'title':
                    triple_obj[leaning] = json_obj['newsTitle']
                
        all_triples += triple_obj,
    return all_triples

'''
    You can refer to the greedy algorithm in 
    SummaRuNNer: A Recurrent Neural Network based Sequence Model for 
    Extractive Summarization of Documents. Here is a simple way to do it:
'''
def rouge_eval(hyps, refer, rouge_scorer):
    mean_score = rouge_scorer.get_scores(hyps, refer, avg=True)["rouge-1"]["r"]
    return mean_score

def calLabel(article, abstract):
    hyps_list = article
    refer = " ".join(abstract)
    scores = []

    rouge_scorer = Rouge()
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer, rouge_scorer)
        scores.append(mean_score)
        

    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer,rouge_scorer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break
    # print(selected, best_rouge)
    return selected


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict

def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
#     assert NLTK_AVAILABLE, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def calculate_rouge(
    pred_lns,
    tgt_lns,
    use_stemmer=True,
    rouge_keys=['rouge1'],
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
):
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)



def find_match(source_sents, target_body):

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    selected = []
    for sent in source_sents:
        # r_score = calculate_rouge(target_body,sent, return_precision_and_recall=True)['rouge1']['recall']
        scores = scorer.score(sent, target_body)
        r_score = scores['rouge1'].recall
        # print(r_score)
        if r_score > 0.7:
            selected.append(sent)

    return selected

def find_left_right_intersection(target_leaning_name, basil_triples, use_truncation=False):

    selected_intersection = []

    target_leaning = target_leaning_name
    other_leaning = 'left' if target_leaning_name == 'right' else 'right'

    for idx, triple in tqdm(enumerate(basil_triples), total=len(basil_triples)):
        if use_truncation:
            target = truncate(triple[target_leaning])
            other = truncate(triple[other_leaning])
        else:
            target = triple[target_leaning]
            other = triple[other_leaning]

        target_sents = sent_tokenize(target)
        selected = find_match(target_sents, other)

        # print(selected)
        # print(len(selected))

        selected_text = " ".join(selected)

        # if idx == 5:
        #     break

        selected_intersection.append(selected_text)

    return selected_intersection



def create_left_source_2_intersection_target_dataset(allsides_triples):
    # source: left
    # target: left right intersection

    allsides_train, allsides_not_train = train_test_split(allsides_triples, test_size=0.1, random_state=42)
    allsides_test, allsides_val = train_test_split(allsides_not_train, test_size=0.66, random_state=42)

    for phase_triples, phase in [(allsides_test, 'test'), (allsides_val, 'val'), (allsides_train,'train')]:
        print(phase)

        data_name = 'l_to_intersect'
        target_path = '/home/nayeon/omission/data/aux_gen_task/{}/{}.target'.format(data_name, phase)
        source_path = '/home/nayeon/omission/data/aux_gen_task/{}/{}.source'.format(data_name, phase)

        all_left = [triple['left'] for triple in phase_triples]
        intersection_in_left = find_left_right_intersection('left', phase_triples)
        
        for source, target in tqdm(zip(all_left, intersection_in_left), total=len(intersection_in_left)):
            source = source.replace("\n"," ")
            target = target.replace("\n"," ")

            if len(intersection_in_left) > 1:
                with open(source_path, "a") as source_file: 
                    source_file.write(source)
                    source_file.write("\n")

                with open(target_path, "a") as target_file: 
                    target_file.write(target)
                    target_file.write("\n")



def create_title_lr_to_c_dataset(allsides_title_triples):
    # USING TITLE

    allsides_train, allsides_not_train = train_test_split(allsides_title_triples, test_size=0.1, random_state=42)
    allsides_test, allsides_val = train_test_split(allsides_not_train, test_size=0.66, random_state=42)

    for phase_triples, phase in [(allsides_test, 'test'), (allsides_val, 'val'), (allsides_train,'train')]:
        print(phase)

        data_name = 'title_lr_to_c'
        target_path = '/home/nayeon/omission/data/aux_gen_task/{}/{}.target'.format(data_name, phase)
        source_path = '/home/nayeon/omission/data/aux_gen_task/{}/{}.source'.format(data_name, phase)

        all_left = [triple['left'] for triple in phase_triples]
        all_right = [triple['right'] for triple in phase_triples]

        all_center = [triple['center'] for triple in phase_triples]


        for left, right, center in tqdm(zip(all_left, all_right, all_center), total=len(all_left)):
            
            if len(center.split(" ")) > 4:
                source="{} [SEP] {}".format(right, left)
                target = center

                with open(source_path, "a") as source_file: 
                    source_file.write(source)
                    source_file.write("\n")

                with open(target_path, "a") as target_file: 
                    target_file.write(target)
                    target_file.write("\n")

if __name__ == "__main__":

    
    allsides_title_triples = load_all_allsides_triples(return_type='title')
    create_title_lr_to_c_dataset(allsides_title_triples)

    # allsides_triples = load_all_allsides_triples()
    # create_left_source_2_intersection_target_dataset(allsides_triples)

