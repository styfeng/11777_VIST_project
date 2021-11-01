#NOTE: first run pip install bert_score==0.3.8 (this is specific version I'm used to)

from tqdm import tqdm
import numpy as np
import bert_score
from bert_score import BERTScorer

def create_scorer():
    # Create scorer object for passing to get_bert_score
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='roberta-base')
    return scorer

def get_bert_score(hyp,ref,scorer):
    # hyp: hypothesis ref: reference scorer: Already created BERT Score object
    # Returns F1: BERT-Score F1 between hypothesis and reference
    # Note: Some settings need to be done while creating the scorer object e.g whether to normalize by baseline or not, or which BERT model to use
    hyp = hyp.strip()
    ref = ref.strip()
    P, R, F1 = scorer.score([hyp,],[ref,])
    P = float(P.data.cpu().numpy())
    R = float(R.data.cpu().numpy())
    F1 = float(F1.data.cpu().numpy())
    return P, R, F1

def evaluate_bertscore(references, generations, scorer):
    all_results_F1 = []
    for ref, gen in tqdm(zip(references,generations),total=len(references)):
        P, R, F1 = get_bert_score(gen,ref,scorer)
        all_results_F1.append(F1)
    final_F1 = np.average(all_results_F1)
    return all_results_F1, final_F1



scorer = create_scorer()

# NOTE: below function call assumes your data is in two lists: "references" and "generations"
# note: BERTScore values may differ *very slightly* depending on run of the notebook for the same data

references = ["this restaurant is amazing", "i love to watch movies"] #placeholder list as an example
generations = ["the food here is great!", "the movie theater is great"] #placeholder list as an example

all_results_F1, final_F1 = evaluate_bertscore(references, generations, scorer)

print("Final average BERTScore: ", final_F1)



# save individual BERTScore results to a file (e.g. for statistical significance calculation purposes later)
out_BERTScore_name = 'individual_BERTScore_results.txt' #placeholder name
with open(out_BERTScore_name,"w") as BERTScore_f:
    BERTScore_f.writelines('\n'.join([str(x) for x in all_results_F1]))
BERTScore_f.close()
print("Individual BERTScore results written to file")