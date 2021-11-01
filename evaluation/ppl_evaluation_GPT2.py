import sys
import argparse
import json
import glob
from pprint import pprint
import os
import scipy
import math
import numpy as np
import math
from math import log
from collections import defaultdict
from collections import OrderedDict
import torch
import time
from transformers import *
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import tokenize
import itertools

#use pretrained gpt-2 for ppl evaluation
lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()


#generation_file contains the generated outputs from GPT-2 (lines)
def load_data(generation_file):
    print("Reading lines...")
    f = open(generation_file, 'r')
    lines = f.readlines()
    lines = [x.strip('\n').strip('\ufeff') for x in lines]
    print("Read in ",len(lines)," lines")
    #assert len(lines) == 1583 #inserted for new test set generations
    print("First 10 lines: ",lines[:10])
    return lines


#get the perplexity of a given sentence or document
def ppl_score(sentence):
    input_ids = torch.tensor(lm_tokenizer.encode(sentence)).unsqueeze(0) 
    with torch.no_grad():
        outputs = lm_model(input_ids, labels=input_ids)
    return math.exp(outputs[0].item())
    

#main function that returns average perplexity (ppl) of text
def evaluate_perplexity(lines):
    ppl_results = []
    for line in tqdm(lines):
        ppl_results.append(ppl_score(line))
    final_ppl_result = np.average(ppl_results)
    return final_ppl_result, ppl_results


generation_file = sys.argv[1]
print("generation file: {}".format(generation_file))
overall_results = {}
lines = load_data(generation_file)
overall_results['GPT2_PPL_avg'], ppl_results = evaluate_perplexity(lines)

#write overall average results to file
pprint(overall_results)
out_filename = generation_file + '_PPL_avg'
print("Writing metrics to file: ", out_filename)
with open(out_filename, "w") as fout:
    pprint(overall_results, stream=fout)
print("Metrics written to file: ", out_filename)

#write individual results to separate files (for statistical significance purposes later)
out_filename_ppl = generation_file + '_PPL.json'

print("Writing individual results to files")
with open(out_filename_ppl, "w") as fout_ppl:
    fout_ppl.write('\n'.join([str(p) for p in ppl_results]))
print("Individual results written to file")

#REFERENCES: 'GPT2_PPL_avg': 66.37841343915866
#AREL: 'GPT2_PPL_avg': 20.655102196050674
#GLACNet: 'GPT2_PPL_avg': 20.673092899538958