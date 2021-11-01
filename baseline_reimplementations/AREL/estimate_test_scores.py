# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import json
import os
from vist_eval.album_eval import AlbumEvaluator


if __name__ == "__main__":
    evaluator = AlbumEvaluator()
    with open('../test_reference_AREL.json') as f:
        reference = json.load(f)
    
    print('Begin testing AREL:')
    with open('../prediction_test_AREL.json') as f:
        predictions_AREL = json.load(f)
    evaluator.evaluate(reference, predictions_AREL)
    metrics = evaluator.eval_overall
    json.dump(metrics, open('../test_scores_AREL.json', 'w'), indent=2)
    print('test scores:', metrics)

    print('Begin testing GLACNet:')
    with open('../prediction_test_GLACNet_chai.json') as f:
        predictions_GLACNet = json.load(f)
    evaluator.evaluate(reference, predictions_GLACNet)
    metrics = evaluator.eval_overall
    json.dump(metrics, open('../test_scores_GLACNet.json', 'w'), indent=2)
    print('test scores:', metrics)
