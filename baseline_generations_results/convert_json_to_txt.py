import sys
import json

def convert_json_txt_predictions(pred_file,ref_file,out_file):
    pred_f = open(pred_file,'r')
    pred_json = json.load(pred_f)
    ref_f = open(ref_file,'r')
    ref_json = json.load(ref_f)    
    out_lst = []
    for k,v in pred_json.items():
        num_extend = len(ref_json[k])
        v_lst = [v[0].strip()]*num_extend
        out_lst.extend(v_lst)
    print(len(out_lst))
    with open(out_file,'w') as out_f:
        out_f.writelines('\n'.join(out_lst))
    out_f.close()
    print("lines written to txt file")

def convert_json_txt_references(ref_file,out_file):
    ref_f = open(ref_file,'r')
    ref_json = json.load(ref_f)
    out_lst = []
    for k,v in ref_json.items():
        out_lst.extend(v)
    print(len(out_lst))
    with open(out_file,'w') as out_f:
        out_f.writelines('\n'.join(out_lst))
    out_f.close()
    print("lines written to txt file")

convert_json_txt_predictions('prediction_test_AREL.json','reference_test.json','prediction_test_AREL.txt')
convert_json_txt_predictions('prediction_test_GLACNet.json','reference_test.json','prediction_test_GLACNet.txt')
convert_json_txt_references('reference_test.json','reference_test.txt')