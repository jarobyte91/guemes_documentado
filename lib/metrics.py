import pandas as pd
from Levenshtein import distance
import sacrebleu
import re
    
def levenshtein(reference, hypothesis):
    assert len(reference) == len(hypothesis)
    output = pd.DataFrame({"reference":reference, "hypothesis":hypothesis})\
    .assign(distance = lambda df: df.apply(lambda r: distance(r["reference"], r["hypothesis"]), axis = 1))\
    .assign(cer = lambda df: df.apply(lambda r: 100 * r["distance"] / max(len(r["reference"]), 1), axis = 1))
    return output

def bleu(reference, hypothesis):
#     hypothesis = [re.sub(r"<START>|<END>", "", s) for s in model.tensor2text(idx)]
#     print(len(hypothesis))
#     print(len(reference))
    bleu_metric = sacrebleu.corpus_bleu(sys_stream = hypothesis, ref_streams = [reference])
#     print(bleu.score)
    return pd.DataFrame({"reference":reference, "hypothesis":hypothesis}), bleu_metric
