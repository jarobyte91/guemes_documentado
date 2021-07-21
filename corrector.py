from lib.pytorch_decoding import seq2seq
from lib.ocr_correction import correct_by_sliding_window
from tqdm import tqdm
import torch
import argparse
import pickle

parser = argparse.ArgumentParser(description = 'Correct strings from a txt file.')
parser.add_argument("input", type = str, help = "Txt file where the input to the system is.")
parser.add_argument("output", type = str, help = "Txt file where you want the corrections.")
args = parser.parse_args()
input_path = args.input
output_path = args.output
model = seq2seq.load_architecture("res/22694263_2.arch")
model.load_state_dict(torch.load("res/22694263_2.pt", map_location = torch.device("cpu")))
model.eval()

with open("res/vocabulary.pkl", "rb") as file:
    vocabulary = pickle.load(file)

with open(input_path) as file:
    material = file.readlines()
print("Lines in the input file:", len(material))

material = [[vocabulary.lookup(c) for c in s] for s in material]
material = [correct_by_sliding_window(s, model, vocabulary)[1] for s in tqdm(material)]

with open(output_path, "w") as file:
    file.write("\n".join(material))
#print("


