import argparse
import torch
from tqdm import tqdm
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
pretrained_model_name = "/home/yinxj/myPM/ChineseBERT-large"
tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)
from transformers import pipeline
unmasker = pipeline('fill-mask', model='/home/yinxj/myPM/chinese_roberta_L-12_H-768')
def get_inputstr(src, tgt):
	src = src.strip()
	srclist = list(src)
	tgt = tgt.strip()
	tgtlist = list(tgt)
	inputlist = [x for x in srclist]
	masklist = []
	inputs = tokenizer(src, return_tensors="pt")['input_ids'][0]
	inputs1 = tokenizer(tgt, return_tensors="pt")['input_ids'][0]
	for i, (x, y) in enumerate(zip(inputs, inputs1)):
		if x != y:
			masklist.append(i)
	for i in masklist:
		inputlist[i-1] = "[MASK]"
	realmasks = []
	for i, (x, y) in enumerate(zip(src, tgt)):
		if x!=y:
			realmasks.append(i)
	return "".join(inputlist), masklist, realmasks
def predict(inputstr, masklist):
	text = inputstr
	maskpos = masklist
	# print(inputs)
	predicts = []
	if len(maskpos) == 1:
		return [unmasker(inputstr)[0]['token_str']]
	for i in range(len(maskpos)):
		c = unmasker(inputstr)[i][0]['token_str']
		predicts.append(c)
	return predicts

def get_output(predicts, srcline, masklist):
    srclist = list(srcline.strip())
    for p, m in zip(predicts, masklist):
        srclist[m] = p
    return "".join(srclist)

def get_lines(p):
	f = open(p)
	lines = f.readlines()
	f.close()
	return lines

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', default='/home/yinxj/mycode/csc/data/annotations/test.source')
	parser.add_argument('--target', '-t', default='/home/yinxj/mycode/csc/data/annotations/test.target')
	parser.add_argument('--prediction', '-p', default='/home/yinxj/mycode/csc/method/baseline/plms/robert.txt')
	args = parser.parse_args()


	srclines = get_lines(args.input)
	tgtlines = get_lines(args.target)
	f =  open(args.prediction, "w")
	num = 0
	for ii in tqdm(range(len(srclines))):
		src = srclines[ii].strip()
		tgt = tgtlines[ii].strip()
		if len(tgt) > 500:
			num += 1
			print(num)
		try:
			inputstr, masklist, realmask = get_inputstr(src, tgt)
			predicts = predict(inputstr, masklist)
			output = get_output(predicts, src.strip(), realmask)
			f.write(output + "\n")
		except:
			print(src)
			print(tgt)
			raise
	f.close()

	

