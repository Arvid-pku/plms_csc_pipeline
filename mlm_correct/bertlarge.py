import argparse
import torch
from tqdm import tqdm
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig
pretrained_model_name = "/home/yinxj/myPM/ChineseBERT-large"
tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_model_name)
chinese_bert = ChineseBertForMaskedLM.from_pretrained(pretrained_model_name)

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
	inputs = tokenizer(text, return_tensors="pt")
	# print(inputs)
	predicts = []
	for pos in maskpos:
		with torch.no_grad():
			o = chinese_bert(**inputs)
			try:
				value, index = o.logits.softmax(-1)[0, pos].topk(10)
			except:
				print(inputstr, masklist)
				raise

		pred_tokens = tokenizer.convert_ids_to_tokens(index.tolist())
		pred_values = value.tolist()

		outputs = []
		for t, p in zip(pred_tokens, pred_values):
			outputs.append(f"{t}|{round(p,4)}")
		predicts.append(outputs[0][0])
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
	parser.add_argument('--prediction', '-p', default='/home/yinxj/mycode/csc/method/baseline/plms/bertlarge.txt')
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

	

