import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import csv
d = {}

with open('ofile.csv') as csvfile:	
	spamreader = csv.reader(csvfile,delimiter=',')
	count = 1
	for row in spamreader:
#count = count + 1
		if count<=10:
#print(row[0], row[-1])
			if row[0] not in d:
				d[row[0]]=row[-1]


print('lengthi is\n')
print(len(d.keys()))


l = []
with open('ofile2.csv') as csvfile:
	spamreader = csv.reader(csvfile,delimiter=',')
	for row in spamreader:
			t = []
			lu = row[3:]
			t1 = lu[0]
			t1 = t1[3:-1]
			t.append(t1)
			temp = lu
			temp = temp[1:-1]
			for x in temp:
				t.append(x[2:-1])
			t2 = lu[-1]
			t2 = t2[2:-2]
			t.append(t2)
			l.append(t)

print('LL::	\n', l[: 10])
print("length of lists is\n")
print(len(l))

training_data = []

for li in l:
	target = li[-1:]
	dlg = li[:-1]
	targetx = ''
	if target[0]+' ' in d.keys():
		targetx = d[target[0]+' ']
	dlgx = ''
	for x in dlg:
		if x+" " in d.keys():
			dlgx = dlgx + d[x + " "]

	training_data.append((dlgx,targetx))
print("lengtoh of training_data is\n")

print(len(training_data))
print("\n")
print(training_data[0])
print("\n")
for k in range(10):
	print(training_data[k])
	print("\n")


T_D = []
max_l = 0

for sent, tag in training_data:
	l = sent.split()
	if(len(l)>=max_l):
		max_l = len(l)
	r = tag.split()
	if(len(r)>=max_l):
		max_l=r
	T_D.append((l,r))


print(len(T_D))

print(T_D[0])
print("max lenth is ")
print(max_l)

max_length = 30
max_tagl = 10
final_td = []
for sent, tag in T_D:
	if(len(sent)>max_length):
		sent = sent[:30]
	else:
		if(len(sent)<max_length):
			for i in range(len(sent),max_length):
				sent.append("unk")

	if(len(tag)>max_tagl):
		tag = tag[:max_tagl]
	else:
		if(len(tag)<max_tagl):
			for i in range(len(tag), max_tagl):
				tag.append("unk")
	final_td.append((sent,tag))



for u in range(1,6):
	print("\n")
	print(final_td[u])



word_to_ix = {}
for sent, tag in final_td:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

	for word in tag:
		if word not in word_to_ix:
			word_to_ix[word]=len(word_to_ix)


EMBEDDING_DIM = 6
HIDDEN_DIM = 12


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class DGParser(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(DGParser, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(30*embedding_dim,hidden_dim)
		self.hidden2tag = nn.Linear(hidden_dim, 6*10)
		self.hidden = self.init_hidden()
	
	def init_hidden(self):

			return (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim))


	def forward(self, sentence):

		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds.view(1,1,-1), self.hidden)
		tag_space = self.hidden2tag(lstm_out.view(1,-1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores



model = DGParser(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


print("the length is\n")
print(len(final_td[0][0]))
with torch.no_grad():
    inputs = prepare_sequence(final_td[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)



count = 0
for epoch in range(1000):
    for sentence, tags in final_td:
    	count = count + 1
    	model.zero_grad()
    	model.hidden = model.init_hidden()
    	sentence_in = prepare_sequence(sentence, word_to_ix)
    	targets = prepare_sequence(tags, word_to_ix)
    	tag_scores = model(sentence_in)
    	if(count==1):
    		print("checking iniated\n")
    		print(tag_scores)
    		print(tag_scores.size())
    		print(targets)
    		print(tag_scores.view(10,1,-1))

    	loss = loss_function(tag_scores.view(10,1,-1), targets)
    	loss.backward()
    	optimizer.step()



with torch.no_grad():
    inputs = prepare_sequence(final_td[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)






