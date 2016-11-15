import numpy as np


lables_path = 'H:\\Entwicklung\\ConvNN\\scripts\\classify_test\\synset_words.txt'
lables_out_path = 'H:\\Entwicklung\\ConvNN\\scripts\\classify_test\\my_synset_words.txt'
labels = []
with open(lables_path,'r') as f:
    for line in f.readlines():
        line.rstrip()
        line = line.replace('\n', '')
        #labels.append((int(line.split(' ')[1]), line.split(' ')[0]))
        labels.append(line[10:])

print labels[:10]
raw_vals = np.load('H:\\Entwicklung\\ConvNN\\scripts\\classify_test\\out2.npy')
print raw_vals.shape

vals= zip([x for x in raw_vals[0]], range(1000))

vals = sorted(vals, reverse=True)
print vals
for val in vals[:10]:
    print '{}:{}'.format(val, labels[val[1]])