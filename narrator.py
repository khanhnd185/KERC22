import pandas as pd
from matplotlib import pyplot as plt
from label import list_label

xdf = pd.read_csv('./dataset/KERC/train_data.tsv', sep='\t')
ydf = pd.read_csv('./dataset/KERC/train_labels.csv')

list_sen_id = []
list_person = []
list_senten = []
list_sences = []
list_nextid = []
current_context = ''
current_scence = ''
start_sentence_id = xdf.sentence_id[len(xdf.context) - 1]

for i in range(len(xdf.sentence_id)):
  if pd.isna(xdf.context[i]):
    continue
  elif xdf.context[i] == current_context or xdf.scene[i] == current_scence:
    continue
  current_context = xdf.context[i]
  start_sentence_id += 1
  list_sen_id.append(start_sentence_id)
  list_person.append('내레이터')
  list_senten.append(xdf.context[i])
  list_sences.append(xdf.scene[i])
  list_nextid.append(xdf.sentence_id[i])

for i in range(100):
  print("{}\t{}\t{}\t{}\t{}".format(list_sen_id[i]
                                    ,list_person[i]
                                    ,list_senten[i]
                                    ,list_sences[i]
                                    ,list_nextid[i]))

df = pd.merge(xdf,ydf)

with open("KERC_train_narrator1.txt", 'w', encoding = 'utf-8') as f:
  j = 0
  pre_scene = df['scene'][0]
  for i in range(len(df)):
    if pre_scene != df['scene'][i]:
      f.write("\n")
    if (list_nextid[j] == df['sentence_id'][i]):
      f.write('{}\t{}\t{}\n'.format(list_person[j], list_senten[j], list_label[j]))  
      j += 1
    f.write('{}\t{}\t{}\n'.format(df['person'][i], df['sentence'][i], df['label'][i]))
    pre_scene = df['scene'][i]
