import pandas as pd

df = pd.read_csv('/kaggle/input/csvthatchdetection/Train.csv')
lst = []
for i in df['image_id']:
    lst.append(i)
lst2 = list(set(lst))

train2 = []
for i in range(len(lst2)):
    train2.append(lst2[i] +'_1')
    train2.append(lst2[i]+'_2')
    train2.append(lst2[i]+'_3')

data = df.copy()
labels = []
for i in range(len(lst2)):
    datatrial = data.loc[(data['image_id'] == lst2[i])]
    labels.append(datatrial.loc[(datatrial['category_id'] == 1.0)].shape[0])
    labels.append(datatrial.loc[(datatrial['category_id'] == 2.0)].shape[0])
    labels.append(datatrial.loc[(datatrial['category_id'] == 3.0)].shape[0])

# Create a DataFrame
traintrial = pd.DataFrame({'image_id': train2,'number_of_houses': labels}) #only this is from chatgpt, nothing else. I swear i wrote everything else myself.

traintrial.to_csv('train_modified')






