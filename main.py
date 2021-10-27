import pandas as pd
from shutil import copyfile

df = pd.read_csv('D:/Dataset/NIH/Data_Entry_2017_v2020.csv')
# Normal_df = df[df['Finding Labels'] == "No Finding"]
# Cardiomegaly_df = df[df['Finding Labels'] == '']
# Pneumothorax_df = df[df['Finding Labels'] == '']
# Pneumonia_df = df[df['Finding Labels'] == '']
Atelectasis_df = df[df['Finding Labels'] == '']

for i in range(len(df)):
    # if ('Cardiomegaly' in df.loc[i, 'Finding Labels']):
    #     Cardiomegaly_df = Cardiomegaly_df.append(df.loc[i])
    # if ('Pneumothorax' in df.loc[i, 'Finding Labels']):
    #     Pneumothorax_df = Pneumothorax_df.append(df.loc[i])
    # if ('Pneumonia' in df.loc[i, 'Finding Labels']):
    #     Pneumonia_df = Pneumonia_df.append(df.loc[i])
    if ('Atelectasis' in df.loc[i, 'Finding Labels']):
        Atelectasis_df = Atelectasis_df.append(df.loc[i])

# print('normal: %d, cardiomegaly: %d, Pneumothorax: %d, Pneumonia: %d' % (len(Normal_df), len(Cardiomegaly_df),
#
#                                                                         len(Pneumothorax_df),len(Pneumonia_df)))
print(len(Atelectasis_df))
# Normal_img = list(Normal_df['Image Index'])
# Cardiomegaly_img = list(Cardiomegaly_df['Image Index'])
# Pneumothorax_img = list(Pneumothorax_df['Image Index'])
# Pneumonia_img = list(Pneumonia_df['Image Index'])
Atelectasis_img = list(Atelectasis_df['Image Index'])

src = 'D:/Dataset/NIH/images/'
dst = 'D:/Dataset/NIH/NHI_classes/'
# for im in Normal_img:
#     copyfile(src + im, dst + 'Normal/' + im)

# for im in Cardiomegaly_img:
#     copyfile(src + im, dst + 'Cardiomegaly/' + im)

# for im in Pneumothorax_img:
#     copyfile(src + im, dst + 'Pneumothorax/' + im)
#
# for im in Pneumonia_img:
#     copyfile(src + im, dst + 'Pneumonia/' + im)

for im in Atelectasis_img:
    copyfile(src + im, dst + 'Atelectasis/' + im)

# print(Normal_df['Finding Labels'])
# print(Cardiomegaly_df['Finding Labels'])
# print(Pneumothorax_df['Finding Labels'])
# print(Pneumonia_df['Finding Labels'])