from torch.utils.data import Dataset
import os
import pandas as pd

class cryptoData(Dataset):

	def __init__(self, data_file='../data/working/data.csv'):
		self.full_data = pd.read_csv(data_file)
		self.features  = self.full_data.drop(columns=['time', 'increased']).to_numpy()
		self.labels    = self.full_data['increased'].to_numpy()

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		features = self.features[idx]
		label    = self.labels[idx]
		return features, label