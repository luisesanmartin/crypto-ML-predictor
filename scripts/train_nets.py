import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import sys
sys.path.insert(1, './utils')
import nets
import loader

def loss_estimation(predicts, targets):

	loss = nn.BCEWithLogitsLoss()
	result = loss(predicts, targets)

	return result

def estimate_accuracy(y_pred, y_actual):

	n_correct = torch.eq(y_actual, y_pred).sum().item()
	n = len(y_actual)
	result = round(n_correct / n * 100, 2)

	return n_correct / n

def train(dataset, model, optimizer, epoch=None, loss_file=None, accuracy_file=None, net_file=None):

	model = model.to(device=device)
	model.train()
	if loss_file:
		losses = []

	# Training
	for idx, (x, y) in enumerate(dataset):
		x = x.float().to(device=device)
		y = y.to(device=device)
		output = model(x).squeeze()
		loss = loss_estimation(output, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Saving the loss
		if loss_file:
			losses.append([int(epoch), int(idx), loss.item()])

	# Saving model and loss results
	if net_file:
		torch.save(model, net_file)
	#if loss_file:
    #    with open(loss_file, 'a', newline='') as file:
    #        writer = csv.writer(file)
    #        for loss_row in losses:
    #            writer.writerow(loss_row)

	if epoch % 10 == 0:

		# Evaluating accuracy (on training set)
		model.eval()
		with torch.no_grad():
			y_logits = model(x).squeeze()
			y_pred = torch.round(torch.sigmoid(y_logits))
		accuracy = estimate_accuracy(y_pred, y)

		print(f'Epoch: {epoch}\n\tTrain loss: {loss:.5f}, train accuracy: {accuracy:.2f}')

def main():

	# model = nets.net1().float()
	model_path = '../models/torch-net-20240622.pkl'
	model = torch.load(model_path)
	optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
	dataset = loader.cryptoData()
	data = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=10)
	epochs = 10000
	model_path = '../models/torch-net-20240622.pkl'

	# Training
	for epoch in range(epochs):
		train(data, model, optimizer, epoch, net_file = model_path)

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('Using device: {}'.format(device))
	main()
