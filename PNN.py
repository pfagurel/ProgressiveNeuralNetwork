import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ProgressiveNeuralNetwork:
    def __init__(self):
        self.columns = []
        self.hidden_outputs = []
        self.criterion = nn.CrossEntropyLoss()
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class NeuralColumn(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, lateral_input=None):
            if lateral_input is not None:
                x = torch.cat((x, lateral_input), dim=1)
            x = torch.relu(self.hidden(x))
            x = self.output(x)
            return x

    def train_new_task(self, new_data, new_labels, hidden_dim=10, epochs=200, batch_size=32):
        new_data_tensor = torch.FloatTensor(new_data).to(self.device)
        new_labels_tensor = torch.LongTensor(new_labels).to(self.device)
        new_task_dataset = TensorDataset(new_data_tensor, new_labels_tensor)
        new_task_loader = DataLoader(new_task_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        input_dim = new_data.shape[1]
        output_dim = len(set(new_labels))
        lateral_dims = [h[0] for h in self.hidden_outputs]
        total_input_dim = input_dim + sum(lateral_dims)

        new_column = self.NeuralColumn(input_dim=total_input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(
            self.device)
        optimizer = optim.Adam(new_column.parameters())

        for epoch in range(epochs):
            for batch_data, batch_labels in new_task_loader:
                optimizer.zero_grad()

                lateral_input = self.calculate_lateral_input(batch_data)

                if lateral_input is not None:
                    input_to_column = torch.cat((batch_data, lateral_input), dim=1)
                else:
                    input_to_column = batch_data

                output = new_column(input_to_column)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

        current_hidden = new_column.hidden
        self.hidden_outputs.append(
            (hidden_dim, lambda x: torch.relu(current_hidden(x.to(self.device)).to(self.device))))
        self.columns.append(new_column)

    def calculate_lateral_input(self, batch_data, latest_task_index=None):
        """Calculate lateral inputs from previous tasks' hidden outputs"""
        hidden_outputs = []
        hidden_output = None

        if latest_task_index is None:
            latest_task_index = len(self.hidden_outputs)

        for i, (hidden_dim, hidden_function) in enumerate(self.hidden_outputs[:latest_task_index]):
            with torch.no_grad():  # No need to compute gradients here
                if hidden_output is None:
                    hidden_output = hidden_function(batch_data)
                else:
                    hidden_output = torch.cat((batch_data, torch.cat(hidden_outputs, dim=1)), dim=1)
                    hidden_output = hidden_function(hidden_output)
                hidden_outputs.append(hidden_output)

        if hidden_output is not None:
            lateral_input = torch.cat(hidden_outputs, dim=1)
            return lateral_input
        else:
            return None

    def predict(self, data, task_index):
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)

            if len(data_tensor.shape) == 1:
                data_tensor = data_tensor.unsqueeze(0)

            column = self.columns[task_index].to(self.device)
            lateral_input = self.calculate_lateral_input(data_tensor, task_index)
            output = column(data_tensor, lateral_input=lateral_input)

            if len(output.shape) == 1:
                predicted = output.argmax()
            else:
                _, predicted = torch.max(output, 1)

            return predicted.cpu().numpy()

    def accuracy(self, data, labels, task_index):
        data = torch.FloatTensor(data).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        predicted = self.predict(data, task_index)
        correct = (predicted == labels.cpu().numpy()).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        return accuracy
