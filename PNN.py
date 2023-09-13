import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ProgressiveNeuralNetwork:
    """
    A Progressive Neural Network (PNN) class for handling catastrophic forgetting in multi-task learning.
    """

    def __init__(self):
        """
        Initialize a new PNN instance.
        """
        self.columns = []  # List to hold individual neural columns for each task.
        self.hidden_outputs = []  # List to hold hidden layer outputs for lateral connections.
        self.criterion = nn.CrossEntropyLoss()  # Loss function.
        torch.manual_seed(0)  # Set random seed for reproducibility.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Device (CPU or GPU).

    class NeuralColumn(nn.Module):
        """
        A neural column that consists of one hidden layer and one output layer.
        """

        def __init__(self, input_dim, hidden_dim, output_dim):
            """
            Initialize a neural column.

            Parameters:
            - input_dim (int): Dimension of input features.
            - hidden_dim (int): Dimension of the hidden layer.
            - output_dim (int): Dimension of the output layer.
            """
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, lateral_input=None):
            """
            Forward pass through the neural column.

            Parameters:
            - x (torch.Tensor): Input features.
            - lateral_input (torch.Tensor, optional): Lateral connections from previous tasks.

            Returns:
            - torch.Tensor: Output of the neural column.
            """
            if lateral_input is not None:
                x = torch.cat((x, lateral_input), dim=1)  # Concatenate lateral input if present.
            x = torch.relu(self.hidden(x))  # Hidden layer with ReLU activation.
            x = self.output(x)  # Output layer.
            return x

    def train_new_task(self, new_data, new_labels, hidden_dim=10, epochs=200, batch_size=32):
        """
        Train a new neural column for a new task.

        Parameters:
        - new_data (numpy.ndarray): Training data for the new task.
        - new_labels (numpy.ndarray): Training labels for the new task.
        - hidden_dim (int, optional): Dimension of the hidden layer. Default is 10.
        - epochs (int, optional): Number of training epochs. Default is 200.
        - batch_size (int, optional): Batch size for training. Default is 32.
        """
        # Data preparation
        new_data_tensor = torch.FloatTensor(new_data).to(self.device)
        new_labels_tensor = torch.LongTensor(new_labels).to(self.device)
        new_task_dataset = TensorDataset(new_data_tensor, new_labels_tensor)
        new_task_loader = DataLoader(new_task_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Initialize a new neural column
        input_dim = new_data.shape[1]
        output_dim = len(set(new_labels))
        lateral_dims = [h[0] for h in self.hidden_outputs]
        total_input_dim = input_dim + sum(lateral_dims)
        new_column = self.NeuralColumn(input_dim=total_input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(
            self.device)

        optimizer = optim.Adam(new_column.parameters())

        # Training loop
        for epoch in range(epochs):
            for batch_data, batch_labels in new_task_loader:
                optimizer.zero_grad()

                # Compute lateral inputs from previous tasks
                lateral_input = self.calculate_lateral_input(batch_data)

                # Concatenate lateral inputs if available
                if lateral_input is not None:
                    input_to_column = torch.cat((batch_data, lateral_input), dim=1)
                else:
                    input_to_column = batch_data

                # Forward pass and loss computation
                output = new_column(input_to_column)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                optimizer.step()

        # Save the newly trained neural column and its hidden layer output function
        current_hidden = new_column.hidden
        self.hidden_outputs.append(
            (hidden_dim, lambda x: torch.relu(current_hidden(x.to(self.device)).to(self.device))))
        self.columns.append(new_column)

    def calculate_lateral_input(self, batch_data, latest_task_index=None):
        """
        Calculate lateral inputs based on the hidden outputs of previously trained tasks.

        Parameters:
        - batch_data (torch.Tensor): The input data for the current batch.
        - latest_task_index (int, optional): Specifies up to which task's hidden outputs should be considered
                                              for generating lateral connections. If None, uses all tasks.

        Returns:
        - torch.Tensor or None: The concatenated lateral inputs for the current batch if they exist,
                                 otherwise returns None.
        """
        hidden_outputs = []
        hidden_output = None

        if latest_task_index is None:
            latest_task_index = len(self.hidden_outputs)

        for i, (hidden_dim, hidden_function) in enumerate(self.hidden_outputs[:latest_task_index]):
            with torch.no_grad():  # No need to compute gradients
                # If this is the first hidden output, compute it based solely on batch_data
                if hidden_output is None:
                    hidden_output = hidden_function(batch_data)
                # Otherwise, concatenate the last hidden output to batch_data and compute the next
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
        """
        Predict the labels of the given data for the task at the given index.

        Parameters:
        - data (numpy.ndarray): The data for which to make predictions.
        - task_index (int): The index of the task for which to make predictions.

        Returns:
        - numpy.ndarray: The predicted labels.
        """
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)

            # If predicting a single sample, add an extra dimension to match the expected input shape
            if len(data_tensor.shape) == 1:
                data_tensor = data_tensor.unsqueeze(0)

            column = self.columns[task_index].to(self.device)

            lateral_input = self.calculate_lateral_input(data_tensor, task_index)

            # Forward pass through the neural column
            output = column(data_tensor, lateral_input=lateral_input)

            # If a single sample, return the index of the max value directly
            if len(output.shape) == 1:
                predicted = output.argmax()
            # For multiple samples, return indices of max values along dimension 1
            else:
                _, predicted = torch.max(output, 1)

            return predicted.cpu().numpy()

    def accuracy(self, data, labels, task_index):
        """
        Calculate the accuracy of the model for the task at the given index.

        Parameters:
        - data (numpy.ndarray): The test data.
        - labels (numpy.ndarray): The true labels for the test data.
        - task_index (int): The index of the task for which to calculate accuracy.

        Returns:
        - float: The accuracy of the model for the given task, as a percentage.
        """
        data = torch.FloatTensor(data).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)

        # Make predictions
        predicted = self.predict(data, task_index)

        # Calculate accuracy
        correct = (predicted == labels.cpu().numpy()).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        return accuracy

