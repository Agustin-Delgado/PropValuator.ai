import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relu(z): 
    return np.maximum(0, z)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def plot_loss_curve(train_losses, test_losses):
    for i in range(len(train_losses)):
        plt.plot(range(epochs), train_losses[i], label=f'Fold {i+1} Train Loss')
        plt.plot(range(epochs), test_losses[i], label=f'Fold {i+1} Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss for Each Fold')
        plt.legend()
        plt.show()

df = pd.read_csv('data.csv')

X_total = df[['surface', 'rooms', 'bathrooms', 'bedrooms', 'age', 'garage']].values
y_total = df['price'].values.reshape(-1, 1)

num_folds = 5
fold_size = len(X_total) // num_folds

train_mae_scores = []
test_mae_scores = []
train_losses_per_fold = []
test_losses_per_fold = []

# Hyperparameters
input_dim = X_total.shape[1]
output_dim = 1
hidden_dim = round(len(X_total) / (2 * (input_dim + output_dim)))
learning_rate = 0.0000000007
epochs = 100

for fold in range(num_folds):
    start_idx = fold * fold_size
    end_idx = (fold + 1) * fold_size
    
    X_test_fold = X_total[start_idx:end_idx]
    y_test_fold = y_total[start_idx:end_idx]

    X_train_fold = np.concatenate((X_total[:start_idx], X_total[end_idx:]), axis=0)
    y_train_fold = np.concatenate((y_total[:start_idx], y_total[end_idx:]), axis=0)

    w_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b_hidden = np.zeros((1, hidden_dim))
    w_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b_output = np.zeros((1, output_dim))

    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):

        # Training forward pass
        h1_in = np.dot(X_train_fold, w_hidden) + b_hidden
        h1_out = relu(h1_in)
        o1_in = np.dot(h1_out, w_output) + b_output
        predicted_out = o1_in

        train_loss = mean_absolute_error(y_train_fold, predicted_out)
        train_losses.append(train_loss)

        # Testing forward pass
        h1_in_test = np.dot(X_test_fold, w_hidden) + b_hidden
        h1_out_test = relu(h1_in_test)
        o1_in_test = np.dot(h1_out_test, w_output) + b_output
        predicted_out_test = o1_in_test

        test_loss = mean_absolute_error(y_test_fold, predicted_out_test)
        test_losses.append(test_loss)

        # Backpropagation
        d_predicted_output = 2 * (predicted_out - y_train_fold)
        d_w_output = np.dot(h1_out.T, d_predicted_output)
        d_b_output = np.sum(d_predicted_output, axis=0, keepdims=True)
        d_hidden = np.dot(d_predicted_output, w_output.T)
        d_hidden[h1_out <= 0] = 0 # ReLU backprop
        d_w_hidden = np.dot(X_train_fold.T, d_hidden)
        d_b_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights
        w_hidden -= learning_rate * d_w_hidden
        b_hidden -= learning_rate * d_b_hidden
        w_output -= learning_rate * d_w_output
        b_output -= learning_rate * d_b_output
    
    train_losses_per_fold.append(train_losses)
    test_losses_per_fold.append(test_losses)

    train_fold_mae = mean_absolute_error(y_train_fold, predicted_out)
    test_fold_mae = mean_absolute_error(y_test_fold, predicted_out_test)

    train_mae_scores.append(train_fold_mae)
    test_mae_scores.append(test_fold_mae)
    for i in range(len(predicted_out_test)):

        print('Predicted price:', round(predicted_out_test[i][0]), 'Ground truth:', y_test_fold[i][0])
average_train_mae = np.mean(train_mae_scores)
average_test_mae = np.mean(test_mae_scores)

plot_loss_curve(train_losses_per_fold, test_losses_per_fold)

# Print all the predictions and the respective ground truth values


# Print the average MAE
print('Average train MAE:', average_train_mae)
print('Average test MAE:', average_test_mae)