import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')

X = df[['superficie', 'ambientes', 'baños', 'dormitorios', 'antiguedad', 'garage']].values
y = df['precio'].values.reshape(-1, 1)

# ReLU activation function
def relu(z): 
    return(np.maximum(0, z))

# Mean Squared Error (MSE) loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def main():
    
    # Hyperparameters
    input_dim = X.shape[1] # number of input layers
    output_dim = 1 # number of output layers
    hidden_dim = 5 # number of hidden layers
    learning_rate = 0.00000000001 # learning rate

    # Weight initialization
    w_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
    b_hidden = np.zeros((1, hidden_dim))
    w_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
    b_output = np.zeros((1, output_dim))

    epochs = 100000

    for epoch in range(epochs):

        # Forward pass
        h1_in = np.dot(X, w_hidden) + b_hidden
        h1_out = relu(h1_in)
        o1_in = np.dot(h1_out, w_output) + b_output
        predicted_out = o1_in

        # Loss function (MSE)
        loss = mean_squared_error(y, predicted_out)

        # Backpropagation
        d_predicted_output = 2 * (predicted_out - y)
        d_w_output = np.dot(h1_out.T, d_predicted_output)
        d_b_output = np.sum(d_predicted_output, axis=0, keepdims=True)
        d_hidden = np.dot(d_predicted_output, w_output.T)
        d_hidden[h1_in <= 0] = 0
        d_w_hidden = np.dot(X.T, d_hidden)
        d_b_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights
        w_hidden -= learning_rate * d_w_hidden
        b_hidden -= learning_rate * d_b_hidden
        w_output -= learning_rate * d_w_output
        b_output -= learning_rate * d_b_output

        # Print loss
        if epoch % 100 == 0:
            print('Epoch', epoch, 'loss', loss)


    x_test = [[33, 1, 1, 0, 1, 0]]

    # Realizar la predicción
    h1_in_test = np.dot(x_test, w_hidden) + b_hidden
    h1_out_test = relu(h1_in_test)
    o1_in_test = np.dot(h1_out_test, w_output) + b_output
    predicted_price = o1_in_test[0, 0]

    print(f'Precio predicho para la propiedad: ${predicted_price:.2f}') 


if __name__ == "__main__":
    main()

