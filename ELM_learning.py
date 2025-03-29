from BiLSTM.bi_lstm_functions import *
from ELM_functions import TuningELM
from ELM_data_preparation import *

# ELM parameters
ELM_input_size = X_train_tensor.shape[1]
ELM_hidden_size = 10
ELM_output_size = 1

ELM_model = ELM(input_size=ELM_input_size, hidden_size=ELM_hidden_size, output_size=ELM_output_size, device=device)

ELM_model.train(X_train_tensor, y_train_tensor)
print()
print("ELM training finished.")

# Save the t-ELM model
ELM_model.save_model(f'{ELM_PATH}/ELM_weights.pkl')
