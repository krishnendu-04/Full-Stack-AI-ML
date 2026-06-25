
print("\nVISUALIZATION OF TRANSFORMER ARCHITECTURE.............\n")
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Define a simplified transformer Encoder Block
def transformer_encoder(input_dim, num_heads, ff_dim):
    inputs = Input(shape=(None, input_dim))
    # Mulit-head self attention
    attention_output = MultiHeadAttention(num_heads=num_heads,key_dim=input_dim)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    # Feed forward nueral network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(input_dim)(ff_output)
    outputs = Add()([attention_output,ff_output])
    outputs = LayerNormalization()(outputs)
    return Model(inputs, outputs)


# Create and visualize a sample transformer encoder block
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file="transformer_encoder.png")



print("\n--------------PYTORCH----------------\n")
from transformers import BertTokenizer, BertModel

# Load a pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Tokenize a sample input
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors='pt')

# Pass the input through the model
outputs = model(**inputs)
print("Hidden States Shape: ", outputs.last_hidden_state.shape)





print("\n--------------TENSORFLOW----------------\n")
from transformers import TFBertModel, BertTokenizer

# Load a pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased")

# Tokenize a sample input
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors='tf')

# Pass the input through the model
outputs = model(**inputs)
print("Hidden States Shape: ", outputs.last_hidden_state.shape)