
import numpy as np
from keras.models import model_from_json

num_decoder_tokens = 29
max_decoder_seq_length = 119
target_char_to_idx = {'\t': 0,'\n': 1,' ': 2,'a': 3,'b': 4,'c': 5,'d': 6,'e': 7,'f': 8,'g': 9,'h': 10,'i': 11,'j': 12,'k': 13,'l': 14,'m': 15,'n': 16,'o': 17,'p': 18,'q': 19,'r': 20,
 's': 21,'t': 22,'u': 23,'v': 24,'w': 25,'x': 26,'y': 27,'z': 28}

input_char_to_idx = {' ': 0,'a': 1,'b': 2,'c': 3,'d': 4,'e': 5,'f': 6, 'g': 7, 'h': 8, 'i': 9,'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 
                     'p': 16, 'q': 17, 'r': 18,'s': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}
target_idx_to_char = {0: '\t',1: '\n', 2: ' ', 3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f', 9: 'g', 10: 'h', 11: 'i', 12: 'j', 13: 'k',
 14: 'l',15: 'm', 16: 'n', 17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u', 24: 'v', 25: 'w', 26: 'x', 27: 'y', 28: 'z'}     

def init(): 
    
    json_file = open('./Model/encode.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    print(json_file)
    # use Keras model_from_json to make a loaded model
    encode_model = model_from_json(loaded_model_json)
    # load weights into new model
    encode_model.load_weights("./Model/encode.h5")

    json_file = open('./Model/decode.json','r')
    loaded_model_json = json_file.read()
    json_file.close()    
    # use Keras model_from_json to make a loaded model
    decode_model = model_from_json(loaded_model_json)
    # load weights into new model
    decode_model.load_weights("./Model/decode.h5")
    

	#compile and evaluate loaded model
    #encode_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    decode_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    return encode_model,decode_model


def convert_sent_encode(sent):
    encoder_input_data = np.zeros((56,27), dtype='float32')
    for t, char in enumerate(sent):
        encoder_input_data[t, input_char_to_idx[char]] = 1.
  
    return encoder_input_data

def decode_sequence(input_seq,encode_model,decode_model):

    states_value = encode_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_char_to_idx['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decode_model.predict([target_seq] + states_value)
    
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_idx_to_char[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
              stop_condition = True
      

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
        states_value = [h, c]
    
    return decoded_sentence

