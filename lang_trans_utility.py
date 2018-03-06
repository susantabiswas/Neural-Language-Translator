# functions for data loading and preprocessing
import numpy as np



# loads the datafile contents and makes vocabulary for input and target language
def load_dataset(data_path, num_samples ):
    input_texts = []  # for storing the input text data

    target_texts = []  # for storing the target text data
    input_chars = set()  # for storing the unique chars in input text data
    target_chars = set()  # for storing the unique chars in target text data
    # Variable Initialization
    encoder_unique_tokens = 0  # unique tokens in encoder input
    decoder_unique_tokens = 0  # unique tokens in decoder output
    Tx = 0  # max length of input sequence for encoder
    Ty = 0  # max length of output sequence for decoder
    
    # read the data file
    with open(data_path, 'r', encoding='utf-8') as file:
        text_data = file.read().split('\n')

    print('Total no. of lines of Original Text data: ' + str(len(text_data)))

    # add input and target data
    end_index = min(num_samples, len(text_data) - 1)
    for line in text_data[: end_index]:
        # since each line is of format: Lang1 + '\t' + Lang2
        input_line, target_line = line.split('\t')

        # we will use '\t' as start_char and '\n' as end character
        target_line = '\t' + str(target_line) + '\n'
        input_texts.append(input_line)
        target_texts.append(target_line)

        # update the max. sequence lengths for encoder and decoder
        Tx = max(Tx, len(input_line))
        Ty = max(Ty, len(target_line))

        # find the unique characters in input and target text data
        for char in input_line:
            if char not in input_chars:
                input_chars.add(char)
        for char in target_line:
            if char not in target_chars:
                target_chars.add(char)

    encoder_unique_tokens = len(input_chars)
    decoder_unique_tokens = len(target_chars)
    input_chars = sorted(input_chars)
    target_chars = sorted(target_chars)

    return encoder_unique_tokens, decoder_unique_tokens, input_chars, target_chars, input_texts, target_texts, Tx, Ty


# creates dictionary mappings for input language and target language
# Two kinds of mappings are created for each language
# 1. char to numerical index
# 2. numerical index to char
def create_mappings(input_chars, target_chars):
    # we feed numerical values to a RNN so for that we need to convert
    # the chars to numbers, so we make a mapping of chars to numbers
    input_char_idx = dict([(char, i) for i, char in enumerate(input_chars)])
    target_char_idx = dict([(char, i) for i, char in enumerate(target_chars)])

    # dict for reverse lookup from indices to tokens
    input_idx_char = dict((i, char) for char, i in input_char_idx.items())
    target_idx_char = dict((i, char) for char, i in target_char_idx.items())

    return input_char_idx, input_idx_char, target_char_idx, target_idx_char


# for create One Hot Encoding of data
'''
    user_input: string input
    T: No. of timesteps 
    encoder_unique_tokens: class labels for OHE
    char_to_idx: mapping from characters to numerical index
'''
def to_OHE(user_input, T, encoder_unique_tokens, char_to_idx):
    # truncate the input string if it is longer than No. of timesteps
    if len(user_input) > T:
        user_input = user_input[:T]

    enc_input_data = np.zeros((1, T, encoder_unique_tokens), dtype='float32')
    for curr_timestep, char in enumerate(user_input):
        enc_input_data[0, curr_timestep, char_to_idx[char]] = 1

    return enc_input_data
