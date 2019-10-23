import torch
# import logging
# logging.basicConfig(level=logging.NOTSET)
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import json
import numpy as np
from tqdm import tqdm
import time as time

class BERTEmbedding(object):
    def __init__(self, data_directory=None, model_name='bert-base-uncased', features_combination_mode=0):
        self.model_name = model_name
        self.features_combination_mode = features_combination_mode
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(model_name)
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        # Determine the modality in which layers are combined to obtain the final features
        self._select_combination_mode()
        self._setup_dim()
        if data_directory:
            self.bert_dict = {}
            self._load_preprocessed_features(data_directory=data_directory)

    def __call__(self, key):
        '''
            returns tuple (feat[numpy],query_length) if integer key is provided
            return features[torch tensor] if tuple is provided.

            Usage:
                - In training we use the preprocessed sentences through annotation_id of each moment
                - For standalone processing we need to compute first the tokenized version of the 
                sentence and then call the model on that tokenization. 
                Check below UNIT TEST in main for more details.
        '''
        if type(key) == int:
            return self.bert_dict[str(key)]
        elif type(key) == tuple:
            return self._compute_features(key)
        else:
            raise('Invalid input to bert module')

    def _compute_features(self, tokens):
        # Compute tokens from sentence
        tokens_tensor, segments_tensors, num_tokens = tokens
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, check = self.model(tokens_tensor, segments_tensors)
        # Convert the hidden state embeddings into single token vectors [# tokens, # layers, # features]
        token_embeddings = self._compute_tokens_vectors(encoded_layers, num_tokens)
        # Word Vectors, compute features for each token
        features = self.features_combination(token_embeddings)
        # remove the special tokens [CLS]/[SEP] and transform to tensor
        features = torch.stack(features[1:-1])   
        return features

    def _load_preprocessed_features(self, data_directory):
        print('Loading language features')
        t = time.time()
        m = self.model_name.replace('-','_')
        f = self.features_combination_mode
        max_words = 50
        filename = f'./data/processed/{data_directory}/bert/{m}_comb_mode_{f}.json'
        feat = json.load(open(filename, 'r'))
        for k,f in feat.items():
            len_query = min(len(f), max_words)
            padding_size = max_words - len_query
            feature = np.pad(np.asarray(f), [(0,padding_size),(0,0)], mode='constant')
            self.bert_dict[k] = (torch.from_numpy(feature).type(torch.FloatTensor),len_query)
        print("Time to load precomputed language features {:.2f}".format(time.time()-t))

    def _tokenization(self, text):
        #TODO: return a dictionary and non a tuple, increase readability
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        num_tokens = len(tokenized_text)
        segments_ids = [1] * num_tokens
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokens_tensor, segments_tensors, num_tokens

    def _compute_tokens_vectors(self, encoded_layers, num_tokens):
        token_embeddings = [] 
        for token_i in range(num_tokens):
            hidden_layers = [] 
            for layer_i in range(len(encoded_layers)):
                vec = encoded_layers[layer_i][0][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        return token_embeddings

    def _select_combination_mode(self):
        mode = self.features_combination_mode
        if mode == 0:
            self.features_combination = self._last_layer
        elif mode == 1:
            self.features_combination = self._summation_last_four_layers
        elif mode == 2:
            self.features_combination = self._concatenation_last_four_layers
        elif mode == 3:
            self.features_combination = self._summation_second_to_last
        else:
            raise('Feature combination modality unknown, specify value in list [0,1]')
        
    def _concatenation_last_four_layers(self, token_embeddings):
        return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] 
    
    def _summation_last_four_layers(self, token_embeddings):
        return [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] 
    
    def _last_layer(self, token_embeddings):
        return [layer[-1] for layer in token_embeddings]
    
    def _summation_second_to_last(self, token_embeddings):
        return [torch.sum(torch.stack(layer)[1:], 0) for layer in token_embeddings] 

    def _setup_dim(self):
        self.dim = 768
        if 'large' in self.model_name:
            self.dim = 1024
        if self.features_combination_mode == 0:
            self.dim = self.dim * 4
        
    def compute_text_tokens(self, text):
        '''
        DEPRECATED, USED FOR DEBUGGIN PURPOSES
        '''
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)[1:-1]
        return tokenized_text

def _precompute_language_features(data_directory, max_words):
    if data_directory == 'didemo':
        subsets = ['train-01.json', 'val-01.json', 'test-01.json']
    elif data_directory == 'charades-sta':
        subsets = ['train-01.json', 'val-02_01.json', 'test-01.json']

    for model_name in ['bert-base-uncased','bert-large-uncased']:
        for features_combination_mode in [0,1,2,3]:
            BERT = BERTEmbedding(data_directory=None,model_name=model_name, 
                        features_combination_mode=features_combination_mode)
            print(f'\nProcessing model {model_name} and feature aggregation {features_combination_mode}')
            lang_features = {}
            for subset in subsets:
                print(f'Loading original data... ({subset})')
                filename = f'./data/processed/{data_directory}/{subset}'
                original_data = json.load(open(filename, 'r'))
                for i in tqdm(range(len(original_data['moments']))):
                    moment_i = original_data['moments'][i]
                    tokens  = BERT._tokenization(moment_i['description'])
                    feature = BERT(tokens)
                    lang_features[moment_i['annotation_id']] = feature.tolist()
            
            m = model_name.replace('-','_')
            filename_d = f'./data/processed/{data_directory}/bert/{m}_comb_mode_{features_combination_mode}.json'
            print(f'Dumping {filename_d}')
            with open(filename_d, 'w') as fid:
                json.dump(lang_features, fid)
    
if __name__ == '__main__':

    # UNIT TEST
    # Instantiate object
    model_name = 'bert-base-uncased'
    #model_name = 'bert-large-uncased'
    features_combination_mode = 3
    BERT = BERTEmbedding(model_name=model_name, 
            features_combination_mode=features_combination_mode)
    # Define sentence to encode
    text = "Here is the sentence I want embeddings for."
    tokens = BERT._tokenization(text)
    #Compute vectors
    feat = BERT(tokens)
    print(BERT.compute_text_tokens(text))
    print(feat.shape)
    print(feat.dtype)

    # TEST PRECOMPUTED FEATURES
    model_name = 'bert-base-uncased'
    features_combination_mode = 3
    BERT = BERTEmbedding(data_directory='didemo', model_name=model_name, 
            features_combination_mode=features_combination_mode)
    feat = BERT(0)
    print(feat[0].shape)
    print(feat[0].dtype)

    # _precompute_language_features(data_directory='didemo', max_words=50)
    # _precompute_language_features(data_directory='charades-sta', max_words=50)