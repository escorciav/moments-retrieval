import torch
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.getLogger('torch').setLevel(logging.NOTSET)

class BERT(object):
    def __init__(self, model_name='bert-base-uncased', features_combination_mode=0):
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

    def __call__(self, text):
        # Compute tokens from sentence
        tokens_tensor, segments_tensors, num_tokens = self._tokenization(text)
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        # Convert the hidden state embeddings into single token vectors [# tokens, # layers, # features]
        token_embeddings = self._compute_tokens_vectors(encoded_layers, num_tokens)
        # Word Vectors, compute features for each token
        features = self.features_combination(token_embeddings)
        # remove the special tokens [CLS]/[SEP] and transform to tensor
        features = torch.stack(features[1:-1])   
        return features

    def _tokenization(self, text):
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
            self.features_combination = self._concatenation_last_four_layers
        elif mode == 1:
            self.features_combination = self._summation_last_four_layers
        elif mode == 2:
            self.features_combination = self._last_layer
        else:
            raise('Feature combination modality unknown, specify value in list [0,1]')
        
    def _concatenation_last_four_layers(self, token_embeddings):
        return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] 
    
    def _summation_last_four_layers(self, token_embeddings):
        return [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] 
    
    def _last_layer(self, token_embeddings):
        return [layer[-1] for layer in token_embeddings]

    def _setup_dim(self):
        self.dim = 768
        if 'large' in self.model_name:
            self.dim = 1024
        if self.features_combination_mode == 0:
            self.dim = self.dim * 4
        
    def compute_tokens(self, text):
        '''
        DEPRECATED, USED FOR DEBUGGIN PURPOSES
        '''
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)[1:-1]
        return tokenized_text

if __name__ == '__main__':
    # Instantiate object
    model_name = 'bert-base-uncased'
    #model_name = 'bert-large-uncased'
    features_combination_mode = 2
    BERT = BERT(model_name, features_combination_mode)
    # Define sentence to encode
    text = "Here is the sentence I want embeddings for."
    #Compute vectors
    feat = BERT(text)
    print(BERT.compute_tokens(text))
    print(feat.shape)