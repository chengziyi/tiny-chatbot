from typing import Any, Text, Optional, Tuple, List, Dict, Match
import copy
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim

import logging

from MyTransformerEncoder import TransformerEncoder

MEMORIZE_POLICY='./model/memorizepolicy.pkl'

class PolicyMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls):
        """The name property is a function of the class - its __name__."""

        return cls.__name__

class Policy(metaclass=PolicyMetaclass):

    @property
    def name(self):
        """Access the class's property name from an instance."""

        return type(self).name

    def __init__(self, domain_file: Text):
        self.user_intent = self.get_user_intent(domain_file)

    def get_user_intent(self, filename: Text) -> Dict:
        domain = self.read_yml_file(filename)
        return domain['intents']

    @staticmethod
    def read_yml_file(filename: Text) -> Dict:
        import yaml
        with open(filename) as f:
            file_data=f.read()
        data=yaml.load(file_data, Loader=yaml.FullLoader)
        return data

    def training_states_and_actions(self, training_data: List[List], max_history: int):
        '''input training data, return states list and actions list'''
        states = []
        actions = []
        for story in training_data:
            tmp_states = ['none']*max_history
            if len(story) is 0:continue
            for i,step in enumerate(story):
                '''data contain bot_actions and user_intents,we need predict bot_actions
                   so states as X, bot_action as Y'''
                if step not in self.user_intent:
                    actions.append(step)
                    states.append(copy.deepcopy(tmp_states))
                tmp_states.append(step)
                tmp_states.pop(0)

        # for s,a in zip(states,actions):
        #     print(s,a)
        return states,actions

    def load(self):
        pass

class MemoizationPolicy(Policy):
    '''just map the dialog states and actions in a dict'''
    def __init__(self, domain_file: Text):
        super(MemoizationPolicy, self).__init__(domain_file)
        self.lookup = dict()

    def train(self, training_data: List[List], max_history=5):
        if os.path.exists(MEMORIZE_POLICY):
            self.lookup = joblib.load(MEMORIZE_POLICY)
        else:
            data_x,data_y = self.training_states_and_actions(training_data, max_history)
            # print(data_x[0],data_y[0])
            for x,y in zip(data_x,data_y):
                x = ' '.join(x)
                self.lookup[x]=y

            joblib.dump(self.lookup,MEMORIZE_POLICY)

    def predict(self, data: List[str]) -> Optional[Dict]:
        key = ' '.join(data)
        if key in self.lookup:
            return {'label':self.lookup[key],'confidence':1.0}
        else:
            return None

    def load(self):
        self.lookup = joblib.load(MEMORIZE_POLICY)

class PytorchPolicy(Policy):
    """use self-attention layer to extract dialog states features"""
    def __init__(self, domain_file: Text):
        super(PytorchPolicy, self).__init__(domain_file)

        self.word2id = None
        self.label2id = None

        ##parameters
        self.LR=0.001
        self.EPOCHS=20
        MAX_LEN = 5
        self.src_len = MAX_LEN # length of source

        # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device='cpu'
        self.model = None

    def train(self, training_data: List[List], max_history=5):
        data_x,data_y = self.training_states_and_actions(training_data, max_history)
        data_x = list(map(lambda x:' '.join(x),data_x))

        sentences = data_x
        labels = data_y
        
        vocab = set((' '.join(sentences)).split()) | set(labels)
        vocab = sorted(vocab)
        self.word2id = {w:i for i,w in enumerate(vocab)}
        vocab_size = len(self.word2id)

        self.label2id = {v:i for i,v in enumerate(sorted(set(labels)))}
        target_size = len(self.label2id)

        if os.path.exists('./model/torch_model.pkl'):
            logging.info('model exists')
            return

        self.model = TransformerEncoder(vocab_size, target_size, self.src_len)
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        enc_inputs,output_batch = self.make_batch(sentences,labels)
        enc_inputs=enc_inputs.to(self.device)
        output_batch=output_batch.to(self.device)
        print(f'inputs:{enc_inputs.shape}',f'outputs:{output_batch.shape}')
        print('device:',enc_inputs.device)

        self.model.train()
        for epoch in range(self.EPOCHS):
            optimizer.zero_grad()
            outputs = self.model(enc_inputs)
            loss = criterion(outputs, output_batch)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()
            if loss<0.1:
                print('loss<0.1,break')
                break

        torch.save(self.model, './model/torch_model.pkl')
        joblib.dump(self.word2id, './model/word2id.pkl')
        joblib.dump(self.label2id, './model/label2id.pkl')

    def load(self):
        self.model = torch.load('./model/torch_model.pkl')
        self.word2id = joblib.load('./model/word2id.pkl')
        self.label2id = joblib.load('./model/label2id.pkl')

    def predict(self, sentences: List[Text]) -> Dict:
        data = ' '.join(sentences)
        data_inputs = [data]

        input_batch = [[self.word2id[t] for t in n.split()] for n in data_inputs]
        id2label = {v:k for k,v in self.label2id.items()}

        self.model.eval()
        predict = self.model(torch.LongTensor(input_batch))
        confidence = predict.softmax(1)
        # print(confidence)
        predict = predict.data.max(1, keepdim=True)[1]

        if len(data_inputs) == 1:
            # print(data_inputs,'->',id2label[predict.squeeze().item()])
            return {'label':id2label[predict.squeeze().item()],'confidence':confidence.max().item()}
        else:
            # print(data_inputs, '->', [id2label[i.item()] for i in predict.squeeze()])
            return [{'label':id2label[v.item()],'confidence':confidence[i].max().item()} for i,v in enumerate(predict.squeeze())]

    def make_batch(self, sentences: List[Text], labels: List[Text]):
        assert(len(sentences)==len(labels))
        input_batch = [[self.word2id[t] for t in n.split()] for n in sentences]
        output_batch = [self.label2id[i] for i in labels]
        return torch.LongTensor(input_batch),torch.LongTensor(output_batch)


if __name__ == '__main__':
    DOMAIN_FILE='../domain.yml'

    from read_core_data import MarkdownStoryReader
    def _load(filename: Text):
        """Loads a single training data file from disk."""

        reader = MarkdownStoryReader()
        return reader.read_from_file(filename)

    print("**module test**")
    training_data=_load('../stories.md')
    core_config=Policy.read_yml_file('../config.yml')
    print(core_config['policies'])

    # t=MemoizationPolicy(DOMAIN_FILE)
    # t.train(training_data)
    # t.load()
    t=PytorchPolicy(DOMAIN_FILE)
    t.train(training_data)
    t.load()

    testdata=[
        ['none', 'none', 'none', 'none', 'none'], 
        ['none', 'none', 'none', 'bot_listen', 'greet'], 
        ['none', 'none', 'bot_listen', 'greet', 'utter_greet'], 
        ['bot_listen', 'greet', 'utter_greet', 'bot_listen', 'request_restaurant'], 
        ['bot_listen', 'request_restaurant', 'activate_restaurant_form','bot_listen','thankyou'], 
    ]
    for data in testdata:
        print(t.predict(data)) 

'''test data
['none', 'none', 'none', 'none', 'none'], 
['none', 'none', 'none', 'bot_listen', 'greet'], 
['none', 'none', 'bot_listen', 'greet', 'utter_greet'], 
['bot_listen', 'greet', 'utter_greet', 'bot_listen', 'request_restaurant'], 
['greet', 'utter_greet', 'bot_listen', 'request_restaurant', 'activate_restaurant_form'], 
['bot_listen', 'request_restaurant', 'activate_restaurant_form', 'bot_listen', 'thankyou']
'''