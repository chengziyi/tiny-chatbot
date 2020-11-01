import sys  
sys.path.append('./nlu')
sys.path.append('./core')

from typing import Text,Dict
import logging
logging.basicConfig(format='%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

import queue
import threading
import time
q1=queue.Queue(maxsize=0)
q2=queue.Queue(maxsize=0)

from nlu.interpreter import Interpreter
from core.policyensemble import PolicyEnsemble

def read_yml_file(filename: Text) -> Dict:
    import yaml
    with open(filename) as f:
        file_data=f.read()
    data=yaml.load(file_data, Loader=yaml.FullLoader)
    return data

class StateTracker:
    def __init__(self):
        self.current_states = ['none']*5

    def update(self, state: Text):
        if state == 'bot_listen' and self.current_states[-1] == 'bot_listen':
            return
        self.current_states.append(state)
        self.current_states.pop(0)

def restaurant_form(domain: Dict):
    responses = domain['responses']
    slots=domain['slots']
    values={}

    while True:
        message=q1.get()
        q1.task_done()
        # print (f'form get message: {message}\n')

        if message == 'deactive':
            break
        elif message == 'active':
            pass
        else:
            k,v=message.split(':')
            values[k]=v
        # print('form values:',values)
        for i in slots:
            if i not in values:
                message_return = responses[f'utter_ask_{i}'][0]['text']
                # print('form message return:',message_return)
                q2.put(message_return)
                break

        if len(values)==len(slots):
            # print('form get all slots')
            a=values['cuisine']
            b=values['num_people']
            c=values['seating']
            q2.put(f"""OK, I am going to run a restaurant search using the following parameters:
    -cuisine:{a}
    -num_people:{b}
    -seating:{c}""")
            break

class Agent:
    '''agent include interpreter, policies, actions, channel'''
    def __init__(self, cfg_file: Text, domain_file: Text):
        self.interpreter = Interpreter(cfg_file)
        self.policy_ensemble = PolicyEnsemble(cfg_file, domain_file)
        domain = read_yml_file(domain_file)
        self.responses = domain['responses']

        form_thread=threading.Thread(target=restaurant_form,args=(domain,))
        form_thread.start()

    def train_nlu(self, nlu_filename: Text):
        self.interpreter.train(nlu_filename)

    def train_core(self, core_filename: Text):
        self.policy_ensemble.train(core_filename)

    def train(self, nlu_filename: Text, core_filename: Text):
        self.train_nlu(nlu_filename)
        self.train_core(core_filename)

    def send_message_to_form(self, send: Text):
        q1.put(send)
        recive=q2.get()
        q2.task_done()
        print()
        print(recive)
        print()

    def deactive_form(self):
        q1.put('deactive')

    def run(self):
        self.interpreter.load()
        self.policy_ensemble.load()
        print("Bot loaded. Type a message and press enter (use '/stop' to exit):")
        state_tracker = StateTracker()
        form_activated = False

        while True:
            core_result = self.policy_ensemble.predict(state_tracker.current_states)
            next_move = core_result['label']
            # print(f'debug: current_states:{state_tracker.current_states}, core_result:{core_result}')
            state_tracker.update(next_move)
            if next_move == 'bot_listen':
                user_input = input('>>> ')
                if user_input == '/stop':
                    break

                nlu_result = self.interpreter.predict(user_input)
                if nlu_result['intent'] == 'inform':
                    if form_activated:
                        for k,v in nlu_result['entities'][0].items():
                            entity=k
                            value=v 
                        message = f'{entity}:{value}'
                        self.send_message_to_form(message)
                else:
                    state_tracker.update(nlu_result['intent'])

            elif next_move in ['utter_greet','utter_noworries','utter_chitchat','utter_ask_continue']:
                print()
                print(self.responses[next_move][0]['text'])
                print()
            elif next_move == 'activate_restaurant_form':
                form_activated = True
                self.send_message_to_form('active')
            else:
                print('not handle move:',next_move)

        self.deactive_form()

if __name__ == '__main__':
    CONFIG_FILE = './config.yml'
    NLU_DATA = './nlu.md'
    CORE_DATA = './stories.md'
    DOMAIN_FILE='./domain.yml'

    agent = Agent(CONFIG_FILE, DOMAIN_FILE)
    agent.train(NLU_DATA, CORE_DATA)
    agent.run()
