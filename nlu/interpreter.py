from typing import Any, Text, Optional, Tuple, List, Dict, Union

from read_nlu_data import MarkdownReader
from train_nlu import Trainer


class Interpreter:
    '''nlu interpreter'''
    def __init__(self, cfg_file: Text):
        nlu_config=self._read_yml_file(cfg_file)
        self.trainer=Trainer(nlu_config)

    def _read_yml_file(self, filename: Text) -> Dict:
        import yaml
        with open(filename) as f:
            file_data=f.read()
        data=yaml.load(file_data, Loader=yaml.FullLoader)
        return data

    def _load_data(self, filename: Text) -> Optional["TrainingData"]:
        reader = MarkdownReader()
        return reader.read(filename)

    def train(self, filename: Text):
        training_data=self._load_data(filename)
        self.trainer.train(training_data)

    def predict(self, data: Text) -> Dict:
        return self.trainer.predict(data)

    def load(self):
        self.trainer.load()

if __name__ == "__main__":
    CONFIG_FILE = '../config.yml'
    interpreter = Interpreter(CONFIG_FILE)
    # interpreter.train('../nlu.md')
    interpreter.load()
    r=interpreter.predict('can i get swedish food in any area')
    print(r)