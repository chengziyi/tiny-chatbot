from typing import Any, Text, Optional, Tuple, List, Dict, Union

from train_core import Policy,MemoizationPolicy,PytorchPolicy
from read_core_data import MarkdownStoryReader

import logging


policy_classes = [
    MemoizationPolicy,
    PytorchPolicy,
]

registered_policies = {c.name: c for c in policy_classes}

class PolicyEnsemble:
    '''policy ensemble'''

    def __init__(
        self,
        cfg_file: Text,
        domain_file: Text,
    ):
        cfg = Policy.read_yml_file(cfg_file)
        # build pipeline
        self.policies = self._build_pipeline(cfg, domain_file)

    def _build_pipeline(
        self, cfg: Dict, domain_file: Text
    ) -> List[Policy]:
        """Transform the passed names of the pipeline components into classes."""

        pipeline = []

        for i in cfg['policies']:
            policy_name=i['name']
            policy=registered_policies[policy_name](domain_file)
            pipeline.append(policy)

        return pipeline

    def _load_data(self, filename: Text) -> Optional["TrainingData"]:
	    reader = MarkdownStoryReader()
	    return reader.read_from_file(filename)

    def train(self, filename: Text):
        working_data=self._load_data(filename)
        [policy.train(working_data) for policy in self.policies]

    def predict(self, data: List[str]) -> Dict:
        for policy in self.policies:
            result = policy.predict(data)
            if policy.name == 'MemoizationPolicy' and result is not None:break
            # logging.debug(f'{policy.name}:{result}')
        return result

    def load(self):
        [policy.load() for policy in self.policies]

if __name__ == '__main__':
    CONFIG_FILE = '../config.yml'
    DOMAIN_FILE='../domain.yml'
    pe=PolicyEnsemble(CONFIG_FILE,DOMAIN_FILE)
    # pe.train('../stories.md')
    pe.load()
    r=pe.predict(['none', 'none', 'none', 'none', 'none'])
    print(r)
