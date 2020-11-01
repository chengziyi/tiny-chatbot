from typing import Any, Text, Optional, Tuple, List, Dict

class Message:
    def __init__(
        self, text: Text, data=None, output_properties=None, time=None
    ) -> None:
        self.text = text
        self.time = time
        self.data = data if data else {}

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    @classmethod
    def build(cls, text, intent=None, entities=None) -> "Message":
        data = {}
        if intent:
            split_intent, response_key = cls.separate_intent_response_key(
                intent  # pytype: disable=attribute-error
            )
            data['intent'] = split_intent
            if response_key:
                data['response_key'] = response_key
        if entities:
            data['entities'] = entities
        return cls(text, data)

    @staticmethod
    def separate_intent_response_key(original_intent) -> Optional[Tuple[Any, Any]]:

        split_title = original_intent.split("/")
        if len(split_title) == 2:
            return split_title[0], split_title[1]
        elif len(split_title) == 1:
            return split_title[0], None

    def set(self, prop, info, add_to_output=False) -> None:
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop, default=None) -> Any:
        if prop == 'text':
            return self.text
        return self.data.get(prop, default)

class TrainingData:
    """Holds loaded intent and entity training data."""
    """将训练数据以Message的list方式保存"""
    def __init__(
        self,
        training_examples: Optional[List[Message]] = None,
    ) -> None:

        if training_examples:
            self.training_examples = training_examples
            # self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []

        self.features_labels = dict()
