from typing import Any, Text, Optional, Tuple, List, Dict, Union
import copy
import random
import spacy
import os
import joblib

from training_data import Message,TrainingData
from read_nlu_data import MarkdownReader

TEXT='text'

INTENT='intent'

ENTITIES='entities'

TEXT_TOKENS='text_tokens'

MESSAGE_ATTRIBUTES = [TEXT, INTENT]

TOKENS_NAMES = {
    TEXT: "text_tokens",
    INTENT: "intent_tokens",
}

NER_MODEL = './model/ner'
INTENT_CLF = './model/intent_clf.pkl'
COUNT_VECTOR_FEATURIZER = './model/count_vector_featurizer.pkl'
ID_LABEL = './model/id2label.pkl'

class ComponentMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls):
        """The name property is a function of the class - its __name__."""

        return cls.__name__

class Component(metaclass=ComponentMetaclass):
    """A component is a message processing unit in a pipeline.

    Components are collected sequentially in a pipeline. Each component
    is called one after another. This holds for
    initialization, training, persisting and loading the components.
    If a component comes first in a pipeline, its
    methods will be called first.

    E.g. to process an incoming message, the ``process`` method of
    each component will be called. During the processing
    (as well as the training, persisting and initialization)
    components can pass information to other components.
    The information is passed to other components by providing
    attributes to the so called pipeline context. The
    pipeline context contains all the information of the previous
    components a component can use to do its own
    processing. For example, a featurizer component can provide
    features that are used by another component down
    the pipeline to do intent classification.
    """

    # Component class name is used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    @property
    def name(self):
        """Access the class's property name from an instance."""

        return type(self).name

    def train(
        self,
        training_data: TrainingData,
        config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.train`
        of components previous to this one.

        Args:
            training_data:
                The :class:`rasa.nlu.training_data.training_data.TrainingData`.
            config: The model configuration parameters.

        """

        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            message: The :class:`rasa.nlu.training_data.message.Message` to process.

        """

        pass

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading.

        Args:
            file_name: The file name of the model.
            model_dir: The directory to store the model to.

        Returns:
            An optional dictionary with any information about the stored model.
        """

        pass

    def load(self):
        pass

class WhitespaceTokenizer(Component):

    def __init__(self):
        pass

    def _split_intent(self, message: Message):
        text = message.get(INTENT)
        return [text]

    def tokenize(self, message: Message):
        text = message.text
        return text.split()

    def predict(self, data: Dict) -> Dict:
        """英语不分词，直接返回"""
        return data

    def train(self,training_data: TrainingData) -> None:
        """Tokenize all training data."""

        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if example.get(attribute) is not None:
                    if attribute == INTENT:
                        tokens = self._split_intent(example)
                    if attribute == TEXT:
                        tokens = self.tokenize(example)

                    example.set(TOKENS_NAMES[attribute], tokens)

class CountVectorsFeaturizer(Component):

    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer
        self.countvectorizer = CountVectorizer()

    def predict(self, data: Dict) -> Dict:
        data['result']=self.countvectorizer.transform([data['result']])
        return data

    def load(self):
        self.countvectorizer = joblib.load(COUNT_VECTOR_FEATURIZER)

    def train(self, training_data: TrainingData) -> None:
        """Featurize all training data."""
        corpus=[]
        labels=[]
        for example in training_data.training_examples:
            for attribute in [TEXT_TOKENS, INTENT]:
                if example.get(attribute) is not None:
                    if attribute == TEXT_TOKENS:
                        corpus.append(' '.join(example.get(attribute)))
                    if attribute == INTENT:
                        labels.append(example.get(attribute))

        countvector = self.countvectorizer.fit_transform(corpus)
        # print(countvector.toarray())

        label2id = dict()
        label_id = 0
        for i in labels:
            if i not in label2id:
                label2id[i]=label_id
                label_id+=1

        id2label = {k:v for v,k in label2id.items()}

        training_data.features_labels['countvector']=countvector
        training_data.features_labels['labels']=labels
        training_data.features_labels['label2id']=label2id
        training_data.features_labels['id2label']=id2label

        joblib.dump(self.countvectorizer, COUNT_VECTOR_FEATURIZER)

class SklearnIntentClassifier(Component):

    def __init__(self):
        from sklearn.naive_bayes import MultinomialNB
        self.clf = MultinomialNB()

    def predict(self, data: Dict) -> Dict:
        label_id = self.clf.predict(data['result'])[0]
        # data['result'] = self.id2label[label_id]
        del data['result']
        data[INTENT] = self.id2label[label_id]
        return data

    def load(self):
        self.clf = joblib.load(INTENT_CLF)
        self.id2label = joblib.load(ID_LABEL)

    def train(self, training_data: TrainingData) -> None:
        x = training_data.features_labels['countvector'].toarray()
        labels = training_data.features_labels['labels']
        label2id = training_data.features_labels['label2id']
        self.id2label = training_data.features_labels['id2label']
        y=[label2id[i] for i in labels]
        # print(y)
        self.clf.fit(X=x, y=y)

        joblib.dump(self.clf, INTENT_CLF)
        joblib.dump(self.id2label, ID_LABEL)

class SpacyEntityExtractor(Component):

    def __init__(self):
        self.nlp = spacy.blank("en")
        self.ner = self.nlp.create_pipe("ner")
        self.nlp.add_pipe(self.ner)

    def train(self, training_data: TrainingData) -> None:
        ner_training_data=[]
        for example in training_data.training_examples:
            if ENTITIES in example.data:
                entities=example.data['entities'][0]
                entity_value=[]
                entity_value.append(entities['start'])
                entity_value.append(entities['end'])
                self.ner.add_label(entities['entity'])
                entity_value.append(entities['entity'])
                entity_dict={'entities':[entity_value]}
                ner_training_data.append([example.text,entity_dict])
            else:
                ner_training_data.append([example.text,{'entities':[]}])
        # print(ner_training_data)

        self.train_ner(ner_training_data)
        self.nlp.to_disk(NER_MODEL)

    def train_ner(self, ner_training_data: List[List]):
        # 开始训练
        self.nlp.begin_training()

        # 迭代10个循环
        for itn in range(10):
            # 随机化训练数据的顺序
            random.shuffle(ner_training_data)
            losses = {}

            # 将例子分为一系列批次并在上面迭代
            for batch in spacy.util.minibatch(ner_training_data, size=2):
                texts = [text for text, entities in batch]
                annotations = [entities for text, entities in batch]

                # 更新模型
                self.nlp.update(texts, annotations, losses=losses)
            print(losses)

    def predict(self, data: Dict) -> Dict:
        doc = self.nlp(data['text'])
        entities = list()
        [entities.append({ent.label_:ent.text}) for ent in doc.ents]
        data[ENTITIES]=entities
        return data

    def load(self):
        self.nlp = self.nlp.from_disk(NER_MODEL)

component_classes = [
    WhitespaceTokenizer,
    CountVectorsFeaturizer,
    SklearnIntentClassifier,
    SpacyEntityExtractor,
]

registered_components = {c.name: c for c in component_classes}

class Trainer:
    """Trainer will load the data and train all components.

    Requires a pipeline specification and configuration to use for
    the training.
    """

    def __init__(
        self,
        cfg: Dict,
    ):

        self.config = cfg
        self.training_data = None  # type: Optional[TrainingData]

        # build pipeline
        self.pipeline = self._build_pipeline(cfg)

    def _build_pipeline(
        self, cfg: Dict,
    ) -> List[Component]:
        """Transform the passed names of the pipeline components into classes."""

        pipeline = []

        for i in cfg['pipeline']:
            component_name=i['name']
            component=registered_components[component_name]()
            pipeline.append(component)

        return pipeline

    def train(self, data: TrainingData):
        """Trains the underlying pipeline using the provided training data."""
        self.training_data = data
        # data gets modified internally during the training - hence the copy
        working_data = copy.deepcopy(data)
        [component.train(working_data) for component in self.pipeline]

    def predict(self, data: Text) -> Dict:
        result_dict=dict()
        result=copy.deepcopy(data)
        result_dict['text']=data
        result_dict['result']=result
        for component in self.pipeline:
            result_dict=component.predict(result_dict)
        return result_dict

    def load(self):
        [component.load() for component in self.pipeline]

if __name__ == "__main__":
    def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
        """Loads a single training data file from disk."""

        reader = MarkdownReader()
        return reader.read(filename)

    def _read_yml_file(filename: Text) -> Dict:
        import yaml
        with open(filename) as f:
            file_data=f.read()
        data=yaml.load(file_data, Loader=yaml.FullLoader)
        return data

    print("**module test**")
    training_data=_load('../nlu.md')
    nlu_config=_read_yml_file('../config.yml')
    trainer=Trainer(nlu_config)
    trainer.train(training_data)
    trainer.load()
    # text = 'can i get [swedish](cuisine) food in any area'
    texts = ['hi','can i get food in any area','what about indian food',
        'i want to seat outside','2 people','you cant help me','yeah']
    for text in texts:
        r=trainer.predict(text)
        print(r)