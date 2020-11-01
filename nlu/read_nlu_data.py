import re
from typing import Any, Text, Optional, Tuple, List, Dict, Match

from training_data import Message,TrainingData

##匹配[]()里的内容，标记为entity_text和entity，可以dict的格式返回
entity_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+?)\](\((?P<entity>[^:)]+?)(?:\:(?P<value>[^)]+))?\))"
)

##匹配实体的标注格式[]()
item_regex = re.compile(r"\s*[-*+]\s*(.+)")

##多行匹配注释<!-- --!>
comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)

##封装open方法
class TrainingDataReader:
    def _read_file(self, filename: Text, encoding: Text = 'utf-8') -> Any:
        """Read text from a file."""
        try:
            with open(filename, encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"File '{filename}' does not exist.")

    def read(self, filename: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a file."""
        return self.reads(self._read_file(filename), **kwargs)

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a string."""
        raise NotImplementedError

##继承父类open方法读取md文件
class MarkdownReader(TrainingDataReader):
    """Reads markdown training data and creates a TrainingData object."""

    def __init__(self) -> None:
        self.current_title = None
        self.current_section = None
        self.training_examples = []

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Read markdown string and create TrainingData object"""
        """读取训练数据中的意图和实体"""
        s = self._strip_comments(s)
        for line in s.splitlines():
            line = line.strip()
            header = self._find_section_header(line)
            if header:
                self._set_current_section(header[0], header[1])
            else:
                self._parse_item(line)

        return TrainingData(
            self.training_examples,
        )
    """静态方法，可以实例化调用也可以不实例化调用"""
    @staticmethod
    def _strip_comments(text: Text) -> Text:
        """ Removes comments defined by `comment_regex` from `text`. """
        return re.sub(comment_regex, "", text)

    """函数如果返回多个值，会封装成tuple的形式，可通过下标取值"""
    @staticmethod
    def _find_section_header(line: Text) -> Optional[Tuple[Text, Text]]:
        """Checks if the current line contains a section header
        and returns the section and the title."""
        """匹配nlu数据的开头，分组后返回，如"## intent:greet"，返回intent，greet"""
        match = re.search(r"##\s*(.+?):(.+)", line)
        if match is not None:
            return match.group(1), match.group(2)

        return None

    def _set_current_section(self, section: Text, title: Text) -> None:
        """Update parsing mode."""
        """记录当前的section和title，如intent, greet"""
        self.current_section = section
        self.current_title = title

    def _parse_item(self, line: Text) -> None:
        """Parses an md list item line based on the current section type."""
        match = re.match(item_regex, line)
        if match:
            item = match.group(1)
            if self.current_section == 'intent':
                parsed = self.parse_training_example(item)
                self.training_examples.append(parsed)

    def parse_training_example(self, example: Text) -> "Message":
        """Extract entities and synonyms, and convert to plain text."""
        """提取实体和位置如：{'start': 36, 'end': 43, 'value': 'chinese', 'entity': 'cuisine'}"""
        entities = self._find_entities_in_training_example(example)
        """将训练数据变成正常的句子，如：[afghan](cuisine) food -> afghan food"""
        plain_text = re.sub(
            entity_regex, lambda m: m.groupdict()["entity_text"], example
        )

        message = Message.build(plain_text, self.current_title)

        if len(entities) > 0:
            message.set("entities", entities)
        return message

    def _find_entities_in_training_example(self, example: Text) -> List[Dict]:
        """Extracts entities from a markdown intent example.

        Args:
            example: markdown intent example

        Returns: list of extracted entities
        """
        entities = []
        offset = 0
        """正则匹配返回一个迭代器，包括位置和内容"""
        for match in re.finditer(entity_regex, example):
            entity_attributes = self._extract_entity_attributes(match)
            """记录实体的位置"""
            start_index = match.start() - offset
            end_index = start_index + len(entity_attributes['text'])
            offset += len(match.group(0)) - len(entity_attributes['text'])

            entity = {'start':start_index,'end':end_index,
                'value':entity_attributes['text'],'entity':entity_attributes['type']}
            entities.append(entity)

        return entities

    def _extract_entity_attributes(self, match: Match) -> Dict:
        """Extract the entity attributes, i.e. type, value, etc., from the
        regex match."""
        entity_text = match.groupdict()["entity_text"]
        entity_type = match.groupdict()["entity"]

        return {'text':entity_text,'type':entity_type}

if __name__ == "__main__":
    def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
        """Loads a single training data file from disk."""

        reader = MarkdownReader()
        return reader.read(filename)

    print("**module test**")
    training_data=_load('../nlu.md')
    print(training_data.training_examples[33].text,training_data.training_examples[33].data)



