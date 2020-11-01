from typing import Any, Text, Optional, Tuple, List, Dict, Match
import re

import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

class StoryReader:
    """Helper class to read a story file."""

    def __init__(
        self,
    ) -> None:
        self.story_steps = []
        self.current_story_part = None
        self.current_story = []

    def read_from_file(self, filename: Text,):
        raise NotImplementedError

    def _new_story_part(self, name: Text):
        self.current_story_part = name
        self.story_steps.append(self.current_story) 
        self.current_story = []

    def _add_event(self, event_name: Text):
        self.current_story.append(event_name)

    def _add_user_intent(self, user_intent: Text):
        self.current_story.extend(['bot_listen',user_intent])

    def _add_current_stories_to_result(self):
        self.story_steps.append(self.current_story)
        self.current_story=[]

class MarkdownStoryReader(StoryReader):
    """Class that reads the core training data in a Markdown format"""

    ##读取core训练数据，返回一个记录对话状态(机器动作，用户意图)的list
    def read_from_file(self, filename: Text):
        """Given a md file reads the contained stories."""

        try:
            with open(filename, "r", encoding='utf-8') as f:
                lines = f.readlines()

            return self._process_lines(lines)
        except ValueError as err:
            file_info = "Invalid story file format. Failed to parse '{}'".format(
                os.path.abspath(filename)
            )
            logging.error(file_info)
            raise

    def _replace_template_variables(self, line: Text) -> Text:
        '''remove space at line head'''
        return re.sub(r'^[ ]*', '', line)

    def _process_lines(self, lines: List[Text]):

        for idx, line in enumerate(lines):
            line_num = idx + 1
            try:
                line = self._replace_template_variables(line)
                if line.strip() == "":
                    continue
                elif line.startswith("#"):
                    # reached a new story block
                    name = line[1:].strip("# ")
                    self._new_story_part(name)
                elif line.startswith("-"):
                    # reached a slot, event, or executed action
                    event_name = line[1:].strip()
                    self._add_event(event_name)
                elif line.startswith("*"):
                    # reached a user message
                    user_intent = line[1:].strip()
                    self._add_user_intent(user_intent)
                else:
                    # reached an unknown type of line
                    logging.warning(
                        f"Skipping line {line_num}. "
                        "No valid command found. "
                        f"Line Content: '{line}'"
                    )
            except Exception as e:
                msg = f"Error in line {line_num}: {e}"
                logging.error(msg)  # pytype: disable=wrong-arg-types
                raise ValueError(msg)
        self._add_current_stories_to_result()
        return self.story_steps

if __name__ == "__main__":
    def _load(filename: Text):
        """Loads a single training data file from disk."""

        reader = MarkdownStoryReader()
        return reader.read_from_file(filename)

    print("**module test**")
    training_data=_load('../stories.md')
    print(training_data)












