# tiny-chatbot
a tiny-chatbot for study or research

参考rasa(https://github.com/RasaHQ/rasa)
实现了一个简单的任务型对话机器人，由NLU，DM等模块组成，能识别用户的意图和实体，记录对话状态并根据对话状态决定机器的下一步动作。

这个项目是我在学习了rasa对话机器人框架源码之后，根据自己的理解实现的，旨在熟悉任务型对话机器人框架和原理，用于学习研究或者二次开发。

#### 设计思路：

###### 1.训练数据的处理：

训练数据是Markdown格式的文件nlu.md和core.md，nlu.md记录训练nlu的句子并标注意图和实体，core.md记录每一轮对话包含的用户意图和机器动作，配置文件是domain.yml和config.yml，domain.yml记录所有意图，实体和需要填充的slot，以及responses模板，config.yml配置nlu的component和DM的policies。

读取数据的方式参考了rasa的MarkdownReader，读取训练数据后，就开始训练NLU和DM。

###### 2.NLU与DM的设计和训练：

NLU部分，使用sklearn识别用户意图，用spacy提取实体，将特征化，文本分类，实体识别等组件构成pipeline，数据封装成Message对象在pipeline里传递，这样可以方便地添加和更换组件，比如想用bert做意图识别，只要实现相应的类和方法，然后把类名加到配置文件即可。

DM部分，将用户意图和机器动作作为对话状态，实现了两个策略根据历史对话状态决定机器的下一步动作，一个是memorization策略，直接记住训练数据中对话状态到机器行为的映射，另一个是embedding策略，使用transformer encoder结构将对话状态编码为一个向量，再通过线性变换和softmax转化为对应动作的概率
DM里的策略也组成了pipeline，如果memorization策略处理不了就交给embedding策略处理。

以上是NLU和DM模块的实现，现在机器已经能识别用户的意图并决定下一步的动作，只要编写对应的动作即可。

比如用户说'hello'，识别意图为'greet'，机器动作为'utter_greet'，编写该动作代码调用responses模板里'utter_greet'的回复内容并输出给用户。

###### 3.具体任务的处理：

对于需要进一步获取信息的多轮对话任务，如查询餐馆，需要获取餐馆风味，人数等信息，采用子线程加消息队列的方式执行此类任务，比如查询餐馆任务，识别用户的request_restaurant意图后，机器动作为active_restaurant_form，编写相应的动作代码，启动一个子线程名为form_function，主线程和子线程通过两个消息队列通信，主线程向消息队列q1发送'active'，子线程收到后会根据需要获取的信息生成相应的询问语句，经过消息队列q2发给主线程显示给用户。

比如用户说'can i get food in any area?'，识别意图为'request_restaurant'，执行动作active_restaurant_form，向消息队列q1发送'active'，子线程收到后查询需要填充的slot如'cuisine'，则生成询问语句'what cuisine?'，经过消息队列q2发给主线程显示给用户，用户说'chinese food'，识别意图为'inform'实体为{cuisine:chinese}，将实体发给子线程，然后子线程询问下一个slot，直到所有信息都获取到为止。

很多时候对话没有想象中那么顺利，比如一个查询餐馆的对话，用户可能提供了一个信息后出现chitchat等其它意图。

对于这种情况的处理，在询问餐馆信息的对话过程中，主线程如果识别用户的意图是inform(提供信息)，则不将该意图添加到对话状态里，将提取的信息传给子线程并接收子线程的response，因为对话状态没更新，所以此时对话策略预测的下一个action仍然是bot_listen，判断当前的action和上一个对话状态都是bot_listen则不更新对话状态，所以在询问餐馆信息的过程中，只要用户的意图是提供信息，就不更新对话状态，让机器的下一个action永远是bot_listen。

如果nlu识别用户的意图不是inform，才会用这个意图更新对话状态，如果core.md文件中有相应情景的训练数据，对话策略就能正确地回应用户意图并继续餐馆任务。

###### 4主要类说明：

class Agent:机器人类，有用于NLU的interpreter对象和DM的policy_ensemble对象，还有具体任务的子线程，主要有train()和run()两个方法。

class Interpreter:nlu解释器类，包含Trainer对象，主要有train(), predict()方法，用于读取训练数据，训练trainer，返回预测结果等。

class Trainer:根据配置文件构建pipeline并训练pipeline里的每个component，主要有train(), predict()方法。

class Component:组件类，每个具体的组件对象如WhitespaceTokenizer，CountVectorsFeaturizer，SklearnIntentClassifier等都是继承这个类，然后实现自己的train, predict等方法。

class CountVectorsFeaturizer：sklearn的countvectorfeaturizer

class SklearnIntentClassifier: sklearn的MultinomialNB分类器

class SpacyEntityExtractor：spacy实体提取

class Message:保存训练数据和每个component的处理结果

class TrainingData:保存Message

class PolicyEnsemble:和Trainer类似，根据配置文件创建和训练policy并将所有policy组成pipeline

class Policy：策略类，读取core.md文件并处理成训练对话策略的数据形式（比如增加bot_listen，把所有机器的action取出来作为要预测的内容，该action之前的状态作为训练数据等），MemoizationPolicy，PytorchPolicy都是继承这个类

class MemoizationPolicy：直接记住训练数据中对话状态到机器行为的映射

class PytorchPolicy：用pytorch实现的transformer encoder将对话状态编码为一个向量，再通过线性变换和softmax转化为对应动作的概率

#### 目录结构:

agent.py:就是Agent对象，整合nlu, dm, 子任务线程，与user交互等

nlu:包括training_data.py定义Message类和trainingdata类，read_nlu_data.py读取训练数据的markdownreader类，train_nlu.py训练nlu所有组件的Trainer类和Component类

core:包括read_core_data.py读取训练数据，MyTransformerEncoder.py用pytorch实现的transformer encoder，train_core.py各种policy的构建和训练

#### 配置&运行:

` git clone git@github.com:chengziyi/tiny-chatbot.git                           `

安装requirements.txt里的依赖

python agent.py

#### 最终效果：
![image](https://github.com/chengziyi/tiny_chatbot/blob/main/images/1.png)
![image](https://github.com/chengziyi/tiny_chatbot/blob/main/images/2.png)
