---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
Cell In[9], line 5
      3     # Test basic chat
      4 messages = [HumanMessage(content="Hello, how are you?")]
----> 5 response = llm.invoke(messages)
      6 print(response.content)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\language_models\chat_models.py:331, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
    319 @override
    320 def invoke(
    321     self,
   (...)    326     **kwargs: Any,
    327 ) -> BaseMessage:
    328     config = ensure_config(config)
    329     return cast(
    330         "ChatGeneration",
--> 331         self.generate_prompt(
    332             [self._convert_input(input)],
    333             stop=stop,
    334             callbacks=config.get("callbacks"),
    335             tags=config.get("tags"),
    336             metadata=config.get("metadata"),
    337             run_name=config.get("run_name"),
    338             run_id=config.pop("run_id", None),
    339             **kwargs,
    340         ).generations[0][0],
    341     ).message

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\language_models\chat_models.py:894, in BaseChatModel.generate_prompt(self, prompts, stop, callbacks, **kwargs)
    885 @override
    886 def generate_prompt(
    887     self,
   (...)    891     **kwargs: Any,
    892 ) -> LLMResult:
    893     prompt_messages = [p.to_messages() for p in prompts]
--> 894     return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\language_models\chat_models.py:719, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
    716 for i, m in enumerate(messages):
    717     try:
    718         results.append(
--> 719             self._generate_with_cache(
    720                 m,
    721                 stop=stop,
    722                 run_manager=run_managers[i] if run_managers else None,
    723                 **kwargs,
    724             )
    725         )
    726     except BaseException as e:
    727         if run_managers:

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\language_models\chat_models.py:960, in BaseChatModel._generate_with_cache(self, messages, stop, run_manager, **kwargs)
    958 else:
    959     if inspect.signature(self._generate).parameters.get("run_manager"):
--> 960         result = self._generate(
    961             messages, stop=stop, run_manager=run_manager, **kwargs
    962         )
    963     else:
    964         result = self._generate(messages, stop=stop, **kwargs)

File c:\Users\AL58379\Downloads\codes_base\GRIP_AI_Agent\gripbackend\ai\chatbot\horizon_dev\temp_horizon.py:133, in TextChatCompletionsLLM._generate(self, messages, stop, run_manager, **kwargs)
    123 # Use the original __call__ method
    124 response_content = self.__call__(
    125     payload=payload,
    126     files=None,
   (...)    130     stream=False
    131 )
--> 133 message = AIMessage(content=response_content)
    134 generation = ChatGeneration(message=message)
    136 return ChatResult(generations=[generation])

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\messages\ai.py:184, in AIMessage.__init__(self, content, **kwargs)
    175 def __init__(
    176     self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    177 ) -> None:
    178     """Pass in content as positional arg.
    179 
    180     Args:
    181         content: The content of the message.
    182         kwargs: Additional arguments to pass to the parent class.
    183     """
--> 184     super().__init__(content=content, **kwargs)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\messages\base.py:78, in BaseMessage.__init__(self, content, **kwargs)
     70 def __init__(
     71     self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
     72 ) -> None:
     73     """Pass in content as positional arg.
     74 
     75     Args:
     76         content: The string contents of the message.
     77     """
---> 78     super().__init__(content=content, **kwargs)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langchain_core\load\serializable.py:130, in Serializable.__init__(self, *args, **kwargs)
    128 def __init__(self, *args: Any, **kwargs: Any) -> None:
    129     """"""  # noqa: D419
--> 130     super().__init__(*args, **kwargs)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\pydantic\main.py:253, in BaseModel.__init__(self, **data)
    251 # `__tracebackhide__` tells pytest and some other tools to omit this function from tracebacks
    252 __tracebackhide__ = True
--> 253 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    254 if self is not validated_self:
    255     warnings.warn(
    256         'A custom validator is returning a value other than `self`.\n'
    257         "Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\n"
    258         'See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
    259         stacklevel=2,
    260     )

ValidationError: 2 validation errors for AIMessage
content.str
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
    For further information visit https://errors.pydantic.dev/2.11/v/string_type
content.list[union[str,dict[any,any]]]
  Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
    For further information visit https://errors.pydantic.dev/2.11/v/list_type
