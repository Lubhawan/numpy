---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[30], line 1
----> 1 agent.invoke(input=_input)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\__init__.py:2718, in Pregel.invoke(self, input, config, stream_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, **kwargs)
   2716 else:
   2717     chunks = []
-> 2718 for chunk in self.stream(
   2719     input,
   2720     config,
   2721     stream_mode=stream_mode,
   2722     output_keys=output_keys,
   2723     interrupt_before=interrupt_before,
   2724     interrupt_after=interrupt_after,
   2725     checkpoint_during=checkpoint_during,
   2726     debug=debug,
   2727     **kwargs,
   2728 ):
   2729     if stream_mode == "values":
   2730         latest = chunk

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\__init__.py:2356, in Pregel.stream(self, input, config, stream_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, subgraphs)
   2350     # Similarly to Bulk Synchronous Parallel / Pregel model
   2351     # computation proceeds in steps, while there are channel updates.
   2352     # Channel updates from step N are only visible in step N+1
   2353     # channels are guaranteed to be immutable for the duration of the step,
   2354     # with channel updates applied only at the transition between steps.
   2355     while loop.tick(input_keys=self.input_channels):
-> 2356         for _ in runner.tick(
   2357             loop.tasks.values(),
   2358             timeout=self.step_timeout,
   2359             retry_policy=self.retry_policy,
   2360             get_waiter=get_waiter,
   2361         ):
   2362             # emit output
   2363             yield from output()
   2364 # emit output

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\runner.py:158, in PregelRunner.tick(self, tasks, reraise, timeout, retry_policy, get_waiter)
    156 t = tasks[0]
    157 try:
--> 158     run_with_retry(
    159         t,
    160         retry_policy,
    161         configurable={
    162             CONFIG_KEY_CALL: partial(
    163                 _call,
    164                 weakref.ref(t),
    165                 retry=retry_policy,
    166                 futures=weakref.ref(futures),
    167                 schedule_task=self.schedule_task,
    168                 submit=self.submit,
    169                 reraise=reraise,
    170             ),
    171         },
    172     )
    173     self.commit(t, None)
    174 except Exception as exc:

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\retry.py:39, in run_with_retry(task, retry_policy, configurable)
     37     task.writes.clear()
     38     # run the task
---> 39     return task.proc.invoke(task.input, config)
     40 except ParentCommand as exc:
     41     ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\utils\runnable.py:622, in RunnableSeq.invoke(self, input, config, **kwargs)
    620     # run in context
    621     with set_config_context(config, run) as context:
--> 622         input = context.run(step.invoke, input, config, **kwargs)
    623 else:
    624     input = step.invoke(input, config)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\__init__.py:2718, in Pregel.invoke(self, input, config, stream_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, **kwargs)
   2716 else:
   2717     chunks = []
-> 2718 for chunk in self.stream(
   2719     input,
   2720     config,
   2721     stream_mode=stream_mode,
   2722     output_keys=output_keys,
   2723     interrupt_before=interrupt_before,
   2724     interrupt_after=interrupt_after,
   2725     checkpoint_during=checkpoint_during,
   2726     debug=debug,
   2727     **kwargs,
   2728 ):
   2729     if stream_mode == "values":
   2730         latest = chunk

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\__init__.py:2356, in Pregel.stream(self, input, config, stream_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, subgraphs)
   2350     # Similarly to Bulk Synchronous Parallel / Pregel model
   2351     # computation proceeds in steps, while there are channel updates.
   2352     # Channel updates from step N are only visible in step N+1
   2353     # channels are guaranteed to be immutable for the duration of the step,
   2354     # with channel updates applied only at the transition between steps.
   2355     while loop.tick(input_keys=self.input_channels):
-> 2356         for _ in runner.tick(
   2357             loop.tasks.values(),
   2358             timeout=self.step_timeout,
   2359             retry_policy=self.retry_policy,
   2360             get_waiter=get_waiter,
   2361         ):
   2362             # emit output
   2363             yield from output()
   2364 # emit output

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\runner.py:252, in PregelRunner.tick(self, tasks, reraise, timeout, retry_policy, get_waiter)
    250 # panic on failure or timeout
    251 try:
--> 252     _panic_or_proceed(
    253         futures.done.union(f for f, t in futures.items() if t is not None),
    254         panic=reraise,
    255     )
    256 except Exception as exc:
    257     if tb := exc.__traceback__:

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\runner.py:504, in _panic_or_proceed(futs, timeout_exc_cls, panic)
    502                 interrupts.append(exc)
    503             else:
--> 504                 raise exc
    505 # raise combined interrupts
    506 if interrupts:

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\executor.py:83, in BackgroundExecutor.done(self, task)
     81 """Remove the task from the tasks dict when it's done."""
     82 try:
---> 83     task.result()
     84 except GraphBubbleUp:
     85     # This exception is an interruption signal, not an error
     86     # so we don't want to re-raise it on exit
     87     self.tasks.pop(task)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\concurrent\futures\_base.py:449, in Future.result(self, timeout)
    447     raise CancelledError()
    448 elif self._state == FINISHED:
--> 449     return self.__get_result()
    451 self._condition.wait(timeout)
    453 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\concurrent\futures\_base.py:401, in Future.__get_result(self)
    399 if self._exception:
    400     try:
--> 401         raise self._exception
    402     finally:
    403         # Break a reference cycle with the exception in self._exception
    404         self = None

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\concurrent\futures\thread.py:59, in _WorkItem.run(self)
     56     return
     58 try:
---> 59     result = self.fn(*self.args, **self.kwargs)
     60 except BaseException as exc:
     61     self.future.set_exception(exc)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\pregel\retry.py:39, in run_with_retry(task, retry_policy, configurable)
     37     task.writes.clear()
     38     # run the task
---> 39     return task.proc.invoke(task.input, config)
     40 except ParentCommand as exc:
     41     ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\utils\runnable.py:622, in RunnableSeq.invoke(self, input, config, **kwargs)
    620     # run in context
    621     with set_config_context(config, run) as context:
--> 622         input = context.run(step.invoke, input, config, **kwargs)
    623 else:
    624     input = step.invoke(input, config)

File c:\ProgramData\Anaconda3\envs\grip-ai-agent\Lib\site-packages\langgraph\utils\runnable.py:317, in RunnableCallable.invoke(self, input, config, **kwargs)
    313 def invoke(
    314     self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    315 ) -> Any:
    316     if self.func is None:
--> 317         raise TypeError(
    318             f'No synchronous function provided to "{self.name}".'
    319             "\nEither initialize with a synchronous function or invoke"
    320             " via the async API (ainvoke, astream, etc.)"
    321         )
    322     if config is None:
    323         config = ensure_config()

TypeError: No synchronous function provided to "retrieve_columns".
Either initialize with a synchronous function or invoke via the async API (ainvoke, astream, etc.)
During task with name 'retrieve_columns' and id '805d07a8-ea72-11f9-8fb0-a51c99e537ee'
During task with name 'data_analyze_graph' and id '75b517a9-7f90-7d16-7ca2-bd52960b57f8'
