# Copyright (c) 2012-2013 Oleksandr Sviridenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from les import backend_solvers
from les.executors import executor_base
from les.executors.executor_base import Result
from les.utils import logging

class Error(Exception):
  pass

class DummyExecutor(executor_base.ExecutorBase):
  '''Dummy executor doesn't know how to parallelize solving process. It
  simply solves models one by one in order they come.

  :param pipeline: A :class:`~les._pipeline.Pipeline` instance.
  '''

  def __init__(self, pipeline):
    executor_base.ExecutorBase.__init__(self, pipeline)

  def run(self):
    for task in self._pipeline:
      result = self.execute(task)
      if result is None:
        self._pipeline.finalize_task(task)
        continue
      self._pipeline.process_result(result)

  @classmethod
  def execute(cls, task):
    logging.debug('Solve model %s with solver %s',
                  str(task.get_model_parameters()), task.get_solver_id())
    solver = backend_solvers.get_instance_of(task.get_solver_id())
    if not solver:
      raise Error('Cannot instantiate backend solver by id: %d' %
                  task.get_solver_id())
    try:
      solver.load_model_params(task.get_model_parameters())
      solver.solve()
    except Exception, e:
      # TODO: send back a report.
      logging.exception('Cannot execute given task: cannot solve the model.')
      return None
    return Result(task.get_id(), solver.get_solution())
