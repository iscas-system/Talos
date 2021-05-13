import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm

def execute_autotune_task(template_name, dtype, N, L, M, target, n_trial, number, log_path):
    task = autotvm.task.create(template_name, args=(N, L, M, dtype), target=target)
    print(task.config_space)
    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=number))
    # Begin tuning with RandomTuner, log records to file `matmul.log`
    # You can use alternatives like XGBTuner.
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
        n_trial=n_trial,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_path)],
    )
