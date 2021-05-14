import onnx
import tvm
from PIL import Image
import numpy as np
from tvm.contrib.download import download_testdata
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner

def xgb_tune_module(target, params, mod, number=10, repeat=1, min_repeat_ms=0, timeout=10, tuner="xgb", trials=10, early_stopping=100, tuning_records="/root/temp/tvm-autojson/resnet-50-v2-autotuning.json"):
    runner = tvm.autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
    )
    tuning_option = {
        "tuner": tuner,
        "trials": trials,
        "early_stopping": 100,
        "measure_option": tvm.autotvm.measure_option(
            builder=tvm.autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": tuning_records,
    }
    # begin by extracting the taks from the onnx model
    tasks = tvm.autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                tvm.autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                tvm.autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )
    return tuning_option