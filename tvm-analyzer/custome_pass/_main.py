import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, runtime, topi
from tvm.auto_scheduler import _ffi_api
from tvm.topi.utils import get_const_tuple
from tvm.topi.sparse.utils import random_bsr_matrix

@auto_scheduler.register_workload
def sparse_dense(M, N, K, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=(M, K), dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")
    B = te.placeholder(shape=(M, N), dtype=dtype)

    out = topi.nn.sparse_dense(topi.nn.relu(X), W_data, W_indices, W_indptr)
    out = te.compute((M, N), lambda i, j: out[i, j] + B[i, j], name="BiasAdd")
    out = topi.nn.relu(out)

    return [X, W_data, W_indices, W_indptr, B, out]

M = 128
K = 256
N = 512
BS_R = 16
BS_C = 1
density = 0.6

X_np = np.random.randn(M, K).astype("float32")
X_np = np.maximum(np.zeros((M, K), dtype="float32"), X_np)  # Relu
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T  # Process the matrix multiplication
B_np = np.random.randn(M, N).astype("float32")
Y_np = Y_np + B_np  # Bias add
Y_np = np.maximum(np.zeros((M, N), dtype="float32"), Y_np)  # Relu

tvm.target.Target(target="llvm", host="llvm")

# Register the sparse data to task inputs
prefix = "sparse_dense_bsr_%d_%d_%d_%d_%.2f_" % (N, K, BS_R, BS_C, density)
task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense,
    args=(M, N, K, W_sp_np.data.shape, W_sp_np.indices.shape, W_sp_np.indptr.shape, "float32"),
    target=target,
    task_inputs={
        prefix + "W_data": runtime.ndarray.array(W_sp_np.data),
        prefix + "W_indices": runtime.ndarray.array(W_sp_np.indices),
        prefix + "W_indptr": runtime.ndarray.array(W_sp_np.indptr),
    },
    task_inputs_save_to_file=True,
)

def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm",
        "sparse_dense_sp_rhs_bsrmm_block",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS


def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    # Set the default consumer of compute block
    consumer = sparse_dense

    # If sparse dense has a single elementwise consumer
    # We can compute inline the sparse_dense output stage
    consumers = _ffi_api.SearchPolicyUtilsGetConsumers(
        search_policy.search_task, s0.state_object, stage_id
    )
    if len(consumers) == 1:
        consumer_id = int(consumers.items()[0][0])
        if _ffi_api.SearchPolicyUtilsIsElementwiseMatch(
            search_policy.search_task, s0.state_object, stage_id, consumer_id
        ):
            consumer = s0.stages[consumer_id].op
            s0.compute_inline(sparse_dense)

    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[consumer].iters
    i0, i1, i2 = s0.split(sparse_dense_block, i, [None, None])
    m0, m1 = s0.follow_split(consumer, m, len(s0.transform_steps) - 1, 1)
    j0, j1 = s0.split(sparse_dense_block, nb_j, [None])
    n0, n1 = s0.follow_split(consumer, n, len(s0.transform_steps) - 1, 1)
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, row_offset, i2, j, c])
    s0.reorder(consumer, [m0, n0, m1, n1])
    s0.compute_at(sparse_dense_block, consumer, n0)

    ret.append([s0.state_object, stage_id - 2])

    return ret

log_file = "/root/temp/sparse_dense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ],
)

task.tune(tune_option, search_policy)

# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

@tvm.tir.transform.prim_func_pass(opt_level=2)
def transform(func, mod, ctx):
    # my transformations here.
    print(func is None)
    print(tvm.tir.analysis.calculate_workspace_bytes(func))
    return func

with tvm.transform.PassContext(config={"tir.add_lower_pass": [(3, transform)]}):
    print(tvm.lower(schedule, args, simple_mode=True))

execute_autotune.execute_autotune_task("tutorial/matmul", "float32", 1024, 1024, 1024, tvm.target.Target(target="llvm", host="llvm"), 10, 5, "/root/temp/matmul.log")

# matmul_module = compile_operator.compile_with_autune_log("/root/temp/matmul.log", schedule, args, tvm.target.Target(target="llvm", host="llvm"), "float32")

# print(type(matmul_module))