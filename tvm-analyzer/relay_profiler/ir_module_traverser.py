import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import memory_usage
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient

class op_graph:
    def __init__(self):
        self.dictionary = {}
        self.var_nodes = []
        self.call_nodes = []
    
    def insert_op(self, current_op_node):
        if self.find_if_exist(current_op_node) == None:
            self.dictionary[current_op_node.id] = current_op_node
            if current_op_node.type == "call":
                self.call_nodes.append(current_op_node)
            if current_op_node.type == "var":
                self.var_nodes.append(current_op_node)
    
    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True 
            for key in temp_op.prior.keys():
                if temp_op.prior[key][1].type == "call":
                    if_started = False
                    break
            if if_started == True:
                result.append(temp_op)
        return result

    def find_if_exist(self, current_op_node):
        for key in self.dictionary.keys():
            if key == current_op_node.id:
                return self.dictionary[key]
        return None

    def find_if_var(self, var_op):
        for temp in self.var_nodes:
            if temp.op_instance == var_op:
                # print("find same params")
                return temp
        return None

    def print_graph(self, ir_params, x):
        # verify is there any replication.
        start_call_nodes = self.find_starting_ops()
        print("Total start call node with ops:", len(start_call_nodes))
        for start_op in start_call_nodes:
            start_op.print_self()
        fw_m = profile_forward_relay_operator(start_call_nodes[0], ir_params, x)
        bw_m = profile_backward_relay_operator(start_call_nodes[0], ir_params, x)
        print("bw_m-fw_m", bw_m-fw_m)
    
    def check_op_ready(self, temp_op):
        for key in temp_op.prior.keys():
            if temp_op.prior[key][1].type == "call":
                if temp_op.prior[key][1].performance_data["fw_value"] is None:
                    return False
        return True
    
    def traverse_and_calculate_per_op(self, ir_params, x, fw=True, bw=False):
        available_op_queue = self.find_starting_ops()
        while len(available_op_queue) >0 :
            temp_op = available_op_queue.pop(0)
            if fw:
                profile_forward_relay_operator(temp_op, ir_params, x)
            if bw:
                bw_m = profile_backward_relay_operator(temp, ir_params, x)
            for key in temp_op.next.keys():
                if self.check_op_ready(temp_op.next[key][1]):
                    available_op_queue.append(temp_op.next[key][1])

class op_node:
    # args_index:{id,}
    def __init__(self, type, current_index, op, attrs = None):
        self.type = type
        if self.type == "var":
            self.name = op.name_hint
        if self.type == "call":
            self.name = op.op.name
            self.attrs = attrs
        self.id = self.name + "-" + str(current_index)
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = {}
    
    def set_next(self, next_op_node, next_op_index, args_index):
        next_op_id = next_op_node.name + "-" + str(next_op_index)
        self.next[next_op_id] = (args_index, next_op_node)
        next_op_node.prior[self.id] = (args_index, self)
    
    def print_self(self):
        print(self.id, self.attrs, self.next, self.prior)

op_index = 0
computation_graph = op_graph()

def construct_op_graph(ir_module, ir_params, x):
    global op_index, computation_graph
    print("current entrance function length:", len(ir_module.functions.items()))
    # print(ir_module)
    entrance_tuple = ir_module.functions.items()[0]
    global_var = entrance_tuple[0]
    main_function = entrance_tuple[1]
    for each_param in main_function.params:
        temp_op_node = op_node("var", op_index, each_param, attrs = None)
        computation_graph.insert_op(temp_op_node)
        op_index+=1
    recursive_traverse_op(main_function.body.attrs, main_function.body.args, temp_op=main_function.body)
    # verify whether the graph is right
    computation_graph.print_graph(ir_params, x)

def recursive_traverse_op(attrs, args, temp_op=None):
    global op_index, computation_graph
    # print(name,attrs)
    args_index = 0
    next_op_node = op_node("call", op_index, temp_op, attrs = attrs)
    op_index+=1
    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            current_node_op = recursive_traverse_op(each_arg.attrs, each_arg.args, temp_op = each_arg)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            args_index+=1
        if isinstance(each_arg,tvm.relay.expr.Var):
            current_node_op = computation_graph.find_if_var(each_arg)
            if current_node_op == None:
                current_node_op = op_node("var", op_index, each_arg, attrs = None)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            computation_graph.insert_op(current_node_op)
            args_index+=1
    computation_graph.insert_op(next_op_node)
    return next_op_node

def op_point():
    print("start op point")
    return

def get_op_args(ready_op_node, dtype):
    intermeidiate_args = []
    args_index = 0
    pick_one = True
    max_index = 0
    for key in ready_op_node.prior.keys():
        if max_index < ready_op_node.prior[key][0]:
            max_index = ready_op_node.prior[key][0]
    while args_index <= max_index:
        for key in ready_op_node.prior.keys():
            if ready_op_node.prior[key][0] == args_index:
                if ready_op_node.prior[key][1].type == "call":
                    intermeidiate_args.append(ready_op_node.prior[key][1].performance_data["fw_value"].astype(dtype))
                if ready_op_node.prior[key][1].type == "var":
                    # do not need to reinsert
                    # intermeidiate_args.append(
                    #     ready_op_node.prior[key][1].op_instance)
                    print("Find var and just neglect it.")
                args_index+=1
    return intermeidiate_args

def profile_forward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    if ready_op_node.type == "var":
        return
    call_body = ready_op_node.op_instance
    call_function = tvm.relay.Function(ready_op_node.op_instance.args, call_body)
    call_functions = {"GlobalVar": None, "main": call_function}
    call_ir_module = tvm.ir.IRModule(functions=call_functions)
    # print(call_ir_module)
    with tvm.transform.PassContext(opt_level=1):
        call_interpreter = relay.build_module.create_executor("graph", call_ir_module, tvm.cpu(0), "llvm")
    call_intput_args = []
    call_intput_args.append(tvm.nd.array(x.astype(dtype)))
    call_intput_args = call_intput_args + get_op_args(ready_op_node, dtype)
    start_memory = max(memory_usage(op_point))
    def op_forward():
        tvm_output = call_interpreter.evaluate()(*call_intput_args, **ir_params)
        print("type(tvm_output)")
        print(type(tvm_output))
        ready_op_node.performance_data["fw_value"] = tvm_output
        # print(tvm_output)
    end_memory = max(memory_usage(op_forward))
    print(end_memory - start_memory)
    ready_op_node.performance_data["fw_memory"] = end_memory - start_memory
    return end_memory - start_memory

def profile_backward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    if ready_op_node.type == "var":
        return
    call_body = ready_op_node.op_instance
    call_function = tvm.relay.Function(ready_op_node.op_instance.args, call_body)
    call_function = run_infer_type(call_function)
    bwd_func = run_infer_type(gradient(call_function))
    # compile in a different way:
    call_interpreter = relay.create_executor(device = tvm.cpu(0), target = "llvm")
    # prepare to run
    call_intput_args = []
    call_intput_args.append(tvm.nd.array(x.astype(dtype)))
    call_intput_args = call_intput_args + get_op_args(ready_op_node, dtype)
    start_memory = max(memory_usage(op_point))
    def op_forward():
        op_res  = call_interpreter.evaluate(bwd_func)(*call_intput_args, **ir_params)
        # print(tvm_output)
        ready_op_node.performance_data["bw_value"] = op_res
    end_memory = max(memory_usage(op_forward))
    print(end_memory - start_memory)
    ready_op_node.performance_data["bw_memory"] = end_memory - start_memory
    return end_memory - start_memory