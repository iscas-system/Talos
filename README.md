# Talos

Talos is a dataflow analysis and scheduling tool for deep learning applications.

# operator speedup analysis for pytorch

[click here to see pytorch operator chrome tracing](pytorch-analyzer)

# operator speedup analysis for TensorFlow

[click here to see tensorflow operator chrome tracing](tensorflow-analyzer)

# operator speedup aware scheduling

[click here to see operator speedup awareness scheudling](java-operator-scheduler)

# road map

1. Select a proper deep learning framework to that can easily control operation location.
2. Modify an existing framework to support operator speedup ratio awareness scheduling.
3. integrate existing data or model parallel mechanisms to make the scheduling practical.
4. optimize existing device placement granularity and strategy. 