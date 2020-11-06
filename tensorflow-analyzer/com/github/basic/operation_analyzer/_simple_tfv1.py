import tensorflow as tf
from tensorflow.python.client import timeline

from com.github.basic.util.AnalysisHelper import get_abs_base_path

tf.compat.v1.disable_eager_execution()
#构造计算图
x = tf.compat.v1.random_normal([1000, 1000])
y = tf.compat.v1.random_normal([1000, 1000])
res = tf.matmul(x, y)
#运行计算图, 同时进行跟踪
with tf.compat.v1.Session() as sess:
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    sess.run(res, options=run_options, run_metadata=run_metadata)

    #创建Timeline对象,并将其写入到一个json文件
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(get_abs_base_path("",'timeline.json'), 'w') as f:
        f.write(ctf)