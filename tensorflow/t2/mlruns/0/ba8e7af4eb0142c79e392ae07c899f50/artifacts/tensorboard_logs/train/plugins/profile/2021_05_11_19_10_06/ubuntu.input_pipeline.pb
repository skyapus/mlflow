	y$^??@y$^??@!y$^??@	X?3??@X?3??@!X?3??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y$^??@b?G??A?Hڍ>?@Y???????*	/?$?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?72?????!?xZO?=W@)????c???1V?NwRS@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?h???c???!??] ?X/@)h???c???1??] ?X/@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???r???!?_r'??@)???r???1?_r'??@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism ?????!p????T@)???S? ??1???r??:Preprocessing2F
Iterator::Model?r۾G???!?pX
?&@)#h?$?o?1?Þ?5??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9X?3??@I<?`?Y
X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b?G??b?G??!b?G??      ??!       "      ??!       *      ??!       2	?Hڍ>?@?Hڍ>?@!?Hڍ>?@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JCPU_ONLYYX?3??@b q<?`?Y
X@