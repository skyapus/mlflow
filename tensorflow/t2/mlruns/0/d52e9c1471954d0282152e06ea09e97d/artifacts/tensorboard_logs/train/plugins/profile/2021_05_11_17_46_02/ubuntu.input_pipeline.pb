	y?&1?@y?&1?@!y?&1?@	~?|???@~?|???@!~?|???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$y?&1?@???&????A??1??#@Y?-??T??*?|?5^?@)      @=2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2%???7??!?(??W@)?c> ???1??~ B?Q@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle???Pk?w??!?X??K?7@)??Pk?w??1?X??K?7@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??????!z????D@)??????1z????D@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismۆQ<???!9?O8,@)
?_??͈?1j??''??:Preprocessing2F
Iterator::Model?VC?K??!P2~ݱ?@)??-?h?1???٘'??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?|???@I?lz?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???&???????&????!???&????      ??!       "      ??!       *      ??!       2	??1??#@??1??#@!??1??#@:      ??!       B      ??!       J	?-??T???-??T??!?-??T??R      ??!       Z	?-??T???-??T??!?-??T??b      ??!       JCPU_ONLYY?|???@b q?lz?X@