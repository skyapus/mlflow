	???)?!@???)?!@!???)?!@	-?%?^??-?%?^??!-?%?^??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???)?!@??96??A?M?G?_@Y?Z?7?q??*	?ʡEV??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2"ĕ??@!?}?p?X@)bJ$??X@1L?ƫ?W@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?O??'????!%?l???@)O??'????1%?l???@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch::ZՒ??!(?c??)::ZՒ??1(?c??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?m?\p??!Z?|5[3??)?Mc{-???1?`??????:Preprocessing2F
Iterator::Model??1?????!Ӕ???c??)?uʣk?1?g:F<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9-?%?^??I??kK??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??96????96??!??96??      ??!       "      ??!       *      ??!       2	?M?G?_@?M?G?_@!?M?G?_@:      ??!       B      ??!       J	?Z?7?q???Z?7?q??!?Z?7?q??R      ??!       Z	?Z?7?q???Z?7?q??!?Z?7?q??b      ??!       JCPU_ONLYY-?%?^??b q??kK??X@