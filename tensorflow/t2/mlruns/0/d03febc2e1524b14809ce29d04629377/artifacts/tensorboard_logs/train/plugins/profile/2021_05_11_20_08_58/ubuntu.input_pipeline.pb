	??'I?@??'I?@!??'I?@	?goo.@?goo.@!?goo.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??'I?@4M?~2???A.rOW @Y??8?#??*	? ?rhχ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?j+?????!?+F}rW@)?.R(_??1$??f??Q@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??d73???!4??Y??6@)?d73???14??Y??6@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\[%X??![?o???@)\[%X??1[?o???@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism6??\??!??8?@)?ю~7??1L0???:Preprocessing2F
Iterator::Model;ŪA??!?@?+(?@)`???Yn?1????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?goo.@I?		??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	4M?~2???4M?~2???!4M?~2???      ??!       "      ??!       *      ??!       2	.rOW @.rOW @!.rOW @:      ??!       B      ??!       J	??8?#????8?#??!??8?#??R      ??!       Z	??8?#????8?#??!??8?#??b      ??!       JCPU_ONLYY?goo.@b q?		??W@