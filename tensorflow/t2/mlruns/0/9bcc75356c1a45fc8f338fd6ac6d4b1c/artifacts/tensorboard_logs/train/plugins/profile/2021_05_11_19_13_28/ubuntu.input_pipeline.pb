	??.??????.????!??.????	~+E\V?@~+E\V?@!~+E\V?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??.???? ?#G:??A??%VF???Y?(?7ӱ?*	a??"?ُ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2"nN%@??!????W@)?qs*???1:?m?~?J@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?|??c?M??!b,ʺ??D@)|??c?M??1b,ʺ??D@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?#EdXś?!`OD?_I@)?#EdXś?1`OD?_I@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?[Ɏ?@??!ꣂ@)(?r?w??1?|?D???:Preprocessing2F
Iterator::Model??+??إ?! ?A?¾@)??(&o?i?1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9}+E\V?@I??M?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?#G:?? ?#G:??! ?#G:??      ??!       "      ??!       *      ??!       2	??%VF?????%VF???!??%VF???:      ??!       B      ??!       J	?(?7ӱ??(?7ӱ?!?(?7ӱ?R      ??!       Z	?(?7ӱ??(?7ӱ?!?(?7ӱ?b      ??!       JCPU_ONLYY}+E\V?@b q??M?X@