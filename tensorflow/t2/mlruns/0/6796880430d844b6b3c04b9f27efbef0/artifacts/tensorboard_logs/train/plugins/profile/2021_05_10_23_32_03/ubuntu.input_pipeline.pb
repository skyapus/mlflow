	H?I?O?@H?I?O?@!H?I?O?@	R?6̻s@R?6̻s@!R?6̻s@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$H?I?O?@??;????A????@Y?\R????*	R????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?B??f??!?善??W@)^?????1??9?P@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??Dׅ???!?*O?@)?Dׅ???1?*O?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?4ӽN???!?DVm{@)?4ӽN???1?DVm{@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism!??^?!)?? x@)?? {??1$;8)???:Preprocessing2F
Iterator::Model?/???"??!j???g?@):vP??h?1$??l???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9R?6̻s@I;?<C??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??;??????;????!??;????      ??!       "      ??!       *      ??!       2	????@????@!????@:      ??!       B      ??!       J	?\R?????\R????!?\R????R      ??!       Z	?\R?????\R????!?\R????b      ??!       JCPU_ONLYYR?6̻s@b q;?<C??W@