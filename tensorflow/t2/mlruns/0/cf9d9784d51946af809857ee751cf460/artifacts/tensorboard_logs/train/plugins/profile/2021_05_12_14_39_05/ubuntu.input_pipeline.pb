	`;?O?@`;?O?@!`;?O?@	??uh??@??uh??@!??uh??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`;?O?@?2o?u???A?????@Y5?Ry;???*	?v?????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2 ?Ȓ9?@!?R=??qX@)?Q?}	@1?R??^R@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??#0????!??7]zM8@)?#0????1??7]zM8@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism#2???̷?!]E?:&@)B]¡??1???^(???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchcD?в???!\S?k???)cD?в???1\S?k???:Preprocessing2F
Iterator::Modelh"lxz???!F?U???@)????k?1?D
????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??uh??@I	P??a!X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?2o?u????2o?u???!?2o?u???      ??!       "      ??!       *      ??!       2	?????@?????@!?????@:      ??!       B      ??!       J	5?Ry;???5?Ry;???!5?Ry;???R      ??!       Z	5?Ry;???5?Ry;???!5?Ry;???b      ??!       JCPU_ONLYY??uh??@b q	P??a!X@