	?{???@?{???@!?{???@	?Ǟ?D????Ǟ?D???!?Ǟ?D???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?{???@?I}Yک??AYک?ܰ@Y-ͭVc??*	+??ձ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2r?Md?@!??Â?X@)a?4?}@1T|ڽ?W@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?h?$????!????`?@)h?$????1????`?@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?h?V???!=|????)?[ A???1*K?1L??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch7QKs+???!?̻?s??)7QKs+???1?̻?s??:Preprocessing2F
Iterator::Modelm ]lZ)??!1?{O??)4?????i?1F?t>?ı?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Ǟ?D???I??M?&?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?I}Yک???I}Yک??!?I}Yک??      ??!       "      ??!       *      ??!       2	Yک?ܰ@Yک?ܰ@!Yک?ܰ@:      ??!       B      ??!       J	-ͭVc??-ͭVc??!-ͭVc??R      ??!       Z	-ͭVc??-ͭVc??!-ͭVc??b      ??!       JCPU_ONLYY?Ǟ?D???b q??M?&?X@