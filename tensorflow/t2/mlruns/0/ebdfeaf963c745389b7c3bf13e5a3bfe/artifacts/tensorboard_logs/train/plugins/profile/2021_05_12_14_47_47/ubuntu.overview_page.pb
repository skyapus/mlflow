?	b?A
??@b?A
??@!b?A
??@	?X???@?X???@!?X???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$b?A
??@??3?????A???ė @YK[\?3ٷ?*	/?$ӌ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??̔????!)????W@)T???f??1????u?S@:Preprocessing2z
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??'???I??!?)?G?V0@)?'???I??1?)?G?V0@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???ާ???!(?D!?;@)???ާ???1(?D!?;@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??*??]??!'?v?m?@)r?Pi?̎?1J?Q?\??:Preprocessing2F
Iterator::Model??N??D??!m?2???@)N?=??jn?1bT?;R???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?X???@IsJ?CR?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??3???????3?????!??3?????      ??!       "      ??!       *      ??!       2	???ė @???ė @!???ė @:      ??!       B      ??!       J	K[\?3ٷ?K[\?3ٷ?!K[\?3ٷ?R      ??!       Z	K[\?3ٷ?K[\?3ٷ?!K[\?3ٷ?b      ??!       JCPU_ONLYY?X???@b qsJ?CR?W@Y      Y@q?1?-?@"?
both?Your program is POTENTIALLY input-bound because 6.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 