"?@
BHostIDLE"IDLE1??x?&??@A??x?&??@a???|???i???|????Unknown
?HostDataset"9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2(1L7?A???@9L7?A???@A?&1,??@I?&1,??@a??0????i\??????Unknown
jHost_FusedMatMul"model/dense/Relu(1+???4?@9+???4?@A+???4?@I+???4?@aFMwQ???i??O???Unknown
tHostMatMul" gradient_tape/model/dense/MatMul(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a?dg|z???i??sn?A???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1V-???m@9V-???m@AV-???m@IV-???m@a?U]?????i?????????Unknown
?HostDataset"BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle(?1??ʡ`@9??ʡ??A??ʡ`@I??ʡ??a??????i!tbK?_???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(17?A`?XR@97?A`?XR@A7?A`?XR@I7?A`?XR@a8?Z??;??iv?W??????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1????x9W@9????x9W@A+???7M@I+???7M@aL??\???i?9?D?#???Unknown
u	HostFlushSummaryWriter"FlushSummaryWriter(1?????KD@9?????KD@A?????KD@I?????KD@a?w/&f ~?iØ?_???Unknown?
x
HostMatMul"$gradient_tape/model/dense_1/MatMul_1(1'1??B@9'1??B@A'1??B@I'1??B@a??n/?{?i???o?????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1o??ʱB@9o??ʱB@Ao??ʱB@Io??ʱB@a?n?տ{?i???????Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1B`??";A@9B`??";A@AB`??";A@IB`??";A@a??o??y?i?gɇ+???Unknown
xHostReluGrad""gradient_tape/model/dense/ReluGrad(1=
ףp?=@9=
ףp?=@A=
ףp?=@I=
ףp?=@a[Ri?v?i???P.???Unknown
oHost_FusedMatMul"model/dense_1/BiasAdd(1?v??:@9?v??:@A?v??:@I?v??:@aK]?s(Ws?iI#???T???Unknown
iHostWriteSummary"WriteSummary(1?? ?r7@9?? ?r7@A?? ?r7@I?? ?r7@a??T[9q?i??9`/w???Unknown?
jHostSoftmax"model/dense_1/Softmax(1F???Ԙ5@9F???Ԙ5@AF???Ԙ5@IF???Ԙ5@a?=?cp?i&yS&>????Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1h??|??4@9h??|??4@Ah??|??4@Ih??|??4@a????t?n?i?fI?*????Unknown
lHostIteratorGetNext"IteratorGetNext(1V-??O2@9V-??O2@AV-??O2@IV-??O2@a??n`.k?i?թ?X????Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?S㥛$2@9?S㥛$2@A?S㥛$2@I?S㥛$2@a?A?D?j?i??+?F????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1j?t??/@9j?t??/@Aj?t??/@Ij?t??/@a~/?M-sg?i??y(????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?????)@9?????)@A?????)@I?????)@a????D+c?i?]4m????Unknown
gHostStridedSlice"strided_slice(1'1??)@9'1??)@A'1??)@I'1??)@a??i4T c?i??h??)???Unknown
[HostAddV2"Adam/add(1????x?%@9????x?%@A????x?%@I????x?%@a?I???`?iط~?9???Unknown
ZHostArgMax"ArgMax(1?p=
?c"@9?p=
?c"@A?p=
?c"@I?p=
?c"@aR?~j L[?i@?;??G???Unknown
?HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1H?z?"@9H?z?"@AH?z?"@IH?z?"@a?r??S?Z?iy?98U???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1^?I?!@9^?I?!@A^?I?!@I^?I?!@a??S??9Z?i`?b???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1???K7?@9???K7?@A???K7?@I???K7?@aV"`w?-U?iq????l???Unknown
eHost
LogicalAnd"
LogicalAnd(1V-??o@9V-??o@AV-??o@IV-??o@a?#s??U?i?6cBw???Unknown?
?HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1?rh???@9?rh???@A?rh???@I?rh???@a??/?IS?i??8?????Unknown
VHostSum"Sum_2(1?E???T@9?E???T@A?E???T@I?E???T@a~???R?i????M????Unknown
vHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@a??J!??R?iZ'??????Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1D?l??)@9D?l??)@AD?l??)@ID?l??)@aRg??sP?i?%?D?????Unknown
?!HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ˡE??}@9ˡE??}@AˡE??}@IˡE??}@a??6??jN?iG3?{????Unknown
?"HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1j?t??@9j?t??@Aj?t??@Ij?t??@a(X?? ?L?i?"=??????Unknown
?#HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@a?S??-J?i?^AdE????Unknown
$HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1;?O??n@9;?O??n@A;?O??n@I;?O??n@akɎ'?I?idBj?????Unknown
?%HostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1?v??/@9?v??/@A?v??/@I?v??/@a?]}??$G?i?????????Unknown
[&HostPow"
Adam/Pow_1(1??S㥛@9??S㥛@A??S㥛@I??S㥛@a???u?E?i?? ????Unknown
V'HostCast"Cast(1?????M@9?????M@A?????M@I?????M@a????E?i}CnE????Unknown
X(HostEqual"Equal(1??(\??	@9??(\??	@A??(\??	@I??(\??	@a??y>YC?i??V????Unknown
d)HostDataset"Iterator::Model(1+?X@9+?X@A?ʡE??@I?ʡE??@a?me?ԄB?iU;y9?????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(19??v??@99??v??@A9??v??@I9??v??@a;??P?@?i}z?Z?????Unknown
?+HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a?ћש=?i??p?f????Unknown
X,HostCast"Cast_3(1?????M @9?????M @A?????M @I?????M @a??xwt38?i??m????Unknown
]-HostCast"Adam/Cast_1(1?/?$??9?/?$??A?/?$??I?/?$??a],?7?i\]??M????Unknown
?.HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a289?-?6?i????+????Unknown
X/HostCast"Cast_2(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a?l$J6?iJ??????Unknown
T0HostMul"Mul(1?/?$??9?/?$??A?/?$??I?/?$??abQ?>	5/?i???*?????Unknown
b1HostDivNoNan"div_no_nan_1(1?/?$??9?/?$??A?/?$??I?/?$??abQ?>	5/?i??q{?????Unknown
}2HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1{?G?z??9{?G?z??A{?G?z??I{?G?z??a?51{Qf.?i٠???????Unknown
X3HostCast"Cast_4(1???Mb??9???Mb??A???Mb??I???Mb??a?m=?-?iIr`d?????Unknown
v4HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1???S????9???S????A???S????I???S????a???^)*-?ib?q????Unknown
t5HostReadVariableOp"Adam/Cast/ReadVariableOp(1y?&1???9y?&1???Ay?&1???Iy?&1???a0D?I?+?iZʇ{,????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1?t?V??9?t?V??A?t?V??I?t?V??aLB?Y?7+?i?h=??????Unknown
?7HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1^?I+??9^?I+??A^?I+??I^?I+??a9???{)?i?y??w????Unknown
?8HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(1V-?????9V-?????AV-?????IV-?????a?ٺƳ'?il'??????Unknown
Y9HostPow"Adam/Pow(1V-???9V-???AV-???IV-???aGF.
&?i@???S????Unknown
o:HostReadVariableOp"Adam/ReadVariableOp(1{?G?z??9{?G?z??A{?G?z??I{?G?z??a?7??$#%?i?0å????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1?????K??9?????K??A?????K??I?????K??a?d?/B$?i(??????Unknown
u<HostReadVariableOp"div_no_nan/ReadVariableOp(1V-????9V-????AV-????IV-????a??-??!?iY?
V????Unknown
`=HostDivNoNan"
div_no_nan(1??C?l???9??C?l???A??C?l???I??C?l???an? ?p??i^0???????Unknown
~>HostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??a?YS???i?RN??????Unknown
y?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a??;?i?i?L???????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1#??~j???9#??~j???A#??~j???I#??~j???ajE????i/?IV????Unknown
aAHostIdentity"Identity(1)\???(??9)\???(??A)\???(??I)\???(??a??4K???i??e/????Unknown?
?BHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1+????9+????A+????I+????a??-???iB?W?????Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1???(\???9???(\???A???(\???I???(\???a????X??i'??
k????Unknown
wDHostReadVariableOp"div_no_nan/ReadVariableOp_1(1J+???9J+???AJ+???IJ+???a?ځ???i?????????Unknown*??
?HostDataset"9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2(1L7?A???@9L7?A???@A?&1,??@I?&1,??@a???i????Unknown
jHost_FusedMatMul"model/dense/Relu(1+???4?@9+???4?@A+???4?@I+???4?@a*׊???i@???D4???Unknown
tHostMatMul" gradient_tape/model/dense/MatMul(1sh??|?@9sh??|?@Ash??|?@Ish??|?@a???]??i??d??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1V-???m@9V-???m@AV-???m@IV-???m@aϵ?{&??i?8?????Unknown
?HostDataset"BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle(?1??ʡ`@9??ʡ??A??ʡ`@I??ʡ??a??<!>???i?K??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(17?A`?XR@97?A`?XR@A7?A`?XR@I7?A`?XR@a\UC?0??i???J???Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1????x9W@9????x9W@A+???7M@I+???7M@a?Q+???i?:??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?????KD@9?????KD@A?????KD@I?????KD@a???[N-??i??χ"???Unknown?
x	HostMatMul"$gradient_tape/model/dense_1/MatMul_1(1'1??B@9'1??B@A'1??B@I'1??B@a??P?jv??i?Y?zat???Unknown
?
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1o??ʱB@9o??ʱB@Ao??ʱB@Io??ʱB@agf?nSm??is?`?????Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1B`??";A@9B`??";A@AB`??";A@IB`??";A@a?????ӂ?i????f???Unknown
xHostReluGrad""gradient_tape/model/dense/ReluGrad(1=
ףp?=@9=
ףp?=@A=
ףp?=@I=
ףp?=@a#<?'????iҢ??dR???Unknown
oHost_FusedMatMul"model/dense_1/BiasAdd(1?v??:@9?v??:@A?v??:@I?v??:@ah9?]y|?iEY?]W????Unknown
iHostWriteSummary"WriteSummary(1?? ?r7@9?? ?r7@A?? ?r7@I?? ?r7@a??k?*y?i?/?1?????Unknown?
jHostSoftmax"model/dense_1/Softmax(1F???Ԙ5@9F???Ԙ5@AF???Ԙ5@IF???Ԙ5@a???:?w?i.???????Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1h??|??4@9h??|??4@Ah??|??4@Ih??|??4@a? ?Ä?v?iC:|?f???Unknown
lHostIteratorGetNext"IteratorGetNext(1V-??O2@9V-??O2@AV-??O2@IV-??O2@a??Lt?iEӶ?jB???Unknown
?HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?S㥛$2@9?S㥛$2@A?S㥛$2@I?S㥛$2@aC????s?i???j???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1j?t??/@9j?t??/@Aj?t??/@Ij?t??/@aeG??Cq?iE???????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?????)@9?????)@A?????)@I?????)@a???/?8l?i??-?Ϩ???Unknown
gHostStridedSlice"strided_slice(1'1??)@9'1??)@A'1??)@I'1??)@a#?|e??k?i??b?????Unknown
[HostAddV2"Adam/add(1????x?%@9????x?%@A????x?%@I????x?%@a??O3i?g?inU??t????Unknown
ZHostArgMax"ArgMax(1?p=
?c"@9?p=
?c"@A?p=
?c"@I?p=
?c"@a?:?G&d?i???????Unknown
?HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1H?z?"@9H?z?"@AH?z?"@IH?z?"@a?CR?r?c?i?i?d@???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1^?I?!@9^?I?!@A^?I?!@I^?I?!@a?n?0Nc?i\"픎???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1???K7?@9???K7?@A???K7?@I???K7?@a8?f=>._?i???%'???Unknown
eHost
LogicalAnd"
LogicalAnd(1V-??o@9V-??o@AV-??o@IV-??o@aY?9EE_?i?r?֮6???Unknown?
?HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1?rh???@9?rh???@A?rh???@I?rh???@a????e\?i???D???Unknown
VHostSum"Sum_2(1?E???T@9?E???T@A?E???T@I?E???T@a??h? ?[?ii?p??R???Unknown
vHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1333333@9333333@A333333@I333333@aْ?L?[?i2??!}`???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1D?l??)@9D?l??)@AD?l??)@ID?l??)@aD\D?7X?iT*9?l???Unknown
? HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ˡE??}@9ˡE??}@AˡE??}@IˡE??}@a????!dV?iF?&?w???Unknown
?!HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1j?t??@9j?t??@Aj?t??@Ij?t??@a???SU?i?挬t????Unknown
?"HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@a???<ES?i??J????Unknown
#HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1;?O??n@9;?O??n@A;?O??n@I;?O??n@a/0l1,S?i.??`?????Unknown
?$HostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1?v??/@9?v??/@A?v??/@I?v??/@a5?Zz	Q?i?%"????Unknown
[%HostPow"
Adam/Pow_1(1??S㥛@9??S㥛@A??S㥛@I??S㥛@a*???-P?i?i?8????Unknown
V&HostCast"Cast(1?????M@9?????M@A?????M@I?????M@agWn?X?N?iD??????Unknown
X'HostEqual"Equal(1??(\??	@9??(\??	@A??(\??	@I??(\??	@a?/h??%L?i&??i?????Unknown
d(HostDataset"Iterator::Model(1+?X@9+?X@A?ʡE??@I?ʡE??@a!V?CK?i???Vλ???Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_4(19??v??@99??v??@A9??v??@I9??v??@adQ4ȠG?i?????????Unknown
?*HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a'?"?E?i6H,????Unknown
X+HostCast"Cast_3(1?????M @9?????M @A?????M @I?????M @a0/u??A?i? 4?????Unknown
],HostCast"Adam/Cast_1(1?/?$??9?/?$??A?/?$??I?/?$??a?_l??@?i.????????Unknown
?-HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a[?0?2?@?iW{)G????Unknown
X.HostCast"Cast_2(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a?|?+[h@?i?r?]/????Unknown
T/HostMul"Mul(1?/?$??9?/?$??A?/?$??I?/?$??a/`$7??6?iBW;|????Unknown
b0HostDivNoNan"div_no_nan_1(1?/?$??9?/?$??A?/?$??I?/?$??a/`$7??6?i?;???????Unknown
}1HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1{?G?z??9{?G?z??A{?G?z??I{?G?z??a۱???`6?i??S??????Unknown
X2HostCast"Cast_4(1???Mb??9???Mb??A???Mb??I???Mb??a?w??h?5?is?p@w????Unknown
v3HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1???S????9???S????A???S????I???S????a?=?Fx5?i{y?A&????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1y?&1???9y?&1???Ay?&1???Iy?&1???a?f?6g4?i?S?"?????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1?t?V??9?t?V??A?t?V??I?t?V??a??W[	4?i??KD4????Unknown
?6HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1^?I+??9^?I+??A^?I+??I^?I+??aK??;S?2?i]9???????Unknown
?7HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(1V-?????9V-?????AV-?????IV-?????a?????r1?ip???????Unknown
Y8HostPow"Adam/Pow(1V-???9V-???AV-???IV-???a&? ?\90?i24>?????Unknown
o9HostReadVariableOp"Adam/ReadVariableOp(1{?G?z??9{?G?z??A{?G?z??I{?G?z??aU???/?i[???????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_3(1?????K??9?????K??A?????K??I?????K??as?8b?-?iuA?.?????Unknown
u;HostReadVariableOp"div_no_nan/ReadVariableOp(1V-????9V-????AV-????IV-????a}S???)?i???.????Unknown
`<HostDivNoNan"
div_no_nan(1??C?l???9??C?l???A??C?l???I??C?l???at&?[??%?iܫ	?????Unknown
~=HostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1h??|?5??9h??|?5??Ah??|?5??Ih??|?5??aSa?P=?#?i???\?????Unknown
y>HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1L7?A`???9L7?A`???AL7?A`???IL7?A`???a"??e=v"?ib???????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_1(1#??~j???9#??~j???A#??~j???I#??~j???a????{I"?i??jX????Unknown
a@HostIdentity"Identity(1)\???(??9)\???(??A)\???(??I)\???(??aWi?\?!?i25?.????Unknown?
?AHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1+????9+????A+????I+????a??:?{? ?i???<????Unknown
wBHostReadVariableOp"div_no_nan_1/ReadVariableOp(1???(\???9???(\???A???(\???I???(\???a???{??iX???$????Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1J+???9J+???AJ+???IJ+???au?i?i      ???Unknown2CPU