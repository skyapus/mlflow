??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.0.02v2.0.0-rc2-26-g64c3d388??

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
?
global_stepVarHandleOp*
shared_nameglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: *
shape: 
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
dtype0	*
_output_shapes
: 
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
\
dnn/CastCastPlaceholder_2*#
_output_shapes
:?????????*

DstT0*

SrcT0
^

dnn/Cast_1CastPlaceholder_3*#
_output_shapes
:?????????*

DstT0*

SrcT0
\

dnn/Cast_2CastPlaceholder*

SrcT0*#
_output_shapes
:?????????*

DstT0
^

dnn/Cast_3CastPlaceholder_1*#
_output_shapes
:?????????*

DstT0*

SrcT0
?
Gdnn/input_from_feature_columns/input_layer/PetalLength_1/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Cdnn/input_from_feature_columns/input_layer/PetalLength_1/ExpandDims
ExpandDimsdnn/CastGdnn/input_from_feature_columns/input_layer/PetalLength_1/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
>dnn/input_from_feature_columns/input_layer/PetalLength_1/ShapeShapeCdnn/input_from_feature_columns/input_layer/PetalLength_1/ExpandDims*
T0*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Fdnn/input_from_feature_columns/input_layer/PetalLength_1/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/PetalLength_1/ShapeLdnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stackNdnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/PetalLength_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Hdnn/input_from_feature_columns/input_layer/PetalLength_1/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
?
Fdnn/input_from_feature_columns/input_layer/PetalLength_1/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/PetalLength_1/strided_sliceHdnn/input_from_feature_columns/input_layer/PetalLength_1/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
@dnn/input_from_feature_columns/input_layer/PetalLength_1/ReshapeReshapeCdnn/input_from_feature_columns/input_layer/PetalLength_1/ExpandDimsFdnn/input_from_feature_columns/input_layer/PetalLength_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Fdnn/input_from_feature_columns/input_layer/PetalWidth_1/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Bdnn/input_from_feature_columns/input_layer/PetalWidth_1/ExpandDims
ExpandDims
dnn/Cast_1Fdnn/input_from_feature_columns/input_layer/PetalWidth_1/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
=dnn/input_from_feature_columns/input_layer/PetalWidth_1/ShapeShapeBdnn/input_from_feature_columns/input_layer/PetalWidth_1/ExpandDims*
_output_shapes
:*
T0
?
Kdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
Ednn/input_from_feature_columns/input_layer/PetalWidth_1/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/PetalWidth_1/ShapeKdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stackMdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
?
Gdnn/input_from_feature_columns/input_layer/PetalWidth_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Ednn/input_from_feature_columns/input_layer/PetalWidth_1/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/PetalWidth_1/strided_sliceGdnn/input_from_feature_columns/input_layer/PetalWidth_1/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/PetalWidth_1/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/PetalWidth_1/ExpandDimsEdnn/input_from_feature_columns/input_layer/PetalWidth_1/Reshape/shape*'
_output_shapes
:?????????*
T0
?
Gdnn/input_from_feature_columns/input_layer/SepalLength_1/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Cdnn/input_from_feature_columns/input_layer/SepalLength_1/ExpandDims
ExpandDims
dnn/Cast_2Gdnn/input_from_feature_columns/input_layer/SepalLength_1/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
>dnn/input_from_feature_columns/input_layer/SepalLength_1/ShapeShapeCdnn/input_from_feature_columns/input_layer/SepalLength_1/ExpandDims*
T0*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
?
Ndnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/SepalLength_1/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/SepalLength_1/ShapeLdnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stackNdnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/SepalLength_1/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
?
Hdnn/input_from_feature_columns/input_layer/SepalLength_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Fdnn/input_from_feature_columns/input_layer/SepalLength_1/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/SepalLength_1/strided_sliceHdnn/input_from_feature_columns/input_layer/SepalLength_1/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
@dnn/input_from_feature_columns/input_layer/SepalLength_1/ReshapeReshapeCdnn/input_from_feature_columns/input_layer/SepalLength_1/ExpandDimsFdnn/input_from_feature_columns/input_layer/SepalLength_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Fdnn/input_from_feature_columns/input_layer/SepalWidth_1/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Bdnn/input_from_feature_columns/input_layer/SepalWidth_1/ExpandDims
ExpandDims
dnn/Cast_3Fdnn/input_from_feature_columns/input_layer/SepalWidth_1/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
=dnn/input_from_feature_columns/input_layer/SepalWidth_1/ShapeShapeBdnn/input_from_feature_columns/input_layer/SepalWidth_1/ExpandDims*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Ednn/input_from_feature_columns/input_layer/SepalWidth_1/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/SepalWidth_1/ShapeKdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stackMdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
Gdnn/input_from_feature_columns/input_layer/SepalWidth_1/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Ednn/input_from_feature_columns/input_layer/SepalWidth_1/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/SepalWidth_1/strided_sliceGdnn/input_from_feature_columns/input_layer/SepalWidth_1/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
?dnn/input_from_feature_columns/input_layer/SepalWidth_1/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/SepalWidth_1/ExpandDimsEdnn/input_from_feature_columns/input_layer/SepalWidth_1/Reshape/shape*
T0*'
_output_shapes
:?????????
?
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
1dnn/input_from_feature_columns/input_layer/concatConcatV2@dnn/input_from_feature_columns/input_layer/PetalLength_1/Reshape?dnn/input_from_feature_columns/input_layer/PetalWidth_1/Reshape@dnn/input_from_feature_columns/input_layer/SepalLength_1/Reshape?dnn/input_from_feature_columns/input_layer/SepalWidth_1/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:?????????
?
9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *+
_class!
loc:@dnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
:
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *b?'?*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
dtype0
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/maxConst*
valueB
 *b?'?*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
: 
?
Adnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_0/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:
*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_0/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
_output_shapes

:

?
3dnn/hiddenlayer_0/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_0/kernel/Initializer/random_uniform/min*
_output_shapes

:
*
T0*+
_class!
loc:@dnn/hiddenlayer_0/kernel
?
dnn/hiddenlayer_0/kernelVarHandleOp*
shape
:
*)
shared_namednn/hiddenlayer_0/kernel*+
_class!
loc:@dnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
: 
?
9dnn/hiddenlayer_0/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_0/kernel/AssignAssignVariableOpdnn/hiddenlayer_0/kernel3dnn/hiddenlayer_0/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_0/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes

:

?
(dnn/hiddenlayer_0/bias/Initializer/zerosConst*
valueB
*    *)
_class
loc:@dnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_0/biasVarHandleOp*
shape:
*'
shared_namednn/hiddenlayer_0/bias*)
_class
loc:@dnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
: 
}
7dnn/hiddenlayer_0/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias*
_output_shapes
: 
?
dnn/hiddenlayer_0/bias/AssignAssignVariableOpdnn/hiddenlayer_0/bias(dnn/hiddenlayer_0/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_0/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:

?
'dnn/hiddenlayer_0/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes

:

?
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concat'dnn/hiddenlayer_0/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0
{
(dnn/hiddenlayer_0/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMul(dnn/hiddenlayer_0/BiasAdd/ReadVariableOp*'
_output_shapes
:?????????
*
T0
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:?????????

g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
_output_shapes
: *
T0*
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
_output_shapes
: *
valueB	 R????*
dtype0	
?
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/condStatelessIfdnn/zero_fraction/LessEqualdnn/hiddenlayer_0/Relu*
Tout

2	*
Tcond0
*3
then_branch$R"
 dnn_zero_fraction_cond_true_2027* 
_output_shapes
: : : : : : *
Tin
2*4
else_branch%R#
!dnn_zero_fraction_cond_false_2028*
output_shapes
: : : : : : *
_lower_using_switch_merge(
d
dnn/zero_fraction/cond/IdentityIdentitydnn/zero_fraction/cond*
T0	*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_1Identitydnn/zero_fraction/cond:1*
_output_shapes
: *
T0
h
!dnn/zero_fraction/cond/Identity_2Identitydnn/zero_fraction/cond:2*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_3Identitydnn/zero_fraction/cond:3*
T0*
_output_shapes
: 
h
!dnn/zero_fraction/cond/Identity_4Identitydnn/zero_fraction/cond:4*
_output_shapes
: *
T0
h
!dnn/zero_fraction/cond/Identity_5Identitydnn/zero_fraction/cond:5*
T0*
_output_shapes
: 
?
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Identity*
_output_shapes
: *
T0	
?
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
.dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
)dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
_output_shapes
: *
T0
}
 dnn/hiddenlayer_0/activation/tagConst*-
value$B" Bdnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
?
dnn/hiddenlayer_0/activationHistogramSummary dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
?
9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shapeConst*
valueB"
   
   *+
_class!
loc:@dnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
:
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/minConst*
valueB
 *?7?*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
: 
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *?7?*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
: 
?
Adnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9dnn/hiddenlayer_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes

:


?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/subSub7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/max7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes
: 
?
7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mulMulAdnn/hiddenlayer_1/kernel/Initializer/random_uniform/RandomUniform7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
_output_shapes

:


?
3dnn/hiddenlayer_1/kernel/Initializer/random_uniformAdd7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/mul7dnn/hiddenlayer_1/kernel/Initializer/random_uniform/min*
_output_shapes

:

*
T0*+
_class!
loc:@dnn/hiddenlayer_1/kernel
?
dnn/hiddenlayer_1/kernelVarHandleOp*
shape
:

*)
shared_namednn/hiddenlayer_1/kernel*+
_class!
loc:@dnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
: 
?
9dnn/hiddenlayer_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel*
_output_shapes
: 
?
dnn/hiddenlayer_1/kernel/AssignAssignVariableOpdnn/hiddenlayer_1/kernel3dnn/hiddenlayer_1/kernel/Initializer/random_uniform*
dtype0
?
,dnn/hiddenlayer_1/kernel/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes

:


?
(dnn/hiddenlayer_1/bias/Initializer/zerosConst*
_output_shapes
:
*
valueB
*    *)
_class
loc:@dnn/hiddenlayer_1/bias*
dtype0
?
dnn/hiddenlayer_1/biasVarHandleOp*
_output_shapes
: *
shape:
*'
shared_namednn/hiddenlayer_1/bias*)
_class
loc:@dnn/hiddenlayer_1/bias*
dtype0
}
7dnn/hiddenlayer_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias*
_output_shapes
: 
?
dnn/hiddenlayer_1/bias/AssignAssignVariableOpdnn/hiddenlayer_1/bias(dnn/hiddenlayer_1/bias/Initializer/zeros*
dtype0
}
*dnn/hiddenlayer_1/bias/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
dtype0*
_output_shapes
:

?
'dnn/hiddenlayer_1/MatMul/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes

:


?
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Relu'dnn/hiddenlayer_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????

{
(dnn/hiddenlayer_1/BiasAdd/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias*
dtype0*
_output_shapes
:

?
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMul(dnn/hiddenlayer_1/BiasAdd/ReadVariableOp*'
_output_shapes
:?????????
*
T0
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:?????????

i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/condStatelessIfdnn/zero_fraction_1/LessEqualdnn/hiddenlayer_1/Relu*
Tin
2* 
_output_shapes
: : : : : : *6
else_branch'R%
#dnn_zero_fraction_1_cond_false_2098*
output_shapes
: : : : : : *
_lower_using_switch_merge(*
Tout

2	*
Tcond0
*5
then_branch&R$
"dnn_zero_fraction_1_cond_true_2097
h
!dnn/zero_fraction_1/cond/IdentityIdentitydnn/zero_fraction_1/cond*
_output_shapes
: *
T0	
l
#dnn/zero_fraction_1/cond/Identity_1Identitydnn/zero_fraction_1/cond:1*
_output_shapes
: *
T0
l
#dnn/zero_fraction_1/cond/Identity_2Identitydnn/zero_fraction_1/cond:2*
_output_shapes
: *
T0
l
#dnn/zero_fraction_1/cond/Identity_3Identitydnn/zero_fraction_1/cond:3*
_output_shapes
: *
T0
l
#dnn/zero_fraction_1/cond/Identity_4Identitydnn/zero_fraction_1/cond:4*
_output_shapes
: *
T0
l
#dnn/zero_fraction_1/cond/Identity_5Identitydnn/zero_fraction_1/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Size!dnn/zero_fraction_1/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
.dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*:
value1B/ B)dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
)dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1/activation/tagConst*-
value$B" Bdnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
?
dnn/hiddenlayer_1/activationHistogramSummary dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
?
2dnn/logits/kernel/Initializer/random_uniform/shapeConst*
valueB"
      *$
_class
loc:@dnn/logits/kernel*
dtype0*
_output_shapes
:
?
0dnn/logits/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *??-?*$
_class
loc:@dnn/logits/kernel
?
0dnn/logits/kernel/Initializer/random_uniform/maxConst*
valueB
 *??-?*$
_class
loc:@dnn/logits/kernel*
dtype0*
_output_shapes
: 
?
:dnn/logits/kernel/Initializer/random_uniform/RandomUniformRandomUniform2dnn/logits/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@dnn/logits/kernel*
dtype0*
_output_shapes

:

?
0dnn/logits/kernel/Initializer/random_uniform/subSub0dnn/logits/kernel/Initializer/random_uniform/max0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes
: 
?
0dnn/logits/kernel/Initializer/random_uniform/mulMul:dnn/logits/kernel/Initializer/random_uniform/RandomUniform0dnn/logits/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:

?
,dnn/logits/kernel/Initializer/random_uniformAdd0dnn/logits/kernel/Initializer/random_uniform/mul0dnn/logits/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@dnn/logits/kernel*
_output_shapes

:

?
dnn/logits/kernelVarHandleOp*
shape
:
*"
shared_namednn/logits/kernel*$
_class
loc:@dnn/logits/kernel*
dtype0*
_output_shapes
: 
s
2dnn/logits/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel*
_output_shapes
: 
z
dnn/logits/kernel/AssignAssignVariableOpdnn/logits/kernel,dnn/logits/kernel/Initializer/random_uniform*
dtype0
w
%dnn/logits/kernel/Read/ReadVariableOpReadVariableOpdnn/logits/kernel*
dtype0*
_output_shapes

:

?
!dnn/logits/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@dnn/logits/bias*
dtype0*
_output_shapes
:
?
dnn/logits/biasVarHandleOp*"
_class
loc:@dnn/logits/bias*
dtype0*
_output_shapes
: *
shape:* 
shared_namednn/logits/bias
o
0dnn/logits/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias*
_output_shapes
: 
k
dnn/logits/bias/AssignAssignVariableOpdnn/logits/bias!dnn/logits/bias/Initializer/zeros*
dtype0
o
#dnn/logits/bias/Read/ReadVariableOpReadVariableOpdnn/logits/bias*
dtype0*
_output_shapes
:
r
 dnn/logits/MatMul/ReadVariableOpReadVariableOpdnn/logits/kernel*
dtype0*
_output_shapes

:

?
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Relu dnn/logits/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
m
!dnn/logits/BiasAdd/ReadVariableOpReadVariableOpdnn/logits/bias*
_output_shapes
:*
dtype0
?
dnn/logits/BiasAddBiasAdddnn/logits/MatMul!dnn/logits/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
out_type0	*
_output_shapes
: *
T0
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/condStatelessIfdnn/zero_fraction_2/LessEqualdnn/logits/BiasAdd*
Tcond0
*5
then_branch&R$
"dnn_zero_fraction_2_cond_true_2166* 
_output_shapes
: : : : : : *
Tin
2*6
else_branch'R%
#dnn_zero_fraction_2_cond_false_2167*
output_shapes
: : : : : : *
_lower_using_switch_merge(*
Tout

2	
h
!dnn/zero_fraction_2/cond/IdentityIdentitydnn/zero_fraction_2/cond*
T0	*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_1Identitydnn/zero_fraction_2/cond:1*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_2Identitydnn/zero_fraction_2/cond:2*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_3Identitydnn/zero_fraction_2/cond:3*
_output_shapes
: *
T0
l
#dnn/zero_fraction_2/cond/Identity_4Identitydnn/zero_fraction_2/cond:4*
T0*
_output_shapes
: 
l
#dnn/zero_fraction_2/cond/Identity_5Identitydnn/zero_fraction_2/cond:5*
T0*
_output_shapes
: 
?
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Size!dnn/zero_fraction_2/cond/Identity*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
'dnn/logits/fraction_of_zero_values/tagsConst*3
value*B( B"dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
"dnn/logits/fraction_of_zero_valuesScalarSummary'dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
o
dnn/logits/activation/tagConst*&
valueB Bdnn/logits/activation*
dtype0*
_output_shapes
: 
p
dnn/logits/activationHistogramSummarydnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
S
head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
g
%head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
W
Ohead/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
H
@head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
o
head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
head/predictions/class_idsArgMaxdnn/logits/BiasAdd$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*'
_output_shapes
:?????????*
T0	
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
X
head/predictions/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&head/predictions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&head/predictions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
^
head/predictions/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
^
head/predictions/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
_output_shapes

:*
T0
c
!head/predictions/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
T0*
N*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*'
_output_shapes
:?????????*
T0
Z
head/predictions/Shape_1Shapednn/logits/BiasAdd*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
r
(head/predictions/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
r
(head/predictions/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
`
head/predictions/range_1/startConst*
_output_shapes
: *
value	B : *
dtype0
`
head/predictions/range_1/limitConst*
_output_shapes
: *
value	B :*
dtype0
`
head/predictions/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
_output_shapes

:*
T0
e
#head/predictions/Tile_1/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
T0*
N*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*'
_output_shapes
:?????????*
T0
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
d
head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
R
head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
R
head/range/limitConst*
dtype0*
_output_shapes
: *
value	B :
R
head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
T0*
N*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
?
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_758f802a1e7a42acaa730c133d7de5a1/part*
dtype0
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices*dnn/hiddenlayer_0/bias/Read/ReadVariableOp,dnn/hiddenlayer_0/kernel/Read/ReadVariableOp*dnn/hiddenlayer_1/bias/Read/ReadVariableOp,dnn/hiddenlayer_1/kernel/Read/ReadVariableOp#dnn/logits/bias/Read/ReadVariableOp%dnn/logits/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOp"/device:CPU:0*
dtypes
	2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
_output_shapes
:*
T0*
N
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	
N
save/Identity_1Identitysave/RestoreV2*
_output_shapes
:*
T0
_
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
c
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
a
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
c
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
Z
save/AssignVariableOp_4AssignVariableOpdnn/logits/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
_output_shapes
:*
T0
\
save/AssignVariableOp_5AssignVariableOpdnn/logits/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0	*
_output_shapes
:
V
save/AssignVariableOp_6AssignVariableOpglobal_stepsave/Identity_7*
dtype0	
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6
-
save/restore_allNoOp^save/restore_shard?T
?
?
#dnn_zero_fraction_2_cond_false_2167-
)count_nonzero_notequal_dnn_logits_biasadd
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneX
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*'
_output_shapes
:?????????*
T0w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*'
_output_shapes
:?????????*

DstT0	*

SrcT0
d
count_nonzero/ConstConst*
dtype0*
_output_shapes
:*
valueB"       y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
_output_shapes
: *
T0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: 4
OptionalNoneOptionalNone*
_output_shapes
: "1
optionalfromvalueOptionalFromValue:optional:0"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????:  
?
?
"dnn_zero_fraction_1_cond_true_20971
-count_nonzero_notequal_dnn_hiddenlayer_1_relu
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_1_relucount_nonzero/zeros:output:0*'
_output_shapes
:?????????
*
T0w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*'
_output_shapes
:?????????
*

DstT0d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
_output_shapes
: *
T0b
CastCast$count_nonzero/nonzero_count:output:0*

SrcT0*
_output_shapes
: *

DstT0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
_output_shapes
: *
Toutput_types
2~
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: "
castCast:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"1
optionalfromvalueOptionalFromValue:optional:0*&
_input_shapes
:?????????
:  
?
?
#dnn_zero_fraction_1_cond_false_20981
-count_nonzero_notequal_dnn_hiddenlayer_1_relu
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneX
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_1_relucount_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*'
_output_shapes
:?????????
*

DstT0	*

SrcT0
d
count_nonzero/ConstConst*
_output_shapes
:*
valueB"       *
dtype0y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
_output_shapes
: *
T0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
_output_shapes
: *
Toutput_types
2t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
_output_shapes
: *
Toutput_types
24
OptionalNoneOptionalNone*
_output_shapes
: "1
optionalfromvalueOptionalFromValue:optional:0"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0*&
_input_shapes
:?????????
:  
?
?
!dnn_zero_fraction_cond_false_20281
-count_nonzero_notequal_dnn_hiddenlayer_0_relu
count_nonzero_nonzero_count	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalnoneX
count_nonzero/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    ?
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_0_relucount_nonzero/zeros:output:0*'
_output_shapes
:?????????
*
T0w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*'
_output_shapes
:?????????
*

DstT0	d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
_output_shapes
: *
T0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
_output_shapes
: *
Toutput_types
2
p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2	*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
_output_shapes
: *
Toutput_types
24
OptionalNoneOptionalNone*
_output_shapes
: "5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"'
optionalnoneOptionalNone:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"C
count_nonzero_nonzero_count$count_nonzero/nonzero_count:output:0*&
_input_shapes
:?????????
:  
?
?
"dnn_zero_fraction_2_cond_true_2166-
)count_nonzero_notequal_dnn_logits_biasadd
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
count_nonzero/NotEqualNotEqual)count_nonzero_notequal_dnn_logits_biasaddcount_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*'
_output_shapes
:?????????*

DstT0*

SrcT0
d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
_output_shapes
: *
T0b
CastCast$count_nonzero/nonzero_count:output:0*

SrcT0*
_output_shapes
: *

DstT0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: ~
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: "1
optionalfromvalueOptionalFromValue:optional:0"
castCast:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0*&
_input_shapes
:?????????:  
?
?
 dnn_zero_fraction_cond_true_20271
-count_nonzero_notequal_dnn_hiddenlayer_0_relu
cast	
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4X
count_nonzero/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
count_nonzero/NotEqualNotEqual-count_nonzero_notequal_dnn_hiddenlayer_0_relucount_nonzero/zeros:output:0*
T0*'
_output_shapes
:?????????
w
count_nonzero/CastCastcount_nonzero/NotEqual:z:0*

SrcT0
*'
_output_shapes
:?????????
*

DstT0d
count_nonzero/ConstConst*
valueB"       *
dtype0*
_output_shapes
:y
count_nonzero/nonzero_countSumcount_nonzero/Cast:y:0count_nonzero/Const:output:0*
T0*
_output_shapes
: b
CastCast$count_nonzero/nonzero_count:output:0*

SrcT0*
_output_shapes
: *

DstT0	t
OptionalFromValueOptionalFromValuecount_nonzero/zeros:output:0*
Toutput_types
2*
_output_shapes
: t
OptionalFromValue_1OptionalFromValuecount_nonzero/NotEqual:z:0*
Toutput_types
2
*
_output_shapes
: p
OptionalFromValue_2OptionalFromValuecount_nonzero/Cast:y:0*
Toutput_types
2*
_output_shapes
: v
OptionalFromValue_3OptionalFromValuecount_nonzero/Const:output:0*
Toutput_types
2*
_output_shapes
: ~
OptionalFromValue_4OptionalFromValue$count_nonzero/nonzero_count:output:0*
Toutput_types
2*
_output_shapes
: "5
optionalfromvalue_1OptionalFromValue_1:optional:0"
castCast:y:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"1
optionalfromvalueOptionalFromValue:optional:0*&
_input_shapes
:?????????
:  "w<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"?
	variables??
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
?
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
?
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
?
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08"%
saved_model_main_op


group_deps"?
trainable_variables??
?
dnn/hiddenlayer_0/kernel:0dnn/hiddenlayer_0/kernel/Assign.dnn/hiddenlayer_0/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_0/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias:0dnn/hiddenlayer_0/bias/Assign,dnn/hiddenlayer_0/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_0/bias/Initializer/zeros:08
?
dnn/hiddenlayer_1/kernel:0dnn/hiddenlayer_1/kernel/Assign.dnn/hiddenlayer_1/kernel/Read/ReadVariableOp:0(25dnn/hiddenlayer_1/kernel/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias:0dnn/hiddenlayer_1/bias/Assign,dnn/hiddenlayer_1/bias/Read/ReadVariableOp:0(2*dnn/hiddenlayer_1/bias/Initializer/zeros:08
?
dnn/logits/kernel:0dnn/logits/kernel/Assign'dnn/logits/kernel/Read/ReadVariableOp:0(2.dnn/logits/kernel/Initializer/random_uniform:08
{
dnn/logits/bias:0dnn/logits/bias/Assign%dnn/logits/bias/Read/ReadVariableOp:0(2#dnn/logits/bias/Initializer/zeros:08"?
	summaries?
?
+dnn/hiddenlayer_0/fraction_of_zero_values:0
dnn/hiddenlayer_0/activation:0
+dnn/hiddenlayer_1/fraction_of_zero_values:0
dnn/hiddenlayer_1/activation:0
$dnn/logits/fraction_of_zero_values:0
dnn/logits/activation:0*?
predict?
/
SepalLength 
Placeholder:0?????????
1
PetalLength"
Placeholder_2:0?????????
0

PetalWidth"
Placeholder_3:0?????????
0

SepalWidth"
Placeholder_1:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????@
classes5
head/predictions/str_classes:0??????????
all_class_ids.
head/predictions/Tile:0??????????
all_classes0
head/predictions/Tile_1:0?????????H
probabilities7
 head/predictions/probabilities:0?????????5
logits+
dnn/logits/BiasAdd:0?????????tensorflow/serving/predict