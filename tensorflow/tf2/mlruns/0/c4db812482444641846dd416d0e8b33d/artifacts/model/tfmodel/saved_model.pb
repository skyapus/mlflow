ž
 Ü
:
Add
x"T
y"T
z"T"
Ttype:
2	

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
¸
AsString

input"T

output"
Ttype:
2		
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

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
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
ö
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

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
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
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58˛Ľ

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
dtype0	*
_output_shapes
: *
shape: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
f
SepalLengthPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e

SepalWidthPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
f
PetalLengthPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
e

PetalWidthPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ednn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ő
Adnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims
ExpandDimsPetalLengthEdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
;dnn/input_from_feature_columns/input_layer/PetalLength/CastCastAdnn/input_from_feature_columns/input_layer/PetalLength/ExpandDims*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
§
<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeShape;dnn/input_from_feature_columns/input_layer/PetalLength/Cast*
T0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0

Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ŕ
Ddnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/PetalLength/ShapeJdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/PetalLength/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

Fdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Ddnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/PetalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape/1*
N*
_output_shapes
:*
T0
ţ
>dnn/input_from_feature_columns/input_layer/PetalLength/ReshapeReshape;dnn/input_from_feature_columns/input_layer/PetalLength/CastDdnn/input_from_feature_columns/input_layer/PetalLength/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ddnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ň
@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims
ExpandDims
PetalWidthDdnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
:dnn/input_from_feature_columns/input_layer/PetalWidth/CastCast@dnn/input_from_feature_columns/input_layer/PetalWidth/ExpandDims*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Ľ
;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeShape:dnn/input_from_feature_columns/input_layer/PetalWidth/Cast*
_output_shapes
:*
T0

Idnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
Cdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/PetalWidth/ShapeIdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/PetalWidth/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ednn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Cdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/PetalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape/1*
N*
_output_shapes
:*
T0
ű
=dnn/input_from_feature_columns/input_layer/PetalWidth/ReshapeReshape:dnn/input_from_feature_columns/input_layer/PetalWidth/CastCdnn/input_from_feature_columns/input_layer/PetalWidth/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ednn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ő
Adnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims
ExpandDimsSepalLengthEdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
;dnn/input_from_feature_columns/input_layer/SepalLength/CastCastAdnn/input_from_feature_columns/input_layer/SepalLength/ExpandDims*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0
§
<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeShape;dnn/input_from_feature_columns/input_layer/SepalLength/Cast*
T0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ŕ
Ddnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/SepalLength/ShapeJdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stackLdnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/SepalLength/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Fdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Ddnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/SepalLength/strided_sliceFdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape/1*
T0*
N*
_output_shapes
:
ţ
>dnn/input_from_feature_columns/input_layer/SepalLength/ReshapeReshape;dnn/input_from_feature_columns/input_layer/SepalLength/CastDdnn/input_from_feature_columns/input_layer/SepalLength/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ddnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dimConst*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
Ň
@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims
ExpandDims
SepalWidthDdnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
:dnn/input_from_feature_columns/input_layer/SepalWidth/CastCast@dnn/input_from_feature_columns/input_layer/SepalWidth/ExpandDims*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Ľ
;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeShape:dnn/input_from_feature_columns/input_layer/SepalWidth/Cast*
_output_shapes
:*
T0

Idnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
Cdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/SepalWidth/ShapeIdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stackKdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/SepalWidth/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0

Cdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/SepalWidth/strided_sliceEdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape/1*
T0*
N*
_output_shapes
:
ű
=dnn/input_from_feature_columns/input_layer/SepalWidth/ReshapeReshape:dnn/input_from_feature_columns/input_layer/SepalWidth/CastCdnn/input_from_feature_columns/input_layer/SepalWidth/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6dnn/input_from_feature_columns/input_layer/concat/axisConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
Ž
1dnn/input_from_feature_columns/input_layer/concatConcatV2>dnn/input_from_feature_columns/input_layer/PetalLength/Reshape=dnn/input_from_feature_columns/input_layer/PetalWidth/Reshape>dnn/input_from_feature_columns/input_layer/SepalLength/Reshape=dnn/input_from_feature_columns/input_layer/SepalWidth/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"   
   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *b'ż*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *b'?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ź
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:
*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
Î
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:
*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ř
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
Ç
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
Ž
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

Ä
dnn/hiddenlayer_0/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: *
shape:
*.
shared_namednn/hiddenlayer_0/bias/part_0

>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
Ç
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
˝
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:


'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes

:

Ą
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
*
dtype0
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:


dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
out_type0	*
_output_shapes
: *
T0
c
dnn/zero_fraction/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R˙˙˙˙

dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
_output_shapes
: *
T0

h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: 

*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
Ď
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ć
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙

Ą
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

DstT0

*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
ą
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0

dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	

,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ó
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

č
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙

Ľ
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

DstT0	

,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
ˇ
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
¤
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 

)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
°
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ż
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
_output_shapes
: *
T0

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
Ĺ
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *7ż*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
ˇ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *7?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:



>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ź
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:



:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:

*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Î
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:

*0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ř
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
Ç
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


Ž
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB
*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

Ä
dnn/hiddenlayer_1/bias/part_0VarHandleOp*
_output_shapes
: *
shape:
*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0

>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
Ç
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
˝
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:


'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes

:



dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:


dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
_output_shapes
: *
T0*
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R˙˙˙˙*
dtype0	*
_output_shapes
: 

dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
_output_shapes
: *
T0


,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
Ő
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ę
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙

Ľ
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

DstT0*

SrcT0

Ą
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
ˇ
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 

dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	

.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ů
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
ě
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙
:˙˙˙˙˙˙˙˙˙

Š
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*

DstT0	*

SrcT0

Ł
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
˝
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ş
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
_output_shapes
: *
T0	

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
ś
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
ą
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
ˇ
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ňę-ż*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *ňę-?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
đ
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
*
T0
ţ
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@dnn/logits/kernel/part_0

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:
*
T0*+
_class!
loc:@dnn/logits/kernel/part_0

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:

š
dnn/logits/kernel/part_0VarHandleOp*
_output_shapes
: *
shape
:
*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0

9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
ź
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
˛
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
Ż
dnn/logits/bias/part_0VarHandleOp*
_output_shapes
: *
shape:*'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
Ť
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
¨
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
_output_shapes

:
*
T0
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
_output_shapes
:*
T0
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_2/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R˙˙˙˙

dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
_output_shapes
: *
T0


,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
Ő
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
Ą
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
ˇ
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
_output_shapes
: *
T0

dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0

.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
_output_shapes
: *
valueB
 *    *
dtype0
Ů
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ä
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	
Ł
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
˝
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ş
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 

+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

SrcT0	*
_output_shapes
: *

DstT0
ś
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *7
value.B, B&dnn/dnn/logits/fraction_of_zero_values
Ł
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
W
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
\
dnn/head/predictions/ShapeShapednn/logits/BiasAdd*
_output_shapes
:*
T0
r
(dnn/head/predictions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*dnn/head/predictions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
t
*dnn/head/predictions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

"dnn/head/predictions/strided_sliceStridedSlicednn/head/predictions/Shape(dnn/head/predictions/strided_slice/stack*dnn/head/predictions/strided_slice/stack_1*dnn/head/predictions/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
b
 dnn/head/predictions/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
b
 dnn/head/predictions/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
b
 dnn/head/predictions/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ľ
dnn/head/predictions/rangeRange dnn/head/predictions/range/start dnn/head/predictions/range/limit dnn/head/predictions/range/delta*
_output_shapes
:
e
#dnn/head/predictions/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/range#dnn/head/predictions/ExpandDims/dim*
T0*
_output_shapes

:
g
%dnn/head/predictions/Tile/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
¤
#dnn/head/predictions/Tile/multiplesPack"dnn/head/predictions/strided_slice%dnn/head/predictions/Tile/multiples/1*
T0*
N*
_output_shapes
:

dnn/head/predictions/TileTilednn/head/predictions/ExpandDims#dnn/head/predictions/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
dnn/head/predictions/Shape_1Shapednn/logits/BiasAdd*
T0*
_output_shapes
:
t
*dnn/head/predictions/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/predictions/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/predictions/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
$dnn/head/predictions/strided_slice_1StridedSlicednn/head/predictions/Shape_1*dnn/head/predictions/strided_slice_1/stack,dnn/head/predictions/strided_slice_1/stack_1,dnn/head/predictions/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
d
"dnn/head/predictions/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"dnn/head/predictions/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
d
"dnn/head/predictions/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
­
dnn/head/predictions/range_1Range"dnn/head/predictions/range_1/start"dnn/head/predictions/range_1/limit"dnn/head/predictions/range_1/delta*
_output_shapes
:
l
dnn/head/predictions/AsStringAsStringdnn/head/predictions/range_1*
T0*
_output_shapes
:
g
%dnn/head/predictions/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

!dnn/head/predictions/ExpandDims_1
ExpandDimsdnn/head/predictions/AsString%dnn/head/predictions/ExpandDims_1/dim*
_output_shapes

:*
T0
i
'dnn/head/predictions/Tile_1/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ş
%dnn/head/predictions/Tile_1/multiplesPack$dnn/head/predictions/strided_slice_1'dnn/head/predictions/Tile_1/multiples/1*
T0*
N*
_output_shapes
:

dnn/head/predictions/Tile_1Tile!dnn/head/predictions/ExpandDims_1%dnn/head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
%dnn/head/predictions/ExpandDims_2/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
¨
!dnn/head/predictions/ExpandDims_2
ExpandDimsdnn/head/predictions/class_ids%dnn/head/predictions/ExpandDims_2/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 dnn/head/predictions/str_classesAsString!dnn/head/predictions/ExpandDims_2*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
"dnn/head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
_output_shapes
:*
T0
f
dnn/head/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
h
dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ú
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
V
dnn/head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
V
dnn/head/range/limitConst*
_output_shapes
: *
value	B :*
dtype0
V
dnn/head/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
u
dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*
_output_shapes
:
R
dnn/head/AsStringAsStringdnn/head/range*
T0*
_output_shapes
:
Y
dnn/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
v
dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*
_output_shapes

:*
T0
[
dnn/head/Tile/multiples/1Const*
_output_shapes
: *
value	B :*
dtype0

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
N*
_output_shapes
:*
T0
u
dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:

X
save/IdentityIdentitysave/Read/ReadVariableOp*
_output_shapes
:
*
T0
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:

`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:

d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:

t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:

\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:

`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:


`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
_output_shapes

:

*
T0
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
_output_shapes

:

*
T0
m
save/Read_4/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
_output_shapes
:*
T0
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:
s
save/Read_5/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:

a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:

f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:


save/StringJoin/inputs_1Const*<
value3B1 B+_temp_6e88919c86af4cc5ad88f450e549a04e/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
_output_shapes
:
*
dtype0
l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
_output_shapes
:
*
T0
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:


save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:


save/Read_8/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:

l
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes
:


save/Read_9/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:


p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:


}
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:

save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:

q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

ű
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
ź
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*[
valueRBPB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3*
dtype0*
_output_shapes
:
ü
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_13save/Identity_15save/Identity_17save/Identity_19save/Identity_21save/Identity_23"/device:CPU:0*
dtypes

2
¨
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
Ô
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
¨
save/Identity_24Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
_output_shapes
: *
T0	*
_class
loc:@global_step
(
save/restore_shardNoOp^save/Assign
ţ
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
ż
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*[
valueRBPB10 0,10B4 10 0,4:0,10B10 0,10B10 10 0,10:0,10B3 0,3B10 3 0,10:0,3*
dtype0*
_output_shapes
:
Ö
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*D
_output_shapes2
0:
:
:
:

::
*
dtypes

2
b
save/Identity_25Identitysave/RestoreV2_1"/device:CPU:0*
_output_shapes
:
*
T0
v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_25"/device:CPU:0*
dtype0
h
save/Identity_26Identitysave/RestoreV2_1:1"/device:CPU:0*
_output_shapes

:
*
T0
z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_26"/device:CPU:0*
dtype0
d
save/Identity_27Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:

x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_27"/device:CPU:0*
dtype0
h
save/Identity_28Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_28"/device:CPU:0*
dtype0
d
save/Identity_29Identitysave/RestoreV2_1:4"/device:CPU:0*
_output_shapes
:*
T0
q
save/AssignVariableOp_4AssignVariableOpdnn/logits/bias/part_0save/Identity_29"/device:CPU:0*
dtype0
h
save/Identity_30Identitysave/RestoreV2_1:5"/device:CPU:0*
_output_shapes

:
*
T0
s
save/AssignVariableOp_5AssignVariableOpdnn/logits/kernel/part_0save/Identity_30"/device:CPU:0*
dtype0
Ĺ
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"&?
save/Const:0save/Identity_24:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"ß 
cond_contextÎ Ë 
Ź
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *Ŕ
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0

"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*Ż
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ę
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *Ř
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
ˇ
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*Ĺ
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
Â
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *Đ
dnn/logits/BiasAdd:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
Ż
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*˝
dnn/logits/BiasAdd:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0"%
saved_model_main_op


group_deps"
	summariesô
ń
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"­

trainable_variables


ě
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
ě
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
ł
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"˙

	variablesń
î

Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
ě
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
ě
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
ł
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08*
predictú
/
SepalLength 
SepalLength:0˙˙˙˙˙˙˙˙˙
/
PetalLength 
PetalLength:0˙˙˙˙˙˙˙˙˙
-

PetalWidth
PetalWidth:0˙˙˙˙˙˙˙˙˙
-

SepalWidth
SepalWidth:0˙˙˙˙˙˙˙˙˙L
probabilities;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙5
logits+
dnn/logits/BiasAdd:0˙˙˙˙˙˙˙˙˙G
	class_ids:
#dnn/head/predictions/ExpandDims_2:0	˙˙˙˙˙˙˙˙˙D
classes9
"dnn/head/predictions/str_classes:0˙˙˙˙˙˙˙˙˙C
all_class_ids2
dnn/head/predictions/Tile:0˙˙˙˙˙˙˙˙˙C
all_classes4
dnn/head/predictions/Tile_1:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict