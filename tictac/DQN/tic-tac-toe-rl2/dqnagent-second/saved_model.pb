??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?	*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	?*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/m
?
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?	*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:	*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	?*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/v
?
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?	*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
R
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?
,iter

-beta_1

.beta_2
	/decay
0learning_ratemYmZm[m\m]m^v_v`vavbvcvd
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
1layer_metrics
2layer_regularization_losses
3non_trainable_variables
	regularization_losses
4metrics

trainable_variables
	variables

5layers
 
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
6layer_metrics
7layer_regularization_losses
8non_trainable_variables
regularization_losses
9metrics
trainable_variables
	variables

:layers
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
>metrics
trainable_variables
	variables

?layers
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses
Cmetrics
trainable_variables
	variables

Dlayers
 
 
 
?
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables
 regularization_losses
Hmetrics
!trainable_variables
"	variables

Ilayers
 
 
 
?
Jlayer_metrics
Klayer_regularization_losses
Lnon_trainable_variables
$regularization_losses
Mmetrics
%trainable_variables
&	variables

Nlayers
 
 
 
?
Olayer_metrics
Player_regularization_losses
Qnon_trainable_variables
(regularization_losses
Rmetrics
)trainable_variables
*	variables

Slayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

T0
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Utotal
	Vcount
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

W	variables
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense_6/kerneldense_6/biasdense_8/kerneldense_8/biasdense_7/kerneldense_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14273138
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_14273492
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_14273577
?:
?

!__inference__traced_save_14273492
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f449ad086bcb4cef882679f4c8f1b705/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:?:	?::	?	:	: : : : : : : :	?:?:	?::	?	:	:	?:?:	?::	?	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?	: 

_output_shapes
:	:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?	: 

_output_shapes
:	:

_output_shapes
: 
?
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273010
input_3
dense_6_14272861
dense_6_14272863
dense_8_14272887
dense_8_14272889
dense_7_14272913
dense_7_14272915
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_6_14272861dense_6_14272863*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_142728502!
dense_6/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_14272887dense_8_14272889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_142728762!
dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_14272913dense_7_14272915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_142729022!
dense_7/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729282
lambda_4/PartitionedCall?
lambda_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729662
lambda_5/PartitionedCall?
add_2/PartitionedCallPartitionedCall!lambda_4/PartitionedCall:output:0!lambda_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_142730002
add_2/PartitionedCall?
IdentityIdentityadd_2/PartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273357

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice_1_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Constx
MeanMeanstrided_slice_1:output:0Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
Meanj
subSubstrided_slice:output:0Mean:output:0*
T0*'
_output_shapes
:?????????	2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
T
(__inference_add_2_layer_call_fn_14273394
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_142730002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????	:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/1
?	
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_14272928

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_4_layer_call_fn_14273337

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729282
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
C__inference_add_2_layer_call_and_return_conditional_losses_14273000

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:?????????	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????	:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
E__inference_dense_6_layer_call_and_return_conditional_losses_14272850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273032
input_3
dense_6_14273013
dense_6_14273015
dense_8_14273018
dense_8_14273020
dense_7_14273023
dense_7_14273025
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_6_14273013dense_6_14273015*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_142728502!
dense_6/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_14273018dense_8_14273020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_142728762!
dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_14273023dense_7_14273025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_142729022!
dense_7/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729382
lambda_4/PartitionedCall?
lambda_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729812
lambda_5/PartitionedCall?
add_2/PartitionedCallPartitionedCall!lambda_4/PartitionedCall:output:0!lambda_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_142730002
add_2/PartitionedCall?
IdentityIdentityadd_2/PartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
&__inference_signature_wrapper_14273138
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_142728352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
G
+__inference_lambda_4_layer_call_fn_14273342

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_7_layer_call_and_return_conditional_losses_14273284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_functional_5_layer_call_fn_14273111
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_functional_5_layer_call_and_return_conditional_losses_142730962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?k
?
$__inference__traced_restore_14273577
file_prefix#
assignvariableop_dense_6_kernel#
assignvariableop_1_dense_6_bias%
!assignvariableop_2_dense_7_kernel#
assignvariableop_3_dense_7_bias%
!assignvariableop_4_dense_8_kernel#
assignvariableop_5_dense_8_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count-
)assignvariableop_13_adam_dense_6_kernel_m+
'assignvariableop_14_adam_dense_6_bias_m-
)assignvariableop_15_adam_dense_7_kernel_m+
'assignvariableop_16_adam_dense_7_bias_m-
)assignvariableop_17_adam_dense_8_kernel_m+
'assignvariableop_18_adam_dense_8_bias_m-
)assignvariableop_19_adam_dense_6_kernel_v+
'assignvariableop_20_adam_dense_6_bias_v-
)assignvariableop_21_adam_dense_7_kernel_v+
'assignvariableop_22_adam_dense_7_bias_v-
)assignvariableop_23_adam_dense_8_kernel_v+
'assignvariableop_24_adam_dense_8_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_8_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_8_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_6_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_6_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_7_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_7_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_8_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_6_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_6_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_7_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_7_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_8_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_8_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_14272938

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_6_layer_call_and_return_conditional_losses_14273265

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_5_layer_call_fn_14273377

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_14272966

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice_1_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Constx
MeanMeanstrided_slice_1:output:0Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
Meanj
subSubstrided_slice:output:0Mean:output:0*
T0*'
_output_shapes
:?????????	2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?8
?
#__inference__wrapped_model_14272835
input_37
3functional_5_dense_6_matmul_readvariableop_resource8
4functional_5_dense_6_biasadd_readvariableop_resource7
3functional_5_dense_8_matmul_readvariableop_resource8
4functional_5_dense_8_biasadd_readvariableop_resource7
3functional_5_dense_7_matmul_readvariableop_resource8
4functional_5_dense_7_biasadd_readvariableop_resource
identity??
*functional_5/dense_6/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*functional_5/dense_6/MatMul/ReadVariableOp?
functional_5/dense_6/MatMulMatMulinput_32functional_5/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
functional_5/dense_6/MatMul?
+functional_5/dense_6/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+functional_5/dense_6/BiasAdd/ReadVariableOp?
functional_5/dense_6/BiasAddBiasAdd%functional_5/dense_6/MatMul:product:03functional_5/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
functional_5/dense_6/BiasAdd?
functional_5/dense_6/ReluRelu%functional_5/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
functional_5/dense_6/Relu?
*functional_5/dense_8/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02,
*functional_5/dense_8/MatMul/ReadVariableOp?
functional_5/dense_8/MatMulMatMul'functional_5/dense_6/Relu:activations:02functional_5/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
functional_5/dense_8/MatMul?
+functional_5/dense_8/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_8_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02-
+functional_5/dense_8/BiasAdd/ReadVariableOp?
functional_5/dense_8/BiasAddBiasAdd%functional_5/dense_8/MatMul:product:03functional_5/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
functional_5/dense_8/BiasAdd?
*functional_5/dense_7/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*functional_5/dense_7/MatMul/ReadVariableOp?
functional_5/dense_7/MatMulMatMul'functional_5/dense_6/Relu:activations:02functional_5/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_5/dense_7/MatMul?
+functional_5/dense_7/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_5/dense_7/BiasAdd/ReadVariableOp?
functional_5/dense_7/BiasAddBiasAdd%functional_5/dense_7/MatMul:product:03functional_5/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_5/dense_7/BiasAdd?
)functional_5/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)functional_5/lambda_4/strided_slice/stack?
+functional_5/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+functional_5/lambda_4/strided_slice/stack_1?
+functional_5/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+functional_5/lambda_4/strided_slice/stack_2?
#functional_5/lambda_4/strided_sliceStridedSlice%functional_5/dense_7/BiasAdd:output:02functional_5/lambda_4/strided_slice/stack:output:04functional_5/lambda_4/strided_slice/stack_1:output:04functional_5/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2%
#functional_5/lambda_4/strided_slice?
$functional_5/lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$functional_5/lambda_4/ExpandDims/dim?
 functional_5/lambda_4/ExpandDims
ExpandDims,functional_5/lambda_4/strided_slice:output:0-functional_5/lambda_4/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2"
 functional_5/lambda_4/ExpandDims?
)functional_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2+
)functional_5/lambda_5/strided_slice/stack?
+functional_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+functional_5/lambda_5/strided_slice/stack_1?
+functional_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+functional_5/lambda_5/strided_slice/stack_2?
#functional_5/lambda_5/strided_sliceStridedSlice%functional_5/dense_8/BiasAdd:output:02functional_5/lambda_5/strided_slice/stack:output:04functional_5/lambda_5/strided_slice/stack_1:output:04functional_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2%
#functional_5/lambda_5/strided_slice?
+functional_5/lambda_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+functional_5/lambda_5/strided_slice_1/stack?
-functional_5/lambda_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2/
-functional_5/lambda_5/strided_slice_1/stack_1?
-functional_5/lambda_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-functional_5/lambda_5/strided_slice_1/stack_2?
%functional_5/lambda_5/strided_slice_1StridedSlice%functional_5/dense_8/BiasAdd:output:04functional_5/lambda_5/strided_slice_1/stack:output:06functional_5/lambda_5/strided_slice_1/stack_1:output:06functional_5/lambda_5/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2'
%functional_5/lambda_5/strided_slice_1?
functional_5/lambda_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
functional_5/lambda_5/Const?
functional_5/lambda_5/MeanMean.functional_5/lambda_5/strided_slice_1:output:0$functional_5/lambda_5/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
functional_5/lambda_5/Mean?
functional_5/lambda_5/subSub,functional_5/lambda_5/strided_slice:output:0#functional_5/lambda_5/Mean:output:0*
T0*'
_output_shapes
:?????????	2
functional_5/lambda_5/sub?
functional_5/add_2/addAddV2)functional_5/lambda_4/ExpandDims:output:0functional_5/lambda_5/sub:z:0*
T0*'
_output_shapes
:?????????	2
functional_5/add_2/addn
IdentityIdentityfunctional_5/add_2/add:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_14272981

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice_1_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Constx
MeanMeanstrided_slice_1:output:0Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
Meanj
subSubstrided_slice:output:0Mean:output:0*
T0*'
_output_shapes
:?????????	2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

*__inference_dense_8_layer_call_fn_14273312

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_142728762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_6_layer_call_fn_14273274

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_142728502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273372

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
strided_slice_1_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Constx
MeanMeanstrided_slice_1:output:0Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
Meanj
subSubstrided_slice:output:0Mean:output:0*
T0*'
_output_shapes
:?????????	2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?-
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273179

inputs*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_6/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_8/BiasAdd?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAdd?
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_4/strided_slice/stack?
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_4/strided_slice/stack_1?
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_4/strided_slice/stack_2?
lambda_4/strided_sliceStridedSlicedense_7/BiasAdd:output:0%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_4/strided_slice}
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimslambda_4/strided_slice:output:0 lambda_4/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
lambda_4/ExpandDims?
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_5/strided_slice/stack?
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
lambda_5/strided_slice/stack_1?
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_5/strided_slice/stack_2?
lambda_5/strided_sliceStridedSlicedense_8/BiasAdd:output:0%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
lambda_5/strided_slice?
lambda_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
lambda_5/strided_slice_1/stack?
 lambda_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 lambda_5/strided_slice_1/stack_1?
 lambda_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_5/strided_slice_1/stack_2?
lambda_5/strided_slice_1StridedSlicedense_8/BiasAdd:output:0'lambda_5/strided_slice_1/stack:output:0)lambda_5/strided_slice_1/stack_1:output:0)lambda_5/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
lambda_5/strided_slice_1q
lambda_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
lambda_5/Const?
lambda_5/MeanMean!lambda_5/strided_slice_1:output:0lambda_5/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
lambda_5/Mean?
lambda_5/subSublambda_5/strided_slice:output:0lambda_5/Mean:output:0*
T0*'
_output_shapes
:?????????	2
lambda_5/sub?
	add_2/addAddV2lambda_4/ExpandDims:output:0lambda_5/sub:z:0*
T0*'
_output_shapes
:?????????	2
	add_2/adda
IdentityIdentityadd_2/add:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_dense_7_layer_call_fn_14273293

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_142729022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_14273303

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273057

inputs
dense_6_14273038
dense_6_14273040
dense_8_14273043
dense_8_14273045
dense_7_14273048
dense_7_14273050
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_14273038dense_6_14273040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_142728502!
dense_6/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_14273043dense_8_14273045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_142728762!
dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_14273048dense_7_14273050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_142729022!
dense_7/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729282
lambda_4/PartitionedCall?
lambda_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729662
lambda_5/PartitionedCall?
add_2/PartitionedCallPartitionedCall!lambda_4/PartitionedCall:output:0!lambda_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_142730002
add_2/PartitionedCall?
IdentityIdentityadd_2/PartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
o
C__inference_add_2_layer_call_and_return_conditional_losses_14273388
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????	2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????	:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/1
?-
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273220

inputs*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity??
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_6/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_8/BiasAdd?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAdd?
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_4/strided_slice/stack?
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2 
lambda_4/strided_slice/stack_1?
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_4/strided_slice/stack_2?
lambda_4/strided_sliceStridedSlicedense_7/BiasAdd:output:0%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_4/strided_slice}
lambda_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda_4/ExpandDims/dim?
lambda_4/ExpandDims
ExpandDimslambda_4/strided_slice:output:0 lambda_4/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2
lambda_4/ExpandDims?
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lambda_5/strided_slice/stack?
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2 
lambda_5/strided_slice/stack_1?
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
lambda_5/strided_slice/stack_2?
lambda_5/strided_sliceStridedSlicedense_8/BiasAdd:output:0%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
lambda_5/strided_slice?
lambda_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
lambda_5/strided_slice_1/stack?
 lambda_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 lambda_5/strided_slice_1/stack_1?
 lambda_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 lambda_5/strided_slice_1/stack_2?
lambda_5/strided_slice_1StridedSlicedense_8/BiasAdd:output:0'lambda_5/strided_slice_1/stack:output:0)lambda_5/strided_slice_1/stack_1:output:0)lambda_5/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????	*

begin_mask*
end_mask2
lambda_5/strided_slice_1q
lambda_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
lambda_5/Const?
lambda_5/MeanMean!lambda_5/strided_slice_1:output:0lambda_5/Const:output:0*
T0*
_output_shapes

:*
	keep_dims(2
lambda_5/Mean?
lambda_5/subSublambda_5/strided_slice:output:0lambda_5/Mean:output:0*
T0*'
_output_shapes
:?????????	2
lambda_5/sub?
	add_2/addAddV2lambda_4/ExpandDims:output:0lambda_5/sub:z:0*
T0*'
_output_shapes
:?????????	2
	add_2/adda
IdentityIdentityadd_2/add:z:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_7_layer_call_and_return_conditional_losses_14272902

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_functional_5_layer_call_fn_14273254

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_functional_5_layer_call_and_return_conditional_losses_142730962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273322

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_lambda_5_layer_call_fn_14273382

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????	:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
/__inference_functional_5_layer_call_fn_14273237

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_functional_5_layer_call_and_return_conditional_losses_142730572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273096

inputs
dense_6_14273077
dense_6_14273079
dense_8_14273082
dense_8_14273084
dense_7_14273087
dense_7_14273089
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_14273077dense_6_14273079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_142728502!
dense_6/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_8_14273082dense_8_14273084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_142728762!
dense_8/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_14273087dense_7_14273089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_142729022!
dense_7/StatefulPartitionedCall?
lambda_4/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_4_layer_call_and_return_conditional_losses_142729382
lambda_4/PartitionedCall?
lambda_5/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lambda_5_layer_call_and_return_conditional_losses_142729812
lambda_5/PartitionedCall?
add_2/PartitionedCallPartitionedCall!lambda_4/PartitionedCall:output:0!lambda_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_142730002
add_2/PartitionedCall?
IdentityIdentityadd_2/PartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_8_layer_call_and_return_conditional_losses_14272876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_functional_5_layer_call_fn_14273072
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_functional_5_layer_call_and_return_conditional_losses_142730572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?	
b
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273332

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0?????????9
add_20
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:??
?=
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
e__call__
*f&call_and_return_all_conditional_losses
g_default_save_signature"?:
_tf_keras_network?:{"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 27]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 243, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMaAAAAdABqAaACfABkAGQAhQJkAWYCGQBkAqECUwAp\nA07pAAAAAOn/////KQPaBWtlcmFz2gdiYWNrZW5k2gtleHBhbmRfZGltcykB2gFzqQByBwAAAPpE\nL1VzZXJzL212b2xrZS9EZXNrdG9wL0RlZXAgUS1MZWFybmluZy90aWMtdGFjLXRvZS9nYW1lL2Fn\nZW50cy9kcW4ucHnaCDxsYW1iZGE+ZAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_4", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHM0AAAAfABkAGQAhQJkAGQAhQJmAhkAdABqAWoCfABk\nAGQAhQJkAGQAhQJmAhkAZAFkAo0CGABTACkDTlQpAdoIa2VlcGRpbXMpA9oFa2VyYXPaB2JhY2tl\nbmTaBG1lYW4pAdoBYakAcgYAAAD6RC9Vc2Vycy9tdm9sa2UvRGVza3RvcC9EZWVwIFEtTGVhcm5p\nbmcvdGljLXRhYy10b2UvZ2FtZS9hZ2VudHMvZHFuLnB52gg8bGFtYmRhPmgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_5", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["lambda_4", 0, 0, {}], ["lambda_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["add_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 27]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 243, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMaAAAAdABqAaACfABkAGQAhQJkAWYCGQBkAqECUwAp\nA07pAAAAAOn/////KQPaBWtlcmFz2gdiYWNrZW5k2gtleHBhbmRfZGltcykB2gFzqQByBwAAAPpE\nL1VzZXJzL212b2xrZS9EZXNrdG9wL0RlZXAgUS1MZWFybmluZy90aWMtdGFjLXRvZS9nYW1lL2Fn\nZW50cy9kcW4ucHnaCDxsYW1iZGE+ZAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_4", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHM0AAAAfABkAGQAhQJkAGQAhQJmAhkAdABqAWoCfABk\nAGQAhQJkAGQAhQJmAhkAZAFkAo0CGABTACkDTlQpAdoIa2VlcGRpbXMpA9oFa2VyYXPaB2JhY2tl\nbmTaBG1lYW4pAdoBYakAcgYAAAD6RC9Vc2Vycy9tdm9sa2UvRGVza3RvcC9EZWVwIFEtTGVhcm5p\nbmcvdGljLXRhYy10b2UvZ2FtZS9hZ2VudHMvZHFuLnB52gg8bGFtYmRhPmgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_5", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["lambda_4", 0, 0, {}], ["lambda_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["add_2", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 27]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 27]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 243, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 27}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 243}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 243]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": 42}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 243}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 243]}}
?
 regularization_losses
!trainable_variables
"	variables
#	keras_api
n__call__
*o&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAUwAAAHMaAAAAdABqAaACfABkAGQAhQJkAWYCGQBkAqECUwAp\nA07pAAAAAOn/////KQPaBWtlcmFz2gdiYWNrZW5k2gtleHBhbmRfZGltcykB2gFzqQByBwAAAPpE\nL1VzZXJzL212b2xrZS9EZXNrdG9wL0RlZXAgUS1MZWFybmluZy90aWMtdGFjLXRvZS9nYW1lL2Fn\nZW50cy9kcW4ucHnaCDxsYW1iZGE+ZAAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_5", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHM0AAAAfABkAGQAhQJkAGQAhQJmAhkAdABqAWoCfABk\nAGQAhQJkAGQAhQJmAhkAZAFkAo0CGABTACkDTlQpAdoIa2VlcGRpbXMpA9oFa2VyYXPaB2JhY2tl\nbmTaBG1lYW4pAdoBYakAcgYAAAD6RC9Vc2Vycy9tdm9sa2UvRGVza3RvcC9EZWVwIFEtTGVhcm5p\nbmcvdGljLXRhYy10b2UvZ2FtZS9hZ2VudHMvZHFuLnB52gg8bGFtYmRhPmgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "game.agents.dqn", "output_shape": {"class_name": "__tuple__", "items": [9]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
(regularization_losses
)trainable_variables
*	variables
+	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 9]}]}
?
,iter

-beta_1

.beta_2
	/decay
0learning_ratemYmZm[m\m]m^v_v`vavbvcvd"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
1layer_metrics
2layer_regularization_losses
3non_trainable_variables
	regularization_losses
4metrics

trainable_variables
	variables

5layers
e__call__
g_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
!:	?2dense_6/kernel
:?2dense_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6layer_metrics
7layer_regularization_losses
8non_trainable_variables
regularization_losses
9metrics
trainable_variables
	variables

:layers
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
;layer_metrics
<layer_regularization_losses
=non_trainable_variables
regularization_losses
>metrics
trainable_variables
	variables

?layers
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
!:	?	2dense_8/kernel
:	2dense_8/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
@layer_metrics
Alayer_regularization_losses
Bnon_trainable_variables
regularization_losses
Cmetrics
trainable_variables
	variables

Dlayers
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Elayer_metrics
Flayer_regularization_losses
Gnon_trainable_variables
 regularization_losses
Hmetrics
!trainable_variables
"	variables

Ilayers
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jlayer_metrics
Klayer_regularization_losses
Lnon_trainable_variables
$regularization_losses
Mmetrics
%trainable_variables
&	variables

Nlayers
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Olayer_metrics
Player_regularization_losses
Qnon_trainable_variables
(regularization_losses
Rmetrics
)trainable_variables
*	variables

Slayers
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
T0"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Utotal
	Vcount
W	variables
X	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
U0
V1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
&:$	?2Adam/dense_6/kernel/m
 :?2Adam/dense_6/bias/m
&:$	?2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
&:$	?	2Adam/dense_8/kernel/m
:	2Adam/dense_8/bias/m
&:$	?2Adam/dense_6/kernel/v
 :?2Adam/dense_6/bias/v
&:$	?2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
&:$	?	2Adam/dense_8/kernel/v
:	2Adam/dense_8/bias/v
?2?
/__inference_functional_5_layer_call_fn_14273072
/__inference_functional_5_layer_call_fn_14273237
/__inference_functional_5_layer_call_fn_14273254
/__inference_functional_5_layer_call_fn_14273111?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273220
J__inference_functional_5_layer_call_and_return_conditional_losses_14273010
J__inference_functional_5_layer_call_and_return_conditional_losses_14273179
J__inference_functional_5_layer_call_and_return_conditional_losses_14273032?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_14272835?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_3?????????
?2?
*__inference_dense_6_layer_call_fn_14273274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_6_layer_call_and_return_conditional_losses_14273265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_7_layer_call_fn_14273293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_7_layer_call_and_return_conditional_losses_14273284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_8_layer_call_fn_14273312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_8_layer_call_and_return_conditional_losses_14273303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_lambda_4_layer_call_fn_14273337
+__inference_lambda_4_layer_call_fn_14273342?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273332
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273322?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lambda_5_layer_call_fn_14273377
+__inference_lambda_5_layer_call_fn_14273382?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273357
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273372?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_add_2_layer_call_fn_14273394?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_add_2_layer_call_and_return_conditional_losses_14273388?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5B3
&__inference_signature_wrapper_14273138input_3?
#__inference__wrapped_model_14272835i0?-
&?#
!?
input_3?????????
? "-?*
(
add_2?
add_2?????????	?
C__inference_add_2_layer_call_and_return_conditional_losses_14273388?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????	
? "%?"
?
0?????????	
? ?
(__inference_add_2_layer_call_fn_14273394vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????	
? "??????????	?
E__inference_dense_6_layer_call_and_return_conditional_losses_14273265]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_dense_6_layer_call_fn_14273274P/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_dense_7_layer_call_and_return_conditional_losses_14273284]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_7_layer_call_fn_14273293P0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_8_layer_call_and_return_conditional_losses_14273303]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????	
? ~
*__inference_dense_8_layer_call_fn_14273312P0?-
&?#
!?
inputs??????????
? "??????????	?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273010i8?5
.?+
!?
input_3?????????
p

 
? "%?"
?
0?????????	
? ?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273032i8?5
.?+
!?
input_3?????????
p 

 
? "%?"
?
0?????????	
? ?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273179h7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????	
? ?
J__inference_functional_5_layer_call_and_return_conditional_losses_14273220h7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????	
? ?
/__inference_functional_5_layer_call_fn_14273072\8?5
.?+
!?
input_3?????????
p

 
? "??????????	?
/__inference_functional_5_layer_call_fn_14273111\8?5
.?+
!?
input_3?????????
p 

 
? "??????????	?
/__inference_functional_5_layer_call_fn_14273237[7?4
-?*
 ?
inputs?????????
p

 
? "??????????	?
/__inference_functional_5_layer_call_fn_14273254[7?4
-?*
 ?
inputs?????????
p 

 
? "??????????	?
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273322`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
F__inference_lambda_4_layer_call_and_return_conditional_losses_14273332`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
+__inference_lambda_4_layer_call_fn_14273337S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
+__inference_lambda_4_layer_call_fn_14273342S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273357`7?4
-?*
 ?
inputs?????????	

 
p
? "%?"
?
0?????????	
? ?
F__inference_lambda_5_layer_call_and_return_conditional_losses_14273372`7?4
-?*
 ?
inputs?????????	

 
p 
? "%?"
?
0?????????	
? ?
+__inference_lambda_5_layer_call_fn_14273377S7?4
-?*
 ?
inputs?????????	

 
p
? "??????????	?
+__inference_lambda_5_layer_call_fn_14273382S7?4
-?*
 ?
inputs?????????	

 
p 
? "??????????	?
&__inference_signature_wrapper_14273138t;?8
? 
1?.
,
input_3!?
input_3?????????"-?*
(
add_2?
add_2?????????	