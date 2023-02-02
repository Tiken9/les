# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: les/backend_solvers/backend_solvers.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from les.mp_model import mp_model_pb2 as les_dot_mp__model_dot_mp__model__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='les/backend_solvers/backend_solvers.proto',
  package='les.backend_solvers',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n)les/backend_solvers/backend_solvers.proto\x12\x13les.backend_solvers\x1a\x1bles/mp_model/mp_model.proto*z\n\rBackendSolver\x12\x07\n\x03\x43LP\x10\x00\x12\x10\n\x0c\x44UMMY_SOLVER\x10\x01\x12\x1e\n\x1a\x46RAKTIONAL_KNAPSACK_SOLVER\x10\x02\x12\x08\n\x04GLPK\x10\x03\x12\x0c\n\x08LP_SOLVE\x10\x04\x12\x08\n\x04SCIP\x10\x05\x12\x0c\n\x08SYMPHONY\x10\x06')
  ,
  dependencies=[les_dot_mp__model_dot_mp__model__pb2.DESCRIPTOR,])

_BACKENDSOLVER = _descriptor.EnumDescriptor(
  name='BackendSolver',
  full_name='les.backend_solvers.BackendSolver',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CLP', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DUMMY_SOLVER', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FRAKTIONAL_KNAPSACK_SOLVER', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GLPK', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LP_SOLVE', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SCIP', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYMPHONY', index=6, number=6,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=95,
  serialized_end=217,
)
_sym_db.RegisterEnumDescriptor(_BACKENDSOLVER)

BackendSolver = enum_type_wrapper.EnumTypeWrapper(_BACKENDSOLVER)
CLP = 0
DUMMY_SOLVER = 1
FRAKTIONAL_KNAPSACK_SOLVER = 2
GLPK = 3
LP_SOLVE = 4
SCIP = 5
SYMPHONY = 6


DESCRIPTOR.enum_types_by_name['BackendSolver'] = _BACKENDSOLVER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
