# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: les/backend_solvers/backend_solvers.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from les.mp_model import mp_model_pb2 as les_dot_mp__model_dot_mp__model__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n)les/backend_solvers/backend_solvers.proto\x12\x13les.backend_solvers\x1a\x1bles/mp_model/mp_model.proto*z\n\rBackendSolver\x12\x07\n\x03\x43LP\x10\x00\x12\x10\n\x0c\x44UMMY_SOLVER\x10\x01\x12\x1e\n\x1a\x46RAKTIONAL_KNAPSACK_SOLVER\x10\x02\x12\x08\n\x04GLPK\x10\x03\x12\x0c\n\x08LP_SOLVE\x10\x04\x12\x08\n\x04SCIP\x10\x05\x12\x0c\n\x08SYMPHONY\x10\x06')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'les.backend_solvers.backend_solvers_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _BACKENDSOLVER._serialized_start = 95
    _BACKENDSOLVER._serialized_end = 217
# @@protoc_insertion_point(module_scope)
