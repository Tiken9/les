# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: les/decomposers/decomposers.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from les.mp_model import mp_model_pb2 as les_dot_mp__model_dot_mp__model__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!les/decomposers/decomposers.proto\x12\x0fles.decomposers\x1a\x1bles/mp_model/mp_model.proto\"j\n\x14\x44\x65\x63omposerParameters\x12R\n\ndecomposer\x18\x01 \x02(\x0e\x32\x1b.les.decomposers.Decomposer:!QUASIBLOCK_FINKELSTEIN_DECOMPOSER*N\n\nDecomposer\x12%\n!QUASIBLOCK_FINKELSTEIN_DECOMPOSER\x10\x00\x12\x19\n\x15MAX_CLIQUE_DECOMPOSER\x10\x01:j\n\x15\x64\x65\x63omposer_parameters\x12$.les.mp_model.OptimizationParameters\x18\x65 \x01(\x0b\x32%.les.decomposers.DecomposerParameters')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'les.decomposers.decomposers_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  les_dot_mp__model_dot_mp__model__pb2.OptimizationParameters.RegisterExtension(decomposer_parameters)

  DESCRIPTOR._options = None
  _DECOMPOSER._serialized_start=191
  _DECOMPOSER._serialized_end=269
  _DECOMPOSERPARAMETERS._serialized_start=83
  _DECOMPOSERPARAMETERS._serialized_end=189
# @@protoc_insertion_point(module_scope)
