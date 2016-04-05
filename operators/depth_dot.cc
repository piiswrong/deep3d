/*!
 * Copyright (c) 2015 by Contributors
 * \file depth_dot.cc
 * \brief product sum op
 * \author Junyuan Xie
*/
#include "./depth_dot-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DepthDotParam param) {
  LOG(FATAL) << "only available for gpu";
  return NULL;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DepthDotProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(DepthDotParam);

MXNET_REGISTER_OP_PROPERTY(DepthDot, DepthDotProp)
.describe("Compute dot product along one dim of 2 tensors.")
.add_arguments(DepthDotParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

