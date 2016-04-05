/*!
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_DEPTH_DOT_INL_H_
#define MXNET_OPERATOR_DEPTH_DOT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"


// #include <mshadow/extension.h>

// namespace mshadow {
// namespace expr {
// template<typename SrcExp, typename DType>
// struct TrilinearExp:
//     public MakeTensorExp<TrilinearExp<SrcExp, DType>,
//                          SrcExp, 4, DType> {
//   /*! \brief source oprand */
//   const SrcExp &src_;
//   /*! \brief upsampling scale */
//   int scale_;

//   explicit TrilinearExp(const SrcExp &src, int scale)
//     : src_(src), scale_(scale) {
//   }
// };  // struct TrilinearExp

// template<typename SrcExp, typename DType, int etype>
// inline TrilinearExp<SrcExp, DType>
// trilinear(const Exp<SrcExp, DType, etype> &src, int scale) {
//   TypeCheckPass<ExpInfo<SrcExp>::kDim == 4>
//       ::Error_Expression_Does_Not_Meet_Dimension_Req();
//   return TrilinearExp<SrcExp, DType>(src.self(), scale);
// }

// template<typename SrcExp, typename DType>
// struct Plan<TrilinearExp<SrcExp, DType>, DType> {
//  public:
//   explicit Plan(const TrilinearExp<SrcExp, DType> &e)
//       : src_(MakePlan(e.src_)), scale_(e.scale_) {}
//   MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    
//   }

//  private:
//   Plan<SrcExp, DType> src_;
//   const int scale_;
// };
// }  // expr
// }  // mshadow

namespace mxnet {
namespace op {

namespace depthdot {
enum DepthDotOpInputs {kData, kLabel};
enum DepthDotOpOutputs {kOut};
}  // namespace depthdot

struct DepthDotParam : public dmlc::Parameter<DepthDotParam> {
  TShape scale;
  int upsample;
  DMLC_DECLARE_PARAMETER(DepthDotParam) {
    DMLC_DECLARE_FIELD(scale)
    .describe("scale.");
    DMLC_DECLARE_FIELD(upsample)
    .set_default(1)
    .describe("upsample.");
  }
};

template<typename xpu>
Operator* CreateOp(DepthDotParam type);

#if DMLC_USE_CXX11
class DepthDotProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    TShape lshape = in_shape->at(depthdot::kData);
    CHECK_EQ(lshape.ndim(), 4);
    CHECK_EQ(lshape[1], param_.scale[1]-param_.scale[0]+1);
    SHAPE_ASSIGN_CHECK(*in_shape, depthdot::kLabel, mshadow::Shape4(lshape[0], 3, lshape[2]*param_.upsample, lshape[3]*param_.upsample));
    out_shape->clear();
    out_shape->push_back(in_shape->at(depthdot::kLabel));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DepthDotProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "DepthDot";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[depthdot::kData], in_data[depthdot::kLabel], out_grad[depthdot::kOut]};
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  DepthDotParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_DEPTH_DOT_INL_H_

