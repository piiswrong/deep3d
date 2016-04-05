/*!
 * Copyright (c) 2015 by Contributors
 * \file depth_dot.cu
 * \brief
 * \author Junyuan Xie
*/
#include "./depth_dot-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace depthdot {
__global__ void DepthDotForward(real_t *data, real_t *label, real_t *out, int s0, int s1) {
  extern __shared__ real_t slabel[];
  const int C = gridDim.y;
  const int H = gridDim.x;
  const int W = blockDim.x;

  const int n = blockIdx.z;
  const int c = blockIdx.y;
  const int h = blockIdx.x;
  const int w = threadIdx.x;

  const int i = ((n*C+c)*H+h)*W+w;
  const int stride = s1 - s0 + 1;

  if (w < W) slabel[w] = label[i];
  __syncthreads();

  real_t o = 0.0f;
  for (int j = s0; j < s1; ++j) {
    real_t l;
    if (w - j < 0) {
      l = slabel[0];
    } else if (w - j >= W) {
      l = slabel[W-1];
    } else {
      l = slabel[w-j];
    }
    int i_data = ((n*stride + j - s0 + 1)*H + h)*W + w;
    real_t d = data[i_data];
    o += l*d;
  }
  out[i] = o;
}

__global__ void DepthDotBackward(real_t *label, real_t *out_grad, real_t *in_grad, int C, int s0, int s1) {
  extern __shared__ real_t shared[];
  const int H = gridDim.x;
  const int W = blockDim.x;
  real_t *slabel = shared;
  real_t *sgrad = shared + C*W;

  const int n = blockIdx.z;
  const int h = blockIdx.x;
  const int w = threadIdx.x;

  const int stride = s1 - s0 + 1;

  for (int c = 0; c < C; ++c) {
    const int i = ((n*C+c)*H+h)*W+w;
    if (w < W) {
      slabel[c*W + w] = label[i];
      sgrad[c*W + w] = out_grad[i];
    }
  }
  __syncthreads();

  for (int j = s0; j < s1; ++j) {
    real_t o = 0.0f;
    for (int c = 0; c < C; ++c) {
      real_t l;
      if (w - j < 0) {
        l = slabel[c*W];
      } else if (w - j >= W) {
        l = slabel[c*W+W-1];
      } else {
        l = slabel[c*W+w-j];
      }
      real_t og = sgrad[c*W+w];
      o += l*og;
    }
    in_grad[((n*stride+j-s0+1)*H+h)*W+w] = o;
  }
  in_grad[((n*stride)*H+h)*W+w] = 0;
}

template<int MAX_D>
__global__ void DepthDotForwardUpSample(real_t *data, real_t *label, real_t *out, int C, int H, int W, int s0, int s1, int upsample) {
  extern __shared__ real_t shared[];

  const int D = s1 - s0 + 1;
  const int uH = H*upsample;
  const int uW = W*upsample;
  const int i = blockIdx.x;
  const int h = i%H;
  const int n = i/H;
  const int w = threadIdx.x;

  real_t *slabel = shared;
  real_t *sout = shared + C*uW;

  real_t sdata[MAX_D+1][2][2];

  for (int d = 0; d < D; ++d) {
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      #pragma unroll
      for (int ww = 0; ww < 2; ++ww) {
        if (w+ww < W && h+hh < H && d < D) {
          sdata[d][hh][ww] = data[((n*D+d)*H+h+hh)*W+w+ww];
        } else {
          sdata[d][hh][ww] = sdata[d][0][0];
        }
      }
    }
  }
  sdata[D][0][0] = sdata[D-1][0][0];
  sdata[D][0][1] = sdata[D-1][0][1];
  sdata[D][1][0] = sdata[D-1][1][0];
  sdata[D][1][1] = sdata[D-1][1][1];

  for (int hh = 0; hh < upsample; ++hh) {
    for (int c = 0; c < C; ++c) {
      for (int u = 0; u < upsample; ++u) {
        slabel[c*uW+u*W+w] = label[((n*C+c)*uH+h*upsample+hh)*uW+u*W+w];
        sout[c*uW+u*W+w] = 0.f;
      }
    }
    __syncthreads();
    for (int ww = 0; ww < upsample; ++ww) {
      int idx = w*upsample+ww;
      real_t wd = static_cast<real_t>(ww)/upsample;
      real_t hd = static_cast<real_t>(hh)/upsample;
      real_t norm = (sdata[0][0][0]*(1.f-wd) + sdata[0][0][1]*wd) * (1.f-hd) + (sdata[0][1][0]*(1.f-wd) + sdata[0][1][1]*wd) * hd;
      for (int d = 1; d < D; ++d) {
        for (int dd = 0; dd < upsample; ++dd) {
          int shift = (d-1+s0)*upsample + dd;
          real_t zd = static_cast<real_t>(dd)/upsample;
          real_t tri00 = sdata[d][0][0]*(1.f - wd) + sdata[d][0][1]*wd;
          real_t tri01 = sdata[d][1][0]*(1.f - wd) + sdata[d][1][1]*wd;
          real_t tri10 = sdata[d+1][0][0]*(1.f - wd) + sdata[d+1][0][1]*wd;
          real_t tri11 = sdata[d+1][1][0]*(1.f - wd) + sdata[d+1][1][1]*wd;
          real_t tri0 = tri00*(1.f - hd) + tri01*hd;
          real_t tri1 = tri10*(1.f - hd) + tri11*hd;
          real_t tri = tri0*(1.f - zd) + tri1*zd;
          tri *= dd==0;
          norm += tri;
          if (idx - shift < 0) {
            for (int c = 0; c < C; ++c) {
              sout[c*uW+idx] += tri * slabel[c*uW];
            }
          } else if (idx - shift >= uW) {
            for (int c = 0; c < C; ++c) {
              sout[c*uW+idx] += tri * slabel[c*uW + uW - 1];
            }
          } else {
            for (int c = 0; c < C; ++c) {
              sout[c*uW+idx] += tri * slabel[c*uW + idx - shift];
            }
          }
        }
      }
      for (int c = 0; c < C; ++c) sout[c*uW+idx] /= norm;
    }
    __syncthreads();
    for (int c = 0; c < C; ++c) {
      for (int u = 0; u < upsample; ++u) {
        out[((n*C+c)*uH+h*upsample+hh)*uW+u*W+w] = sout[c*uW+u*W+w];
      }
    }
    __syncthreads();
  }
}
}  // depthdot

template<typename xpu>
class DepthDotOp : public Operator {
 public:
  explicit DepthDotOp(DepthDotParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[depthdot::kOut], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[depthdot::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> label = in_data[depthdot::kLabel].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[depthdot::kOut].FlatTo2D<xpu, real_t>(s);
    if (param_.upsample > 1) {
      TShape dshape = in_data[depthdot::kData].shape_;
      TShape lshape = in_data[depthdot::kLabel].shape_;
      dim3 dimBlock(dshape[3]);
      dim3 dimGrid(dshape[2]*dshape[0]);
      CHECK_LE(param_.scale[1]-param_.scale[0]+1, 33);
      mxnet::op::depthdot::DepthDotForwardUpSample<33><<<dimGrid, dimBlock, 2*param_.upsample*lshape[1]*dshape[3]*sizeof(real_t), Stream<gpu>::GetStream(s)>>>(
        data.dptr_, label.dptr_, out.dptr_, lshape[1], dshape[2], dshape[3], param_.scale[0], param_.scale[1], param_.upsample);
    } else {
      TShape oshape = out_data[depthdot::kOut].shape_;
      dim3 dimBlock(oshape[3]);
      dim3 dimGrid(oshape[2], oshape[1], oshape[0]);
      mxnet::op::depthdot::DepthDotForward<<<dimGrid, dimBlock, oshape[3]*sizeof(real_t), Stream<gpu>::GetStream(s)>>>(
        data.dptr_, label.dptr_, out.dptr_, param_.scale[0], param_.scale[1]);
    }
    cudaStreamSynchronize(Stream<gpu>::GetStream(s));
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[depthdot::kData], kWriteTo);
    CHECK_EQ(req[depthdot::kLabel], kNullOp);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> label = in_data[depthdot::kLabel].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> igrad = in_grad[depthdot::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> ograd = out_grad[depthdot::kOut].FlatTo2D<xpu, real_t>(s);
    TShape oshape = out_grad[depthdot::kOut].shape_;
    dim3 dimBlock(oshape[3]);
    dim3 dimGrid(oshape[2], 1, oshape[0]);
    mxnet::op::depthdot::DepthDotBackward<<<dimGrid, dimBlock, 2*oshape[3]*oshape[1]*sizeof(real_t), Stream<gpu>::GetStream(s)>>>(
      label.dptr_, ograd.dptr_, igrad.dptr_, oshape[1], param_.scale[0], param_.scale[1]);
    cudaStreamSynchronize(Stream<gpu>::GetStream(s));
  }

 private:
  DepthDotParam param_;
};  // class DepthDotOp

template<>
Operator *CreateOp<gpu>(DepthDotParam param) {
  return new DepthDotOp<gpu>(param);
}
}  // op
}  // namespace mxnet
