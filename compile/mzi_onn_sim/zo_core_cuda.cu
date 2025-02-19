#include <torch/extension.h>

using namespace c10::complex_literals;
using namespace c10_complex_math;

namespace mzi_onn_sim_zo {

static constexpr double QuarterPI = 0.25*M_PI;
static constexpr double modReLUeps = 0.001; // epsilon: Magic number

__device__ c10::complex<float> conj(c10::complex<float> input) {
	return c10::complex<float>(input.real(), -input.imag());
}

__global__ void forwardPSBS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,3> output_a,
	const torch::PackedTensorAccessor32<float,3> angleAB_a,
	const torch::PackedTensorAccessor32<int32_t,3> indexAB_a,
	const torch::PackedTensorAccessor32<float,2> split_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> atten_a)
{
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nLayers = indexAB_a.size(0);
	const int nIndexAB = indexAB_a.size(1);
	const int sample = blockIdx.x;
	const int variation = blockIdx.y;
	const int param = threadIdx.x;
	if (param >= nIndexAB || sample >= nSamples || variation >= nVariation) return;

	auto angleAB = angleAB_a[variation];
	auto output = output_a[variation][sample];
	for (int layer = 0; layer < nLayers; ++layer){
		const auto index = indexAB_a[layer][param];
		const int idx0 = index[0];
		if (idx0 >= 0) {
			const int idx1 = index[1];
			const float PauliP = QuarterPI + 0.5 * split_a[layer][param];
			const float cos_pi_4 = cos(PauliP);
			const float sin_pi_4 = sin(PauliP);
			const auto atten = atten_a[layer];
			const auto angle = angleAB[layer];
			const auto ps_out = atten[idx0] * output[idx0] * exp(1._if*angle[param]);
			const auto in1_atten = atten[idx1] * output[idx1];
			output[idx0] = ps_out * cos_pi_4 + 1._if * in1_atten * sin_pi_4;
			output[idx1] = 1._if * ps_out * sin_pi_4 + in1_atten * cos_pi_4;	
		}
		__syncthreads();
	}
}

at::Tensor forwardPSBS_cuda(const at::Tensor input, const at::Tensor angleAB,
		const at::Tensor indexAB, const at::Tensor split, const at::Tensor atten) {
	const int nVariation = input.size(0);
	const int nSamples = input.size(1);
	const int nIndexAB = indexAB.size(1);
	dim3 grid(nSamples, nVariation);
	dim3 block(nIndexAB);
	at::Tensor output = input.detach().clone();
	const auto output_a = output.packed_accessor32<c10::complex<float>,3>();
	const auto angleAB_a = angleAB.packed_accessor32<float,3>();
	const auto indexAB_a = indexAB.packed_accessor32<int32_t,3>();
	const auto split_a = split.packed_accessor32<float,2>();
	const auto atten_a = atten.packed_accessor32<c10::complex<float>,2>();
	forwardPSBS_kernel<<<grid, block>>>(output_a, angleAB_a, indexAB_a, split_a, atten_a);
	return output;
}

__global__ void forwardPS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,3> output_a,
	const torch::PackedTensorAccessor32<float,2> angle_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,1> atten_a)
{
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nFeatures = output_a.size(2);
	const int sample = blockIdx.x;
	const int variation = blockIdx.y;
	const int feature = threadIdx.x;
	if (feature >= nFeatures || sample >= nSamples || variation >= nVariation) return;

	const auto exp_iangle_h = atten_a[feature] * exp(1._if * angle_a[variation][feature]);
	output_a[variation][sample][feature] = exp_iangle_h * output_a[variation][sample][feature];
}

at::Tensor forwardPS_cuda(const at::Tensor input, const at::Tensor angle, const at::Tensor atten) {
	const int nVariation = input.size(0);
	const int nSamples = input.size(1);
	const int nFeatures = input.size(2);
	dim3 grid(nSamples, nVariation);
	dim3 block(nFeatures);
	at::Tensor output = input.detach().clone();
	auto output_a = output.packed_accessor32<c10::complex<float>,3>();
	const auto angle_a = angle.packed_accessor32<float,2>();
	const auto atten_a = atten.packed_accessor32<c10::complex<float>,1>();
	forwardPS_kernel<<<grid, block>>>(output_a, angle_a, atten_a);
	return output;
}

__global__ void forwardmodReLU_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,3> output_a,
	const torch::PackedTensorAccessor32<float,2> bias_a)
{
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nFeatures = output_a.size(2);
	const int sample = blockIdx.x;
	const int variation = blockIdx.y;
	const int feature = threadIdx.x;
	if (feature >= nFeatures || sample >= nSamples || variation >= nVariation) return;

	const auto incmplx = output_a[variation][sample][feature];
	const float norm = std::abs(incmplx) + modReLUeps;
	const float scale = 1.0f + bias_a[variation][feature] / norm;				
	output_a[variation][sample][feature] = (scale >= 0) ? incmplx*scale : 0;
}

at::Tensor forwardmodReLU_cuda(const at::Tensor input, const at::Tensor bias) {
	const int nVariation = input.size(0);
	const int nSamples = input.size(1);
	const int nFeatures = input.size(2);
	dim3 grid(nSamples, nVariation);
	dim3 block(nFeatures);
	at::Tensor output = input.detach().clone();
	auto output_a = output.packed_accessor32<c10::complex<float>,3>();
	const auto bias_a = bias.packed_accessor32<float,2>();
	forwardmodReLU_kernel<<<grid, block>>>(output_a, bias_a);
	return output;
}

TORCH_LIBRARY_IMPL(mzi_onn_sim_zo, CUDA, m) {
  m.impl("forwardPSBS", &forwardPSBS_cuda);
  m.impl("forwardPS", &forwardPS_cuda);
  m.impl("forwardmodReLU", &forwardmodReLU_cuda);
}

}
