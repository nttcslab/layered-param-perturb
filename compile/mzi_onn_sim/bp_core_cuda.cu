#include <torch/extension.h>

using namespace c10::complex_literals;
using namespace c10_complex_math;

namespace mzi_onn_sim_bp {

static constexpr double QuarterPI = 0.25*M_PI;
static constexpr double modReLUeps = 0.001; // epsilon: Magic number

__device__ c10::complex<float> conj(c10::complex<float> input) {
	return c10::complex<float>(input.real(), -input.imag());
}

__global__ void forwardAD_PSBS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> output_a,
	torch::PackedTensorAccessor32<c10::complex<float>,2> grad_a,
	const torch::PackedTensorAccessor32<float,2> angleAB_a,
	const torch::PackedTensorAccessor32<float,2> tangent_a,
	const torch::PackedTensorAccessor32<int32_t,3> indexAB_a)
{
	const int nSamples = output_a.size(0); // Batch size
	const int nIndexAB = indexAB_a.size(1);
	const int param = threadIdx.x;
	const int sample = blockIdx.x;
	if (param >= nIndexAB || sample >= nSamples) return;

	const int nLayers = angleAB_a.size(0);
	auto output = output_a[sample];
	auto grad = grad_a[sample];
	const float cos_pi_4 = cos(QuarterPI);
	const float sin_pi_4 = sin(QuarterPI);
	for (int layer = 0; layer < nLayers; ++layer){
		const auto angle = angleAB_a[layer];
		const auto tangent = tangent_a[layer];
		const auto index = indexAB_a[layer][param];
		const int idx0 = index[0];
		const int idx1 = index[1];
		if (idx0 >= 0) {
			const auto exp_val = exp(1._if*angle[param]);
			const auto ps_out = output[idx0] * exp_val;
			output[idx0] = ps_out * cos_pi_4 + 1._if * output[idx1] * sin_pi_4;
			output[idx1] = 1._if * ps_out * sin_pi_4 + output[idx1] * cos_pi_4;
			const auto eg = exp_val * grad[idx0];
			const auto eit = ps_out * tangent[param];
			grad[idx0] = (eg + 1._if * grad[idx1] + 1._if * eit) * sin_pi_4;
			grad[idx1] = (1._if * eg + grad[idx1] - eit) * cos_pi_4;
		}
		__syncthreads();
	}
}

std::vector<at::Tensor>
forwardAD_PSBS_cuda(const at::Tensor input, const at::Tensor angleAB,
			   const at::Tensor tangent, const at::Tensor indexAB) {
	const int nSamples = input.size(0); // Batch size
	const int nFeatures = input.size(1);
	const int nIndexAB = indexAB.size(1);
	at::Tensor output = input.detach().clone();
	auto output_a = output.packed_accessor32<c10::complex<float>,2>();
	at::Tensor grad = torch::zeros_like(input);
	auto grad_a = grad.packed_accessor32<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.packed_accessor32<float,2>();
	const auto tangent_a = tangent.packed_accessor32<float,2>();
	const auto indexAB_a = indexAB.packed_accessor32<int32_t,3>();
	forwardAD_PSBS_kernel<<<nSamples,nIndexAB>>>(output_a, grad_a, angleAB_a, tangent_a, indexAB_a);
	return {output, grad};
}

__global__ void forwardPSBS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,3> outputs_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,2> angleAB_a,
	const torch::PackedTensorAccessor32<int32_t,3> indexAB_a,
	const torch::PackedTensorAccessor32<float,2> split_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> atten_a)
{
	const int nSamples = input_a.size(0); // Batch size
	const int nIndexAB = indexAB_a.size(1);
	const int param = threadIdx.x;
	const int sample = blockIdx.x;
	if (param >= nIndexAB || sample >= nSamples) return;

	const int nLayers = angleAB_a.size(0);
	const auto output_layers_s = outputs_a[sample];
	auto input = input_a[sample];
	for (int layer = 0; layer < nLayers; ++layer){
		auto output = output_layers_s[layer];
		const auto angle = angleAB_a[layer];
		const auto split = split_a[layer];
		const auto atten = atten_a[layer];
		const auto index = indexAB_a[layer][param];
		const int idx0 = index[0];
		const int idx1 = index[1];
		if (idx0 >= 0) {
			const float PauliP = QuarterPI + 0.5 * split[param];
			const float cos_pi_4 = cos(PauliP);
			const float sin_pi_4 = sin(PauliP);
			const auto ps_out = atten[idx0] * input[idx0] * exp(1._if*angle[param]);
			const auto in1_atten = atten[idx1] * input[idx1];
			output[idx0] = ps_out * cos_pi_4 + 1._if * in1_atten * sin_pi_4;
			output[idx1] = 1._if * ps_out * sin_pi_4 + in1_atten * cos_pi_4;
		}
		else { // copy
			output[~idx0] = input[~idx0];
			output[~idx1] = input[~idx1];
		}
		__syncthreads();
		input = output;
	}
}

at::Tensor forwardPSBS_cuda(const at::Tensor input, const at::Tensor angleAB,
		const at::Tensor indexAB, const at::Tensor split, const at::Tensor atten) {
	const int nSamples = input.size(0); // Batch size
	const int nFeatures = input.size(1);
	const int nLayers = angleAB.size(0);
	const auto devType = input.device().type();
	const auto devID = input.device().index();
	const int nIndexAB = indexAB.size(1);
	at::Tensor outputs = torch::empty({nSamples,nLayers,nFeatures}, torch::device({devType,devID}).dtype(c10::kComplexFloat));
	auto outputs_a = outputs.packed_accessor32<c10::complex<float>,3>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.packed_accessor32<float,2>();
	const auto indexAB_a = indexAB.packed_accessor32<int32_t,3>();
	const auto split_a = split.packed_accessor32<float,2>();
	const auto atten_a = atten.packed_accessor32<c10::complex<float>,2>();
	forwardPSBS_kernel<<<nSamples,nIndexAB>>>(outputs_a, input_a, angleAB_a, indexAB_a, split_a, atten_a);
	return outputs;
}

__global__ void backwardPSBS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> grad_input_a,
	torch::PackedTensorAccessor32<float,3> grad_angleAB_a,
	torch::PackedTensorAccessor32<float,3> grad_split_a,
	torch::PackedTensorAccessor32<c10::complex<float>,3> grad_atten_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,3> outputs_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,2> angleAB_a,
	const torch::PackedTensorAccessor32<int32_t,3> indexAB_a,
	const torch::PackedTensorAccessor32<float,2> split_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> atten_a)
{
	const int nSamples = grad_input_a.size(0); // Batch size
	const int nIndexAB = indexAB_a.size(1);
	const int param = threadIdx.x;
	const int sample = blockIdx.x;
	if (param >= nIndexAB || sample >= nSamples) return;

	const int nLayers = angleAB_a.size(0);
	auto gradin = grad_input_a[sample];
	const auto grad_angleABs = grad_angleAB_a[sample];
	const auto grad_split_s = grad_split_a[sample];
	const auto grad_atten_s = grad_atten_a[sample];
	for (int layer = nLayers-1; layer >= 0; --layer){
		const auto input_s = (layer > 0) ? outputs_a[sample][layer-1] : input_a[sample];
		const auto angle = angleAB_a[layer];
		const auto index = indexAB_a[layer][param];
		const auto atten = atten_a[layer];
		const auto split = split_a[layer];
		auto grad_angle = grad_angleABs[layer];
		auto grad_split = grad_split_s[layer];
		auto grad_atten = grad_atten_s[layer];
		int idx0 = index[0];
		int idx1 = index[1];
		if (idx0 >= 0) {
			const float PauliP = QuarterPI + 0.5 * split[param];
			const float cos_pi_4 = cos(PauliP);
			const float sin_pi_4 = sin(PauliP);
			const auto val_exp = exp(1._if*(angle[param]));
			const auto gtmp0 = cos_pi_4 * gradin[idx0] -1._if * sin_pi_4 * gradin[idx1];
			const auto gtmp1 = -1._if * sin_pi_4 * gradin[idx0] + cos_pi_4 * gradin[idx1];
			const auto tmp0conj = conj(atten[idx0]*input_s[idx0]*val_exp);
			const auto tmp1conj = conj(atten[idx1]*input_s[idx1]);
			const float tmp_real = (gradin[idx0]*tmp0conj + gradin[idx1]*tmp1conj).real();
			const float tmp_imag = (gradin[idx0]*tmp1conj + gradin[idx1]*tmp0conj).imag();
			gradin[idx0] = conj(atten[idx0] * val_exp) * gtmp0;
			gradin[idx1] = conj(atten[idx1]) * gtmp1;
			grad_angle[param] = 2.0f * (conj(input_s[idx0])*gradin[idx0]).imag();
			grad_split[param] = -sin_pi_4 * tmp_real + cos_pi_4 * tmp_imag;
			grad_atten[idx0] = gtmp0 * conj(input_s[idx0]*val_exp);
			grad_atten[idx1] = gtmp1 * conj(input_s[idx1]);
		}
		else {
			idx0 = ~idx0;
			idx1 = ~idx1;
			grad_atten[idx0] = gradin[idx0] * conj(input_s[idx0]);  // needs to check
			grad_atten[idx1] = gradin[idx1] * conj(input_s[idx1]);  // needs to check
		}
		__syncthreads();
	}
}

std::vector<at::Tensor>
backwardPSBS_cuda(const at::Tensor grad_output, const at::Tensor outputs, const at::Tensor input,
		const at::Tensor angleAB, const at::Tensor indexAB, const at::Tensor split, const at::Tensor atten) {
	const int nLayers = angleAB.size(0);
	const int nAnglesAB = angleAB.size(1);
	const int nSamples = grad_output.size(0); // Batch size
	const int nFeatures = atten.size(1);
	const int nIndexAB = indexAB.size(1);
	const auto devType = input.device().type();
	const auto devID = input.device().index();
	at::Tensor grad_input = grad_output.detach().clone();
	at::Tensor grad_angleAB = torch::zeros({nSamples,nLayers,nAnglesAB}, torch::device({devType,devID}).dtype(at::kFloat));
	at::Tensor grad_split = torch::zeros({nSamples,nLayers,nAnglesAB}, torch::device({devType,devID}).dtype(at::kFloat));
	at::Tensor grad_atten = torch::zeros({nSamples,nLayers,nFeatures}, torch::device({devType,devID}).dtype(c10::kComplexFloat));
	auto grad_input_a = grad_input.packed_accessor32<c10::complex<float>,2>();
	auto grad_angleAB_a = grad_angleAB.packed_accessor32<float,3>();
	auto grad_split_a = grad_split.packed_accessor32<float,3>();
	auto grad_atten_a = grad_atten.packed_accessor32<c10::complex<float>,3>();
	const auto outputs_a = outputs.packed_accessor32<c10::complex<float>,3>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.packed_accessor32<float,2>();
	const auto indexAB_a = indexAB.packed_accessor32<int32_t,3>();
	const auto split_a = split.packed_accessor32<float,2>();
	const auto atten_a = atten.packed_accessor32<c10::complex<float>,2>();
	backwardPSBS_kernel<<<nSamples,nIndexAB>>>(grad_input_a, grad_angleAB_a, grad_split_a, grad_atten_a,
				outputs_a, input_a, angleAB_a, indexAB_a, split_a, atten_a);
	return {grad_input, grad_angleAB, grad_split, grad_atten};
}

__global__ void forwardAD_PS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> grad_a,
	torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,1> angle_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> tangent_input_a,
	const torch::PackedTensorAccessor32<float,1> tangent_params_a)
{
	const int nFeatures = input_a.size(1);
	const int nSamples = input_a.size(0);
	const int feature = threadIdx.x;
	const int sample = blockIdx.x;
	if (feature >= nFeatures || sample >= nSamples) return;

	const auto exp_iangle = exp(1._if * angle_a[feature]);
	input_a[sample][feature] = exp_iangle * input_a[sample][feature];
	grad_a[sample][feature] = exp_iangle * tangent_input_a[sample][feature] + 1._if * input_a[sample][feature] * tangent_params_a[feature];
}

std::vector<at::Tensor> forwardAD_PS_cuda(at::Tensor input, // over-written to be output
		const at::Tensor angle, const at::Tensor tangent_input, const at::Tensor tangent_params) {
	const int nSamples = input.size(0);
	const int nFeatures = input.size(1);
	at::Tensor grad = torch::empty_like(input);
	auto grad_a = grad.packed_accessor32<c10::complex<float>,2>();
	auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto angle_a = angle.packed_accessor32<float,1>();
	const auto tangent_input_a = tangent_input.packed_accessor32<c10::complex<float>,2>();
	const auto tangent_params_a = tangent_params.packed_accessor32<float,1>();
	forwardAD_PS_kernel<<<nSamples,nFeatures>>>(grad_a, input_a, angle_a, tangent_input_a, tangent_params_a);
	return {input, grad};
}

__global__ void forwardPS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> output_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,1> angle_a)
{
	const int nFeatures = input_a.size(1);
	const int nSamples = input_a.size(0);
	const int feature = threadIdx.x;
	const int sample = blockIdx.x;
	if (feature >= nFeatures || sample >= nSamples) return;

	const auto exp_iangle = exp(1._if * angle_a[feature]);
	output_a[sample][feature] = exp_iangle * input_a[sample][feature];
}

at::Tensor forwardPS_cuda(const at::Tensor input, const at::Tensor angle) {
	const int nFeatures = input.size(1);
	const int nSamples = input.size(0);
	at::Tensor output = input.detach().clone();
	auto output_a = output.packed_accessor32<c10::complex<float>,2>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto angle_a = angle.packed_accessor32<float,1>();
	forwardPS_kernel<<<nSamples,nFeatures>>>(output_a, input_a, angle_a);
	return output;
}

__global__ void backwardPS_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> grad_input_a,
	torch::PackedTensorAccessor32<float,2> grad_angle_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> grad_output_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,1> angle_a)
{
	const int nFeatures = grad_output_a.size(1);
	const int nSamples = grad_output_a.size(0);
	const int feature = threadIdx.x;
	const int sample = blockIdx.x;
	if (feature >= nFeatures || sample >= nSamples) return;

	const auto exp_miangle = exp(-1._if * angle_a[feature]);
	grad_input_a[sample][feature] = exp_miangle * grad_output_a[sample][feature];
	grad_angle_a[sample][feature] = 2.0 * (conj(input_a[sample][feature]) * grad_input_a[sample][feature]).imag();
}

std::vector<at::Tensor>
backwardPS_cuda(const at::Tensor grad_output, const at::Tensor input, const at::Tensor angle) {
	const int nFeatures = grad_output.size(1);
	const int nSamples = grad_output.size(0);
	at::Tensor grad_input = grad_output.detach().clone();
	auto grad_input_a = grad_input.packed_accessor32<c10::complex<float>,2>();
	at::Tensor grad_angle = torch::real(grad_output.detach()).clone();
	auto grad_angle_a = grad_angle.packed_accessor32<float,2>();
	const auto grad_output_a = grad_output.packed_accessor32<c10::complex<float>,2>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto angle_a = angle.packed_accessor32<float,1>();
	backwardPS_kernel<<<nSamples,nFeatures>>>(grad_input_a, grad_angle_a, grad_output_a, input_a, angle_a);
	return {grad_input, grad_angle};
}

__global__ void forwardmodReLU_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> output_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,1> bias_a)
{
	const int nFeatures = input_a.size(1);
	const int nSamples = input_a.size(0);
	const int feature = threadIdx.x;
	const int sample = blockIdx.x;
	if (feature >= nFeatures || sample >= nSamples) return;

	const auto incmplx = input_a[sample][feature];
	const float norm = std::abs(incmplx) + modReLUeps;
	const float scale = 1.0f + bias_a[feature] / norm;				
	output_a[sample][feature] = (scale >= 0) ? incmplx*scale : 0;
}

at::Tensor forwardmodReLU_cuda(const at::Tensor input, const at::Tensor bias) {
	const int nFeatures = input.size(1);
	const int nSamples = input.size(0);
	at::Tensor output = input.detach().clone();
	auto output_a = output.packed_accessor32<c10::complex<float>,2>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto bias_a = bias.packed_accessor32<float,1>();
	forwardmodReLU_kernel<<<nSamples,nFeatures>>>(output_a, input_a, bias_a);
	return output;
}

__global__ void backwardmodReLU_kernel(
	torch::PackedTensorAccessor32<c10::complex<float>,2> grad_input_a,
	torch::PackedTensorAccessor32<float,2> grad_bias_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> grad_output_a,
	const torch::PackedTensorAccessor32<c10::complex<float>,2> input_a,
	const torch::PackedTensorAccessor32<float,1> bias_a)
{
	const int nFeatures = grad_output_a.size(1);
	const int nSamples = grad_output_a.size(0);
	const int feature = threadIdx.x;
	const int sample = blockIdx.x;
	if (feature >= nFeatures || sample >= nSamples) return;

	const auto incmplx = input_a[sample][feature];
	const float inv_norm = 1.0f / (std::abs(incmplx) + modReLUeps);
	const float scale = 1.0f + bias_a[feature] * inv_norm;				
	if (scale >= 0) {
		const auto gout = grad_output_a[sample][feature];
		grad_input_a[sample][feature] = scale * gout;
		grad_bias_a[sample][feature] = 2.0f * inv_norm * (gout*conj(incmplx)).real();
	} else {
		grad_input_a[sample][feature] = 0.0;
		grad_bias_a[sample][feature] = 0.0;
	}
}

std::vector<at::Tensor>
backwardmodReLU_cuda(const at::Tensor grad_output, const at::Tensor input, const at::Tensor bias) {
	const int nSamples = grad_output.size(0);
	const int nFeatures = grad_output.size(1);
	at::Tensor grad_input = grad_output.detach().clone();
	auto grad_input_a = grad_input.packed_accessor32<c10::complex<float>,2>();
	at::Tensor grad_bias = torch::real(grad_output.detach()).clone();
	auto grad_bias_a = grad_bias.packed_accessor32<float,2>();
	const auto grad_output_a = grad_output.packed_accessor32<c10::complex<float>,2>();
	const auto input_a = input.packed_accessor32<c10::complex<float>,2>();
	const auto bias_a = bias.packed_accessor32<float,1>();
	backwardmodReLU_kernel<<<nSamples,nFeatures>>>(grad_input_a, grad_bias_a, grad_output_a, input_a, bias_a);
	return {grad_input, grad_bias};
}

TORCH_LIBRARY_IMPL(mzi_onn_sim_bp, CUDA, m) {
  m.impl("forwardAD_PSBS", &forwardAD_PSBS_cuda);
  m.impl("forwardPSBS", &forwardPSBS_cuda);
  m.impl("backwardPSBS", &backwardPSBS_cuda);
  m.impl("forwardAD_PS_", &forwardAD_PS_cuda);
  m.impl("forwardPS", &forwardPS_cuda);
  m.impl("backwardPS", &backwardPS_cuda);
  m.impl("forwardmodReLU", &forwardmodReLU_cuda);
  m.impl("backwardmodReLU", &backwardmodReLU_cuda);
}

}
