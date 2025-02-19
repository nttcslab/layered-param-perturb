#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

using namespace c10::complex_literals;
using namespace c10_complex_math;

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace mzi_onn_sim_bp {

static constexpr double QuarterPI = 0.25*M_PI;
static constexpr double modReLUeps = 0.001; // epsilon: Magic number

template<typename T>
c10::complex<T> conj(c10::complex<T> z)
{return c10::complex<T>(z.real(),-z.imag());}

std::vector<at::Tensor>
forwardAD_PSBS_cpu(const at::Tensor input, const at::Tensor angleAB,
			   const at::Tensor tangent, const at::Tensor indexAB) {
	const int nSamples = input.size(0); // Batch size
	const int nFeatures = input.size(1);
	const int nIndexAB = indexAB.size(1);
	const int nLayers = angleAB.size(0);
	at::Tensor output = input.detach().clone();
	const auto output_a = output.accessor<c10::complex<float>,2>();
	at::Tensor grad = torch::zeros_like(input);
	const auto grad_a = grad.accessor<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.accessor<float,2>();
	const auto tangent_a = tangent.accessor<float,2>();
	const auto indexAB_a = indexAB.accessor<int32_t,3>();
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
		auto output = output_a[sample];
		auto grad = grad_a[sample];
		for (int layer = 0; layer < nLayers; ++layer) {
			const auto angle = angleAB_a[layer];
			const auto tangent = tangent_a[layer];
			const auto index = indexAB_a[layer];
			for (int param = 0; param < nIndexAB; ++param) {
				const int idx0 = index[param][0];
				const int idx1 = index[param][1];
				if (idx0 >= 0) {
					const float cos_pi_4 = cos(QuarterPI);
					const float sin_pi_4 = sin(QuarterPI);
					const auto exp_val = exp(1._if*angle[param]);
					const auto ps_out = exp_val * output[idx0];
					output[idx0] = ps_out * cos_pi_4 + 1._if * output[idx1] * sin_pi_4;
					output[idx1] = 1._if * ps_out * sin_pi_4 + output[idx1] * cos_pi_4;
					const auto eg = exp_val * grad[idx0];
					const auto eit = ps_out * tangent[param];
					grad[idx0] = (eg + 1._if * grad[idx1] + 1._if * eit) * sin_pi_4;
					grad[idx1] = (1._if * eg + grad[idx1] - eit) * cos_pi_4;
				}
			}
		}
	}
	return {output, grad};
}

at::Tensor forwardPSBS_cpu(const at::Tensor input,
			const at::Tensor angleAB, const at::Tensor indexAB,
			const at::Tensor split, const at::Tensor atten) {
	const int nSamples = input.size(0); // Batch size
	const int nFeatures = input.size(1);
	const int nLayers = angleAB.size(0);
	const int nIndexAB = indexAB.size(1);
	const auto devType = input.device().type();
	const auto devID = input.device().index();
	at::Tensor outputs = torch::empty({nSamples,nLayers,nFeatures}, torch::device({devType,devID}).dtype(c10::kComplexFloat));
	const auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.accessor<float,2>();
	const auto indexAB_a = indexAB.accessor<int32_t,3>();
	const auto split_a = split.accessor<float,2>();
	const auto atten_a = atten.accessor<c10::complex<float>,2>();
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
		const auto input_a_s = input_a[sample];
		const auto outputs_a_s = outputs_a[sample];
		for (int layer = 0; layer < nLayers; ++layer) {
			const auto input = (layer > 0) ? outputs_a_s[layer-1] : input_a_s;
			auto output = outputs_a_s[layer];
			const auto angle = angleAB_a[layer];
			const auto index = indexAB_a[layer];
			const auto split_l = split_a[layer];
			const auto atten_l = atten_a[layer];
			for (int param = 0; param < nIndexAB; ++param) {
				const int idx0 = index[param][0];
				const int idx1 = index[param][1];
				if (idx0 >= 0) {
					const float PauliP = QuarterPI + 0.5 * split_l[param];
					const float cos_pi_4 = cos(PauliP);
					const float sin_pi_4 = sin(PauliP);
					const auto ps_out = atten_l[idx0] * input[idx0] * exp(1._if*angle[param]);
					const auto in1_atten = atten_l[idx1] * input[idx1];
					output[idx0] = ps_out * cos_pi_4 + 1._if * in1_atten * sin_pi_4;
					output[idx1] = 1._if * ps_out * sin_pi_4 + in1_atten * cos_pi_4;
				}
				else { // copy
					output[~idx0] = input[~idx0];
					output[~idx1] = input[~idx1];
				}
			}
		}
	}
	return outputs;
}

std::vector<at::Tensor>
backwardPSBS_cpu(const at::Tensor grad_output,
			const at::Tensor outputs, const at::Tensor input,
			const at::Tensor angleAB, const at::Tensor indexAB,
			const at::Tensor split, const at::Tensor atten) {
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
	auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	auto grad_angleAB_a = grad_angleAB.accessor<float,3>();
	auto grad_split_a = grad_split.accessor<float,3>();
	auto grad_atten_a = grad_atten.accessor<c10::complex<float>,3>();
	const auto outputs_a = outputs.accessor<c10::complex<float>,3>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angleAB_a = angleAB.accessor<float,2>();
	const auto indexAB_a = indexAB.accessor<int32_t,3>();
	const auto split_a = split.accessor<float,2>();
	const auto atten_a = atten.accessor<c10::complex<float>,2>();
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
		const auto grad_angleABs = grad_angleAB_a[sample];
		const auto grad_split_s = grad_split_a[sample];
		const auto grad_atten_s = grad_atten_a[sample];
		auto gradin = grad_input_a[sample];
		for(int layer = nLayers-1; layer >= 0; --layer) {
			const auto input_s = (layer > 0) ? outputs_a[sample][layer-1] : input_a[sample];
			const auto angle = angleAB_a[layer];
			const auto index = indexAB_a[layer];
			const auto atten_l = atten_a[layer];
			const auto split_l = split_a[layer];
			auto grad_angle = grad_angleABs[layer];
			auto grad_split_l = grad_split_s[layer];
			auto grad_atten_l = grad_atten_s[layer];
			for (int param = 0; param < nIndexAB; ++param) {
				int idx0 = index[param][0];
				int idx1 = index[param][1];
				if (idx0 >= 0) {
					const float PauliP = QuarterPI + 0.5 * split_l[param];
					const float cos_pi_4 = cos(PauliP);
					const float sin_pi_4 = sin(PauliP);
					const auto val_exp = exp(1._if * angle[param]);
					const auto gtmp0 = cos_pi_4 * gradin[idx0] -1._if * sin_pi_4 * gradin[idx1];
					const auto gtmp1 = -1._if * sin_pi_4 * gradin[idx0] + cos_pi_4 * gradin[idx1];
					const auto tmp0conj = conj(atten_l[idx0]*input_s[idx0]*val_exp);
					const auto tmp1conj = conj(atten_l[idx1]*input_s[idx1]);
					const float tmp_real = (gradin[idx0]*tmp0conj + gradin[idx1]*tmp1conj).real();
					const float tmp_imag = (gradin[idx0]*tmp1conj + gradin[idx1]*tmp0conj).imag();
					gradin[idx0] = conj(atten_l[idx0] * val_exp) * gtmp0;
					gradin[idx1] = conj(atten_l[idx1]) * gtmp1;
					grad_angle[param] = 2.0f * (conj(input_s[idx0]) * gradin[idx0]).imag();
					grad_split_l[param] = -sin_pi_4 * tmp_real + cos_pi_4 * tmp_imag;
					grad_atten_l[idx0] = gtmp0 * conj(input_s[idx0]*val_exp);
					grad_atten_l[idx1] = gtmp1 * conj(input_s[idx1]);
				}
				else {
					idx0 = ~idx0;
					idx1 = ~idx1;
					grad_atten_l[idx0] = gradin[idx0] * conj(input_s[idx0]);  // needs to check
					grad_atten_l[idx1] = gradin[idx1] * conj(input_s[idx1]);  // needs to check
				}
			}
		}
	}
	return {grad_input, grad_angleAB, grad_split, grad_atten};
}

std::vector<at::Tensor> forwardAD_PS_cpu(at::Tensor input, // over-written to be output
						const at::Tensor angle,
						const at::Tensor tangent_input,
						const at::Tensor tangent_params) {
	const int nFeatures = input.size(1);
	const int nSamples = input.size(0);
	const auto devType = input.device().type();
	const auto devID = input.device().index();
	at::Tensor grad = torch::empty({nSamples,nFeatures},torch::device({devType,devID}).dtype(c10::kComplexFloat));
	auto grad_a = grad.accessor<c10::complex<float>,2>();
	auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angle_a = angle.accessor<float,1>();
	const auto tangent_input_a = tangent_input.accessor<c10::complex<float>,2>();
	const auto tangent_params_a = tangent_params.accessor<float,1>();
	#pragma omp parallel for
	for (int64_t feature = 0; feature < nFeatures; ++feature) {
		const auto exp_iangle = exp(1._if * angle_a[feature]);
		for (int64_t sample = 0; sample < nSamples; ++sample) {
			input_a[sample][feature] = exp_iangle * input_a[sample][feature];
			grad_a[sample][feature] = exp_iangle * tangent_input_a[sample][feature] + 1._if * input_a[sample][feature] * tangent_params_a[feature];
		}
	}
	return {input, grad};
}

at::Tensor forwardPS_cpu(const at::Tensor input, const at::Tensor angle) {
	const auto devType = input.device().type();
	at::Tensor output = input.detach().clone();
	auto output_a = output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angle_a = angle.accessor<float,1>();
	const int nFeatures = input_a.size(1);
	const int nSamples = input_a.size(0);
	#pragma omp parallel for
	for (int64_t feature = 0; feature < nFeatures; ++feature) {
		const auto exp_iangle_h = exp(1._if * angle_a[feature]);
		for (int64_t sample = 0; sample < nSamples; ++sample) {
			output_a[sample][feature] = exp_iangle_h * input_a[sample][feature];
		}
	}
	return output;
}

std::vector<at::Tensor>
backwardPS_cpu(const at::Tensor grad_output,
			const at::Tensor input, const at::Tensor angle) {
	const auto devType = input.device().type();
	at::Tensor grad_input = grad_output.detach().clone();
	at::Tensor grad_angle = torch::real(grad_output.detach()).clone();
	auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	auto grad_angle_a = grad_angle.accessor<float,2>();
	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto angle_a = angle.accessor<float,1>();
	const int nFeatures = grad_output_a.size(1);
	const int nSamples = grad_output_a.size(0);
	#pragma omp parallel for
	for (int64_t feature = 0; feature < nFeatures; ++feature) {
		const auto exp_miangle_h = exp(-1._if * angle_a[feature]);
		for (int64_t sample = 0; sample < nSamples; ++sample) {
			grad_input_a[sample][feature] = exp_miangle_h * grad_output_a[sample][feature];
			grad_angle_a[sample][feature] = 2.0 * (conj(input_a[sample][feature]) * grad_input_a[sample][feature]).imag();
		}
	}
	return {grad_input, grad_angle};
}

at::Tensor forwardmodReLU_cpu(const at::Tensor input, const at::Tensor bias) {
	const int nFeatures = input.size(1);
	const int nSamples = input.size(0);
	at::Tensor output = input.detach().clone();
	const auto output_a = output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto bias_a = bias.accessor<float,1>();
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
		const auto input_a_s = input_a[sample];
		auto output_a_s = output_a[sample];
		for (int feature = 0; feature < nFeatures; ++feature) {
			const auto incmplx = input_a_s[feature];
			const float norm = std::abs(incmplx) + modReLUeps;
			const float scale = 1.0f + bias_a[feature] / norm;				
			output_a_s[feature] = (scale >= 0) ? incmplx*scale : 0;
		}
	}
	return output;
}

std::vector<at::Tensor>
backwardmodReLU_cpu(const at::Tensor grad_output,
			const at::Tensor input, const at::Tensor bias) {
	const int nSamples = grad_output.size(0);
	const int nFeatures = grad_output.size(1);
	at::Tensor grad_input = grad_output.detach().clone();
	at::Tensor grad_bias = torch::real(grad_output.detach()).clone();
	const auto grad_input_a = grad_input.accessor<c10::complex<float>,2>();
	const auto grad_bias_a = grad_bias.accessor<float,2>();
	const auto grad_output_a = grad_output.accessor<c10::complex<float>,2>();
	const auto input_a = input.accessor<c10::complex<float>,2>();
	const auto bias_a = bias.accessor<float,1>();

	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
		const auto input_a_s = input_a[sample];
		const auto grad_output_a_s = grad_output_a[sample];
		auto grad_input_a_s = grad_input_a[sample];
		auto grad_bias_a_s = grad_bias_a[sample];
		for (int feature = 0; feature < nFeatures; ++feature) {
			const auto incmplx = input_a_s[feature];
			const float inv_norm = 1.0f / (std::abs(incmplx) + modReLUeps);
			const float scale = 1.0f + bias_a[feature] * inv_norm;				
			if (scale >= 0) {
				const auto gout = grad_output_a_s[feature];
				grad_input_a_s[feature] = scale * gout;
				grad_bias_a_s[feature] = 2.0f * inv_norm * (gout*conj(incmplx)).real();
			} else {
				grad_input_a_s[feature] = 0.0;
				grad_bias_a_s[feature] = 0.0;
			}
		}
	}
	return {grad_input, grad_bias};
}

TORCH_LIBRARY(mzi_onn_sim_bp, m) {
  m.def("forwardAD_PSBS(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor[]");
  m.def("forwardPSBS(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor");
  m.def("backwardPSBS(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e, Tensor f, Tensor g) -> Tensor[]");
  m.def("forwardAD_PS_(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor[]");
  m.def("forwardPS(Tensor a, Tensor b) -> Tensor");
  m.def("backwardPS(Tensor a, Tensor b, Tensor c) -> Tensor[]");
  m.def("forwardmodReLU(Tensor a, Tensor b) -> Tensor");
  m.def("backwardmodReLU(Tensor a, Tensor b, Tensor c) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mzi_onn_sim_bp, CPU, m) {
  m.impl("forwardAD_PSBS", &forwardAD_PSBS_cpu);
  m.impl("forwardPSBS", &forwardPSBS_cpu);
  m.impl("backwardPSBS", &backwardPSBS_cpu);
  m.impl("forwardAD_PS_", &forwardAD_PS_cpu);
  m.impl("forwardPS", &forwardPS_cpu);
  m.impl("backwardPS", &backwardPS_cpu);
  m.impl("forwardmodReLU", &forwardmodReLU_cpu);
  m.impl("backwardmodReLU", &backwardmodReLU_cpu);
}

}
