#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

using namespace c10::complex_literals;
using namespace c10_complex_math;

namespace mzi_onn_sim_zo {

static constexpr double QuarterPI = 0.25*M_PI;
static constexpr double modReLUeps = 0.001; // epsilon: Magic number

template<typename T>
c10::complex<T> conj(c10::complex<T> z)
{return c10::complex<T>(z.real(),-z.imag());}

at::Tensor forwardPSBS_cpu(const at::Tensor input, const at::Tensor angleAB,
		const at::Tensor indexAB, const at::Tensor split, const at::Tensor atten) {
	at::Tensor output = input.detach().clone();
	auto output_a = output.accessor<c10::complex<float>,3>();
	const auto angleAB_a = angleAB.accessor<float,3>();
	const auto indexAB_a = indexAB.accessor<int32_t,3>();
	const auto split_a = split.accessor<float,2>();
	const auto atten_a = atten.accessor<c10::complex<float>,2>();
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nLayers = indexAB_a.size(0);
	const int nIndexAB = indexAB_a.size(1);
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
	for (int64_t variation = 0; variation < nVariation; ++variation) {
		auto output_vs = output_a[variation][sample];
		for (int layer = 0; layer < nLayers; ++layer) {
			const auto angle = angleAB_a[variation][layer];
			const auto index = indexAB_a[layer];
			const auto split_l = split_a[layer];
			const auto atten_l = atten_a[layer];
			for (int param = 0; param < nIndexAB; ++param) {
				const int idx0 = index[param][0];
				if (idx0 >= 0) {
					const int idx1 = index[param][1];
					const float PauliP = QuarterPI + 0.5 * split_l[param];
					const float cos_pi_4 = cos(PauliP);
					const float sin_pi_4 = sin(PauliP);
					const auto ps_out = atten_l[idx0] * output_vs[idx0] * exp(1._if*angle[param]);
					const auto in1_atten = atten_l[idx1] * output_vs[idx1];
					output_vs[idx0] = ps_out * cos_pi_4 + 1._if * in1_atten * sin_pi_4;
					output_vs[idx1] = 1._if * ps_out * sin_pi_4 + in1_atten * cos_pi_4;	
				}
			}
		}
	}
	}
	return output;
}

at::Tensor forwardPS_cpu(const at::Tensor input, const at::Tensor angle, const at::Tensor atten) {
	at::Tensor output = input.detach().clone();
	auto output_a = output.accessor<c10::complex<float>,3>();
	const auto angle_a = angle.accessor<float,2>();
	const auto atten_a = atten.accessor<c10::complex<float>,1>();
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nFeatures = output_a.size(2);
	#pragma omp parallel for
	for (int64_t feature = 0; feature < nFeatures; ++feature) {
	for (int64_t variation = 0; variation < nVariation; ++variation) {
		const auto exp_iangle_h = atten_a[feature] * exp(1._if * angle_a[variation][feature]);
		for (int64_t sample = 0; sample < nSamples; ++sample) {
			output_a[variation][sample][feature] = exp_iangle_h * output_a[variation][sample][feature];
		}
	}
	}
	return output;
}

at::Tensor forwardmodReLU_cpu(const at::Tensor input, const at::Tensor bias) {
	at::Tensor output = input.detach().clone();
	auto output_a = output.accessor<c10::complex<float>,3>();
	const auto bias_a = bias.accessor<float,2>();
	const int nVariation = output_a.size(0);
	const int nSamples = output_a.size(1);
	const int nFeatures = output_a.size(2);
	#pragma omp parallel for
	for (int64_t sample = 0; sample < nSamples; ++sample) {
	for (int64_t variation = 0; variation < nVariation; ++variation) {
		auto output_vs = output_a[variation][sample];
		for (int feature = 0; feature < nFeatures; ++feature) {
			const auto incmplx = output_vs[feature];
			const float norm = std::abs(incmplx) + modReLUeps;
			const float scale = 1.0f + bias_a[variation][feature] / norm;				
			output_vs[feature] = (scale >= 0) ? incmplx*scale : 0;
		}
	}
	}
	return output;
}

TORCH_LIBRARY(mzi_onn_sim_zo, m) {
  m.def("forwardPSBS(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor");
  m.def("forwardPS(Tensor a, Tensor b, Tensor c) -> Tensor");
  m.def("forwardmodReLU(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(mzi_onn_sim_zo, CPU, m) {
  m.impl("forwardPSBS", &forwardPSBS_cpu);
  m.impl("forwardPS", &forwardPS_cpu);
  m.impl("forwardmodReLU", &forwardmodReLU_cpu);
}

}
