#include "ChSolverBiCG.h"
using namespace chrono;


ChSolverBiCG::ChSolverBiCG() {
}

void ChSolverBiCG::Solve(real step, gpu_container &gpu_data_) {
    gpu_data = &gpu_data_;
    step_size = step;
    Setup();
    if (number_of_constraints > 0) {
        ComputeRHS();
        SolveBiCG(gpu_data->device_gam_data, rhs, 100);
        ComputeImpulses();
        gpu_data->device_vel_data += gpu_data->device_QXYZ_data;
        gpu_data->device_omg_data += gpu_data->device_QUVW_data;
    }
}

uint ChSolverBiCG::SolveBiCG(custom_vector<real> &x, const custom_vector<real> &b, const uint max_iter) {
	real rho_1, rho_2, alpha, beta;
	custom_vector<real> z, ztilde, p, ptilde, q, qtilde;
		real normb = Norm(b);
		custom_vector<real> r = b - ShurProduct(x);
		custom_vector<real> rtilde = r;

		if (normb == 0.0) {normb = 1;}

		if ((residual = Norm(r) / normb) <= tolerance) {
			return 0;
		}

		for (current_iteration = 0; current_iteration <= max_iter; current_iteration++) {
			z = (r);
			ztilde = (rtilde);
			rho_1 = Dot(z, rtilde);

			if (rho_1 == 0) {
				break;
			}

			if (current_iteration == 0) {
				p = z;
				ptilde = ztilde;
			} else {
				beta = rho_1 / rho_2;
				p = z + beta * p;
				ptilde = ztilde + beta * ptilde;
			}

			q = ShurProduct(p);
			qtilde = ShurProduct(ptilde);
			alpha = rho_1 / Dot(ptilde, q);
			x = x + alpha * p;
			r = r - alpha * q;
			rtilde = rtilde - alpha * qtilde;
			rho_2 = rho_1;
			residual = Norm(r) / normb;

			if (residual < tolerance) {break;}
		}

		return current_iteration;

}
