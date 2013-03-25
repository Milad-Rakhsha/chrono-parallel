#include "ChSolverCG.h"
using namespace chrono;


ChSolverCG::ChSolverCG() {
}

void ChSolverCG::Solve(real step, gpu_container &gpu_data_) {
    gpu_data = &gpu_data_;
    step_size = step;
    Setup();
    if (number_of_constraints > 0) {
        ComputeRHS();
        SolveCG(gpu_data->device_gam_data, rhs, 500);
        ComputeImpulses();
        gpu_data->device_vel_data += gpu_data->device_QXYZ_data;
        gpu_data->device_omg_data += gpu_data->device_QUVW_data;
    }
}

uint ChSolverCG::SolveCG(custom_vector<real> &x, const custom_vector<real> &b, const uint max_iter) {
    custom_vector<real> r(x.size()), p, Ap;
    real rsold, alpha, rsnew = 0, normb = Norm(b);
    if (normb == 0.0) {
        normb = 1;
    }
    p = r = b - ShurProduct(x);
    rsold = Dot(r, r);
    normb = 1.0 / normb;
    if (sqrt(rsold) * normb <= tolerance) {
        return 0;
    }
    for (current_iteration = 0; current_iteration < max_iter; current_iteration++) {
        Ap = ShurProduct(p);
        alpha = rsold / Dot(p, Ap);
        rsnew = 0;
#ifdef SIM_ENABLE_GPU_MODE
        SEAXPY(alpha, p, x, x);
        SEAXPY(-alpha, Ap, r, r);
        rsnew=Dot(r,r);
#else
        #pragma omp parallel for reduction(+:rsnew)
        for (int i = 0; i < x.size(); i++) {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
            rsnew += r[i] * r[i];
        }
#endif
        residual = sqrtf(rsnew) * normb;
        if (residual < tolerance) {
            break;
        }
        SEAXPY(rsnew / rsold, p, r, p); //p = r + rsnew / rsold * p;
        rsold = rsnew;
    }
    Project(x);
    return current_iteration;
}