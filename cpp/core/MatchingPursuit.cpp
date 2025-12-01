#include "MatchingPursuit.h"
#include "MathUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace BVC {
    namespace MP {
        // Solves A * x = b where A is G_sub
        // Uses Regularized Cholesky Decomposition for stability
        // Returns true if successful, false if singular
        bool solve_symmetric_robust(int n, std::vector<std::vector<double>>& A, std::vector<double>& b, std::vector<double>& x) {
            // 1. Regularize Diagonal (Ridge Regression)
            for(int i=0; i<n; ++i) A[i][i] += 1e-2; 

            // 2. Cholesky Decomposition L * L^T = A
            std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];

                    if (i == j) {
                        double val = A[i][i] - sum;
                        if (val <= 0.0) return false; // Not Positive Definite
                        L[i][j] = std::sqrt(val);
                    } else {
                        L[i][j] = (1.0 / L[j][j] * (A[i][j] - sum));
                    }
                }
            }

            // 3. Solve Ly = b (Forward sub)
            std::vector<double> y(n);
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < i; k++) sum += L[i][k] * y[k];
                y[i] = (b[i] - sum) / L[i][i];
            }

            // 4. Solve L^T x = y (Backward sub)
            for (int i = n - 1; i >= 0; i--) {
                double sum = 0;
                for (int k = i + 1; k < n; k++) sum += L[k][i] * x[k];
                x[i] = (y[i] - sum) / L[i][i];
            }
            
            return true;
        }

        void least_squares_refine(const std::shared_ptr<DictionaryEntry>& dict,
                                  const std::vector<float>& residual_orig,
                                  std::vector<Atom>& atoms) {
            if (atoms.empty()) return;
            int k = static_cast<int>(atoms.size());
            int N = dict->length;
            
            std::vector<double> b(k);
            std::vector<std::vector<double>> G_sub(k, std::vector<double>(k));
            
            for(int i=0; i<k; ++i) {
                const float* atom_i = &dict->D_flat[atoms[i].idx * N];
                b[i] = MathUtils::dot_product(atom_i, residual_orig.data(), N);
                
                for(int j=i; j<k; ++j) { // Fill upper triangle + diagonal
                    double val;
                    if (dict->has_gram) {
                        val = dict->G_flat[atoms[i].idx * dict->n_atoms + atoms[j].idx];
                    } else {
                        // Compute on-the-fly if Gram missing
                        const float* atom_j = &dict->D_flat[atoms[j].idx * N];
                        val = MathUtils::dot_product(atom_i, atom_j, N);
                    }
                    G_sub[i][j] = val;
                    G_sub[j][i] = val;
                }
            }

            // Backup original values in case solver fails
            std::vector<float> original_values;
            for(auto& a : atoms) original_values.push_back(a.value);

            std::vector<double> x(k);
            if (solve_symmetric_robust(k, G_sub, b, x)) {
                for(int i=0; i<k; ++i) atoms[i].value = static_cast<float>(x[i]);
            } else {
                // Solver failed (singular), keep original MP coefficients
            }
        }

        std::vector<Atom> matching_pursuit(const std::vector<float>& residual,
                                           const std::shared_ptr<DictionaryEntry>& dict,
                                           int max_atoms, float threshold) {
            std::vector<Atom> selected;
            if (dict->n_atoms == 0) return selected;

            int N = dict->length;
            size_t n_dict = dict->n_atoms;

            // Initial correlations: D.T * r
            std::vector<float> correlations(n_dict);

            // Calculate correlations in parallel
            #pragma omp parallel for
            for(int i=0; i < (int)n_dict; ++i) {
                correlations[i] = MathUtils::dot_product(&dict->D_flat[i*N], residual.data(), N);
            }

            float current_energy = MathUtils::dot_product(residual.data(), residual.data(), N);

            for(int iter=0; iter<max_atoms; ++iter) {
                int best_idx = -1;
                float max_corr = -1.0f;

                // Find max correlation (Pure Greedy MP, like Python)
                for(size_t i=0; i<n_dict; ++i) {
                    float val = std::abs(correlations[i]);
                    if(val > max_corr) {
                        max_corr = val;
                        best_idx = static_cast<int>(i);
                    }
                }

                if(best_idx == -1 || max_corr < 1e-6f) break;

                float coeff = correlations[best_idx];
                selected.push_back({(uint16_t)best_idx, coeff});

                current_energy -= coeff * coeff;
                if(current_energy < threshold) break;

                // Update correlations
                if (dict->has_gram) {
                    const float* G_col = &dict->G_flat[best_idx * n_dict];

                    // Unrolled loop for Gram update
                    int i = 0;
                    int n_int = (int)n_dict;
                    for(; i + 4 <= n_int; i += 4) {
                        correlations[i]   -= coeff * G_col[i];
                        correlations[i+1] -= coeff * G_col[i+1];
                        correlations[i+2] -= coeff * G_col[i+2];
                        correlations[i+3] -= coeff * G_col[i+3];
                    }
                    for(; i < n_int; ++i) {
                        correlations[i] -= coeff * G_col[i];
                    }
                } else {
                    // Direct Update: corr[i] -= coeff * (d_best . d_i)
                    const float* d_best = &dict->D_flat[best_idx * N];

                    // Use OpenMP parallelization but only for reasonably large dictionaries
                    if (n_dict > 1024) {
                        #pragma omp parallel for
                        for(int i=0; i < (int)n_dict; ++i) {
                            float dot = MathUtils::dot_product(d_best, &dict->D_flat[i*N], N);
                            correlations[i] -= coeff * dot;
                        }
                    } else {
                        // For smaller dictionaries, do it directly without threading overhead
                        for(size_t i=0; i<n_dict; ++i) {
                            float dot = MathUtils::dot_product(d_best, &dict->D_flat[i*N], N);
                            correlations[i] -= coeff * dot;
                        }
                    }
                }
            }

            // Refinement (Standard in BVC)
            if (!selected.empty()) {
                least_squares_refine(dict, residual, selected);
            }

            return selected;
        }
    }
}