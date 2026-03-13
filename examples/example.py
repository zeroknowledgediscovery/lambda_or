from lambda_or import lambda_or
import numpy as np
from math import erfc, sqrt, log10, pi

def neglog10_p_from_z(z_abs):
    if z_abs <= 7.0:
        p_two = max(min(erfc(z_abs / sqrt(2.0)), 1.0), 1e-323)
        return -log10(p_two)
    return (z_abs * z_abs) / (2.0 * np.log(10.0)) + np.log10(np.sqrt(2.0 * np.pi) * z_abs) - np.log10(2.0)

# rows = X in {1,0}
# cols = Y~ in {1,0}
tilde_counts = np.array([
    [20,  30],
    [10, 240]
], dtype=float)

# naive OR
a, b = tilde_counts[0]
c, d = tilde_counts[1]

naive_or = (a * d) / (b * c)
naive_log_or = np.log(naive_or)
naive_se = np.sqrt(1/a + 1/b + 1/c + 1/d)
naive_z = naive_log_or / naive_se
naive_neglog10_p = neglog10_p_from_z(abs(naive_z))

# lambda-OR
res = lambda_or(
    tilde_counts=tilde_counts,
    p_sel=0.92,
    q_sel=0.88,
    n_val=1000
)

print("Naive OR:", naive_or)
print("Naive log OR:", naive_log_or)
print("Naive z:", naive_z)
print("Naive -log10(p):", naive_neglog10_p)

print("\nLambda-OR:", np.exp(res.log_or))
print("Log Lambda-OR:", res.log_or)
print("SE:", res.se)
print("z:", res.z)
print("Lambda -log10(p):", res.neglog10_p)
print("lambda used:", res.lambda_used)

print("\nCorrected contingency table:")
print(res.counts)
