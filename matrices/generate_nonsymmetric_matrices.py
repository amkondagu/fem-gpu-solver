import argparse
from pathlib import Path

import numpy as np
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, diags, block_diag


def write_matrix(A, name, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    A = A.tocsr()
    mtx_path = outdir / f"{name}.mtx"

    # Force explicit/general Matrix Market storage.
    mmwrite(str(mtx_path), A, symmetry="general")

    print(f"{name}: shape={A.shape}, nnz={A.nnz}")
    return mtx_path


def bandtoep(u, n):
    # Sparse equivalent of Purdue bandtoep.m.
    u = np.asarray(u, dtype=float).reshape(-1)
    k = (len(u) + 1) // 2

    offsets = []
    diagonals = []

    for idx, value in enumerate(u):
        offset = idx - (k - 1)
        length = n - abs(offset)

        if length <= 0:
            continue

        offsets.append(offset)
        diagonals.append(np.full(length, value, dtype=float))

    return diags(
        diagonals,
        offsets,
        shape=(n, n),
        format="csr",
    )


def comrade(n):
    # Deterministic project instance based on Purdue comrade.m.
    a = np.ones(n, dtype=float)
    b = 2.0 + 0.25 * np.linspace(0.0, 1.0, n)
    g = 0.5 + 0.10 * np.linspace(0.0, 1.0, n)
    c = 1.0 / (np.arange(n, dtype=float) + 1.0)

    upper = 1.0 / a[:-1]
    main = -(b / a)
    lower = g[1:] / a[1:]

    C = diags(
        diagonals=[lower, main, upper],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="lil",
    )

    last_col_update = c[::-1] / a[-1]

    for i in range(n):
        C[i, n - 1] = C[i, n - 1] - last_col_update[i]

    return C.tocsr()


def bf(n, rng):
    # Sparse construction of the helper bf(n) from Purdue butterfly.m.
    n1 = n // 2
    B0 = rng.standard_normal(n1)
    B1 = rng.standard_normal(n1)

    rows = []
    cols = []
    vals = []

    scale = 1.0 / np.sqrt(2.0)

    for i in range(n1):
        rows.extend([i, i])
        cols.extend([i, i + n1])
        vals.extend([B0[i] * scale, B1[i] * scale])

        rows.extend([i + n1, i + n1])
        cols.extend([i, i + n1])
        vals.extend([B0[i] * scale, -B1[i] * scale])

    return coo_matrix(
        (vals, (rows, cols)),
        shape=(n, n),
    ).tocsr()


def butterfly(n, levels, seed=379):
    # Translation of Purdue butterfly.m using sparse blocks.
    rng = np.random.default_rng(seed)

    m = n // (2 ** (levels - 1))
    n = m * (2 ** (levels - 1))
    n1 = m

    B = bf(n, rng)

    for k in range(levels - 1, 0, -1):
        blocks = [bf(n1, rng) for _ in range(2 ** k)]
        C = block_diag(blocks, format="csr")

        B = C @ B
        B.eliminate_zeros()

        n1 *= 2

    return B.tocsr()


def generate_suite(outdir):
    generated = []

    # Nonsymmetric banded Toeplitz matrices.
    band_cases = [
        (20000, [-2, 4, -1], "bandtoep_nonsym_n20000_um2_4_m1"),
        (50000, [-2, 4, -1], "bandtoep_nonsym_n50000_um2_4_m1"),
        (
            20000,
            [-3, -1, 6, -2, -1],
            "bandtoep_nonsym_n20000_um3_m1_6_m2_m1",
        ),
        (
            50000,
            [-3, -1, 6, -2, -1],
            "bandtoep_nonsym_n50000_um3_m1_6_m2_m1",
        ),
    ]

    for n, u, name in band_cases:
        A = bandtoep(u, n)
        generated.append(write_matrix(A, name, outdir))

    # Comrade matrices.
    for n in [5000, 20000, 50000]:
        A = comrade(n)
        name = f"comrade_n{n}"
        generated.append(write_matrix(A, name, outdir))

    # Butterfly matrices.
    butterfly_cases = [
        (4096, 3),
        (16384, 4),
        (32768, 4),
    ]

    for n, levels in butterfly_cases:
        A = butterfly(n, levels)
        name = f"butterfly_n{n}_l{levels}"
        generated.append(write_matrix(A, name, outdir))

    print(
        f"\nGenerated {len(generated)} nonsymmetric matrices in: {outdir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate additional nonsymmetric ECE379K sparse matrices."
    )

    parser.add_argument(
        "--outdir",
        default="generated_nonsymmetric_matrices",
        help="Output directory (default: generated_nonsymmetric_matrices)",
    )

    args = parser.parse_args()
    generate_suite(args.outdir)


if __name__ == "__main__":
    main()
