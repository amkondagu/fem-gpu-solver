import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite


def lap2d(k1, k2):
    # Direct translation of Purdue lap2d.m using Kronecker form.
    e = np.ones(k1)
    a = sparse.diags(
        [-e[:-1], 4 * e, -e[:-1]],
        offsets=[-1, 0, 1],
        shape=(k1, k1),
        format="csr",
    )
    I = sparse.eye(k1, format="csr")
    E = sparse.diags(
        [np.ones(k2 - 1), np.ones(k2 - 1)],
        offsets=[-1, 1],
        shape=(k2, k2),
        format="csr",
    )

    return (
        sparse.kron(sparse.eye(k2, format="csr"), a, format="csr")
        + sparse.kron(E, -I, format="csr")
    )


def lap2d7pt(k1, k2):
    # Direct translation of Purdue lap2d7pt.m block pattern.
    e = np.ones(k1)

    a = sparse.diags(
        [-e[:-1], 6 * e, -e[:-1]],
        offsets=[-1, 0, 1],
        shape=(k1, k1),
        format="csr",
    )

    b = sparse.diags(
        [-e, -e[:-1]],
        offsets=[0, 1],
        shape=(k1, k1),
        format="csr",
    )

    L = sparse.diags(
        [np.ones(k2 - 1)],
        offsets=[-1],
        shape=(k2, k2),
        format="csr",
    )

    U = sparse.diags(
        [np.ones(k2 - 1)],
        offsets=[1],
        shape=(k2, k2),
        format="csr",
    )

    return (
        sparse.kron(sparse.eye(k2, format="csr"), a, format="csr")
        + sparse.kron(L, b, format="csr")
        + sparse.kron(U, b.T, format="csr")
    )


def bandtoep(u, n):
    # Sparse equivalent of Purdue bandtoep.m.
    # u is the middle row pattern.
    u = np.asarray(u, dtype=float).reshape(-1)
    k = (len(u) + 1) // 2

    diagonals = []
    offsets = []

    for idx, value in enumerate(u):
        offset = idx - (k - 1)
        length = n - abs(offset)

        if length <= 0:
            continue

        diagonals.append(np.full(length, value, dtype=float))
        offsets.append(offset)

    return sparse.diags(
        diagonals,
        offsets,
        shape=(n, n),
        format="csr",
    )


def lap3d(k1, k2, k3):
    # Direct translation of Purdue lap3d.m in equivalent Kronecker form.
    T1 = bandtoep([-1, 6, -1], k1)
    I1 = sparse.eye(k1, format="csr")

    E2 = sparse.diags(
        [np.ones(k2 - 1), np.ones(k2 - 1)],
        offsets=[-1, 1],
        shape=(k2, k2),
        format="csr",
    )

    aa = (
        sparse.kron(sparse.eye(k2, format="csr"), T1, format="csr")
        + sparse.kron(E2, -I1, format="csr")
    )

    I2 = sparse.eye(k1 * k2, format="csr")

    E3 = sparse.diags(
        [np.ones(k3 - 1), np.ones(k3 - 1)],
        offsets=[-1, 1],
        shape=(k3, k3),
        format="csr",
    )

    return (
        sparse.kron(sparse.eye(k3, format="csr"), aa, format="csr")
        + sparse.kron(E3, -I2, format="csr")
    )


def elas2d(n, lam):
    # Translation of Purdue elas2d.m.
    # Kept close to source for reproducibility.
    N = 2 * (2 * n - 1) * 6
    A = sparse.eye(N, format="lil")
    D = np.zeros(2 * (2 * n - 1) * 7, dtype=float)

    for k in range(2, 2 * n + 1):
        for s in range(2, 7):
            ksd = 2 * (s - 2) * (2 * n - 1) + 2 * (k - 2)

            if (k % 2 == 0) and (s % 2 == 0):
                D[ksd:ksd + 2] = (8 + 4 * (lam + 1)) * np.array([1.0, 1.0])

            if (k % 2 == 1) and (s % 2 == 0):
                D[ksd:ksd + 2] = np.array([4 + 4 * (lam + 1), 4.0])

            if (k % 2 == 0) and (s % 2 == 1):
                D[ksd:ksd + 2] = np.array([4.0, 4 + 4 * (lam + 1)])

    D = np.sqrt(D)
    ly = 2 * (2 * n - 1)

    for k in range(2, 2 * n + 1):
        for s in range(4, 6):
            ksd = 2 * (s - 2) * (2 * n - 1) + 2 * (k - 2)

            if (k % 2 == 0) and (s % 2 == 0):
                den = np.outer(D[ksd:ksd + 2], D[ksd:ksd + 2])
                A[ksd:ksd + 2, ksd:ksd + 2] = np.array(
                    [
                        [8 + 4 * (lam + 1), -4 * (lam + 1)],
                        [-4 * (lam + 1), 8 + 4 * (lam + 1)],
                    ]
                ) / den

                if s > 2:
                    den = np.outer(D[ksd:ksd + 2], D[ksd - ly:ksd - ly + 2])
                    A[ksd:ksd + 2, ksd - ly:ksd - ly + 2] = np.array(
                        [[-2, 2 * (lam + 1)], [0, -2 - 2 * (lam + 1)]]
                    ) / den

                if s < 2 * n:
                    den = np.outer(D[ksd:ksd + 2], D[ksd + ly:ksd + ly + 2])
                    A[ksd:ksd + 2, ksd + ly:ksd + ly + 2] = np.array(
                        [[-2, 2 * (lam + 1)], [0, -2 - 2 * (lam + 1)]]
                    ) / den

                if k > 2:
                    den = np.outer(D[ksd:ksd + 2], D[ksd - 2:ksd])
                    A[ksd:ksd + 2, ksd - 2:ksd] = np.array(
                        [[-2 - 2 * (lam + 1), 0], [2 * (lam + 1), -2]]
                    ) / den

                if k < 2 * n:
                    den = np.outer(D[ksd:ksd + 2], D[ksd + 2:ksd + 4])
                    A[ksd:ksd + 2, ksd + 2:ksd + 4] = np.array(
                        [[-2 - 2 * (lam + 1), 0], [2 * (lam + 1), -2]]
                    ) / den

            if (k % 2 == 1) and (s % 2 == 0):
                if s > 2 and k < 2 * n:
                    den = np.outer(D[ksd:ksd + 2], D[ksd - ly + 2:ksd - ly + 4])
                    A[ksd:ksd + 2, ksd - ly + 2:ksd - ly + 4] = np.array(
                        [[0, -2 * (lam + 1)], [0, 0]]
                    ) / den

                if s < 2 * n and k > 2:
                    den = np.outer(D[ksd:ksd + 2], D[ksd + ly - 2:ksd + ly])
                    A[ksd:ksd + 2, ksd + ly - 2:ksd + ly] = np.array(
                        [[0, -2 * (lam + 1)], [0, 0]]
                    ) / den

                if k > 2:
                    den = np.outer(D[ksd:ksd + 2], D[ksd - 2:ksd])
                    A[ksd:ksd + 2, ksd - 2:ksd] = np.array(
                        [[-2 - 2 * (lam + 1), 2 * (lam + 1)], [0, -2]]
                    ) / den

                if k < 2 * n:
                    den = np.outer(D[ksd:ksd + 2], D[ksd + 2:ksd + 4])
                    A[ksd:ksd + 2, ksd + 2:ksd + 4] = np.array(
                        [[-2 - 2 * (lam + 1), 2 * (lam + 1)], [0, -2]]
                    ) / den

            if (k % 2 == 0) and (s % 2 == 1):
                if s > 2:
                    den = np.outer(D[ksd:ksd + 2], D[ksd - ly:ksd - ly + 2])
                    A[ksd:ksd + 2, ksd - ly:ksd - ly + 2] = np.array(
                        [[-2, 0], [2 * (lam + 1), -2 - 2 * (lam + 1)]]
                    ) / den

                    if k < 2 * n:
                        den = np.outer(D[ksd:ksd + 2], D[ksd - ly + 2:ksd - ly + 4])
                        A[ksd:ksd + 2, ksd - ly + 2:ksd - ly + 4] = np.array(
                            [[0, 0], [-2 * (lam + 1), 0]]
                        ) / den

                if s < 2 * n:
                    den = np.outer(D[ksd:ksd + 2], D[ksd + ly:ksd + ly + 2])
                    A[ksd:ksd + 2, ksd + ly:ksd + ly + 2] = np.array(
                        [[-2, 0], [2 * (lam + 1), -2 - 2 * (lam + 1)]]
                    ) / den

                    if k > 2:
                        den = np.outer(D[ksd:ksd + 2], D[ksd + ly - 2:ksd + ly])
                        A[ksd:ksd + 2, ksd + ly - 2:ksd + ly] = np.array(
                            [[0, 0], [-2 * (lam + 1), 0]]
                        ) / den

    A = A.tocsr()

    a1 = 2 * (2 * n - 1) * 2
    a2 = 2 * (2 * n - 1) * 3
    a3 = 2 * (2 * n - 1) * 4
    a4 = 2 * (2 * n - 1) * 5

    part1 = A[a1:a2, 2 * (2 * n - 1):a1]
    part2 = A[a1:a2, a1:a2]
    part3 = A[a1:a2, a2:a3]
    part4 = A[a2:a3, a1:a2]
    part5 = A[a2:a3, a2:a3]
    part6 = A[a2:a3, a3:a4]

    v10 = np.tile([1, 0], n - 1).tolist() + [1]
    v01 = np.tile([0, 1], n - 1).tolist() + [0]

    D10 = sparse.diags(v10, 0, format="csr")
    D01 = sparse.diags(v01, 0, format="csr")
    U10 = sparse.diags(
        np.tile([1, 0], n - 1),
        1,
        shape=(2 * n - 1, 2 * n - 1),
        format="csr",
    )
    U01 = sparse.diags(
        np.tile([0, 1], n - 1),
        1,
        shape=(2 * n - 1, 2 * n - 1),
        format="csr",
    )
    L10 = sparse.diags(
        np.tile([1, 0], n - 1),
        -1,
        shape=(2 * n - 1, 2 * n - 1),
        format="csr",
    )
    L01 = sparse.diags(
        np.tile([0, 1], n - 1),
        -1,
        shape=(2 * n - 1, 2 * n - 1),
        format="csr",
    )

    out = (
        sparse.kron(D10, part2, format="csr")
        + sparse.kron(D01, part5, format="csr")
        + sparse.kron(U10, part3, format="csr")
        + sparse.kron(U01, part6, format="csr")
        + sparse.kron(L10, part4, format="csr")
        + sparse.kron(L01, part1, format="csr")
    )

    return out.tocsr()


def save_matrix(A, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.mtx"

    # Force explicit Matrix Market "general" storage so the C++ reader
    # does not need to expand symmetric entries.
    mmwrite(path, A.tocoo(), symmetry="general")

    print(f"wrote {path}  shape={A.shape} nnz={A.nnz}")


def build_suite(outdir):
    specs = []

    for s in [32, 64, 128, 256, 512, 768]:
        specs.append(("lap2d", (s, s)))

    for s in [32, 64, 128, 256, 512, 768]:
        specs.append(("lap2d7pt", (s, s)))

    for s in [8, 12, 16, 24, 32]:
        specs.append(("lap3d", (s, s, s)))

    for n in [8, 16, 32, 64]:
        specs.append(("elas2d", (n, 10.0)))

    toeplitz_specs = [
        (20000, [-1, 2, -1]),
        (20000, [-1, -1, 4, -1, -1]),
        (50000, [-1, 2, -1]),
        (50000, [-1, -1, 4, -1, -1]),
    ]

    for kind, args in specs:
        if kind == "lap2d":
            A = lap2d(*args)
            name = f"lap2d_{args[0]}x{args[1]}"

        elif kind == "lap2d7pt":
            A = lap2d7pt(*args)
            name = f"lap2d7pt_{args[0]}x{args[1]}"

        elif kind == "lap3d":
            A = lap3d(*args)
            name = f"lap3d_{args[0]}x{args[1]}x{args[2]}"

        elif kind == "elas2d":
            A = elas2d(*args)
            name = f"elas2d_n{args[0]}_lambda{str(args[1]).replace('.', 'p')}"

        save_matrix(A, outdir, name)

    for n, u in toeplitz_specs:
        A = bandtoep(u, n)
        name = f"bandtoep_n{n}_u{'_'.join(str(x).replace('-', 'm') for x in u)}"
        save_matrix(A, outdir, name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Purdue-SMArt-derived sparse benchmark matrices."
    )

    parser.add_argument(
        "--outdir",
        default="generated_symmetric_matrices",
        help="output directory",
    )

    parser.add_argument(
        "--suite",
        action="store_true",
        help="generate the default 25-matrix benchmark suite",
    )

    sub = parser.add_subparsers(dest="cmd")

    q = sub.add_parser("lap2d")
    q.add_argument("k1", type=int)
    q.add_argument("k2", type=int)

    q = sub.add_parser("lap2d7pt")
    q.add_argument("k1", type=int)
    q.add_argument("k2", type=int)

    q = sub.add_parser("lap3d")
    q.add_argument("k1", type=int)
    q.add_argument("k2", type=int)
    q.add_argument("k3", type=int)

    q = sub.add_parser("elas2d")
    q.add_argument("n", type=int)
    q.add_argument("lam", type=float)

    q = sub.add_parser("bandtoep")
    q.add_argument("n", type=int)
    q.add_argument("u", nargs="+", type=float)

    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.suite:
        build_suite(outdir)
        return

    if args.cmd == "lap2d":
        A = lap2d(args.k1, args.k2)
        name = f"lap2d_{args.k1}x{args.k2}"

    elif args.cmd == "lap2d7pt":
        A = lap2d7pt(args.k1, args.k2)
        name = f"lap2d7pt_{args.k1}x{args.k2}"

    elif args.cmd == "lap3d":
        A = lap3d(args.k1, args.k2, args.k3)
        name = f"lap3d_{args.k1}x{args.k2}x{args.k3}"

    elif args.cmd == "elas2d":
        A = elas2d(args.n, args.lam)
        name = f"elas2d_n{args.n}_lambda{str(args.lam).replace('.', 'p')}"

    elif args.cmd == "bandtoep":
        A = bandtoep(args.u, args.n)
        name = f"bandtoep_n{args.n}"

    else:
        parser.print_help()
        return

    save_matrix(A, outdir, name)


if __name__ == "__main__":
    main()
