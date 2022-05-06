/*
 * Copyright (c) NM LTD.
 * https://nm.dev/
 * 
 * THIS SOFTWARE IS LICENSED, NOT SOLD.
 * 
 * YOU MAY USE THIS SOFTWARE ONLY AS DESCRIBED IN THE LICENSE.
 * IF YOU ARE NOT AWARE OF AND/OR DO NOT AGREE TO THE TERMS OF THE LICENSE,
 * DO NOT USE THIS SOFTWARE.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITH NO WARRANTY WHATSOEVER,
 * EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION,
 * ANY WARRANTIES OF ACCURACY, ACCESSIBILITY, COMPLETENESS,
 * FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABILITY, NON-INFRINGEMENT, 
 * TITLE AND USEFULNESS.
 * 
 * IN NO EVENT AND UNDER NO LEGAL THEORY,
 * WHETHER IN ACTION, CONTRACT, NEGLIGENCE, TORT, OR OTHERWISE,
 * SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIMS, DAMAGES OR OTHER LIABILITIES,
 * ARISING AS A RESULT OF USING OR OTHER DEALINGS IN THE SOFTWARE.
 */
package dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem;

import dev.nm.algebra.linear.matrix.doubles.ImmutableMatrix;
import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixFactory;
import dev.nm.algebra.linear.vector.doubles.ImmutableVector;
import dev.nm.algebra.linear.vector.doubles.SubVectorRef;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.operation.VectorFactory;
import dev.nm.analysis.function.rn2r1.RealScalarFunction;
import dev.nm.misc.ArgumentAssertion;
import dev.nm.solver.multivariate.constrained.constraint.LessThanConstraints;
import dev.nm.solver.multivariate.constrained.problem.ConstrainedOptimProblem;
import dev.nm.solver.multivariate.constrained.problem.ConstrainedOptimProblemImpl1;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

/**
 * This is the Dual Second Order Conic Programming problem.
 * \[
 * \max_y \mathbf{b'y} \textrm{ s.t.,} \\
 * \mathbf{{A^q}_i'y + s_i = c^q_i}, s_i \in K_i, i = 1, 2, ..., q \\
 * \mathbf{{A^{\ell}}^T y + z^{\ell} = c^{\ell}} \\
 * \mathbf{{A^{u}}^T y = c^{u}} \\
 * \]
 *
 * @author Haksun Li
 * @see "Andreas Antoniou, Wu-Sheng Lu, "eq 14.102, Section 14.7, Second-Order
 * Cone Programming, Practical Optimization: Algorithms and Engineering
 * Applications."
 */
public class SOCPDualProblem1 implements ConstrainedOptimProblem {

    //<editor-fold defaultstate="collapsed" desc="EqualityConstraints">
    public static class EqualityConstraints implements dev.nm.solver.multivariate.constrained.constraint.EqualityConstraints {

        private final ImmutableVector b;
        private final ImmutableMatrix[] A;
        private final ImmutableVector[] c;
        private final int m;
        private final int q;

        /**
         * Constructs the equality constraints for a dual SOCP problem,
         * \[
         * \max_y \mathbf{b'y} \textrm{ s.t.,} \\
         * \mathbf{{A^q}_i'y + s_i = c^q_i}, s_i \in K_i, i = 1, 2, ..., q \\
         * \mathbf{{A^{\ell}}^T y + z^{\ell} = c^{\ell}} \\
         * \mathbf{{A^{u}}^T y = c^{u}} \\
         * \]
         *
         * @param b \(b\)
         * @param A \(A_i\)'s
         * @param c \(c_i\)'s
         */
        public EqualityConstraints(Vector b, Matrix[] A, Vector[] c) {
            this.m = b.size();
            this.b = new ImmutableVector(b);

            this.q = A.length;
            this.A = new ImmutableMatrix[q];
            for (int i = 0; i < A.length; ++i) {
                this.A[i] = new ImmutableMatrix(A[i]);
            }

            this.c = new ImmutableVector[q];
            for (int i = 0; i < c.length; ++i) {
                this.c[i] = new ImmutableVector(c[i]);
            }
        }

        @Override
        public List<RealScalarFunction> getConstraints() {
            List<RealScalarFunction> h = new ArrayList<>();

            for (int j = 0; j < q; ++j) {
                final int i = j;
                h.add(
                    new RealScalarFunction() {
                    @Override
                    public Double evaluate(Vector x) {
                        Vector y = new SubVectorRef(x, 1, m);

                        int begin = m + 1;
                        int end = m + A[0].nCols();
                        for (int k = 1; k <= i; ++k) {//TODO: check logic; not sure if this is correct
                            begin += A[k - 1].nCols();
                            end += A[k].nCols();
                        }
                        Vector si = new SubVectorRef(x, begin, end);

                        Vector sum = A[i].t().multiply(y).add(si).minus(c[i]);
                        return sum.norm();
                    }

                    @Override
                    public int dimensionOfDomain() {
                        int dim = m + Stream.of(A).mapToInt(Matrix::nCols).sum();
                        return dim;
                    }

                    @Override
                    public int dimensionOfRange() {
                        return 1;
                    }
                });
            }

            return h;
        }

        @Override
        public int dimension() {
            int dim = m + Stream.of(A).mapToInt(Matrix::nCols).sum();
            return dim;
        }

        @Override
        public int size() {
            return q;
        }
    }
    //</editor-fold>

    private final ConstrainedOptimProblem problem;
    private final Vector b;
    private final int m; // y's dimension

    /* conic constraints */
    private final Matrix[] A_q;
    private final Vector[] c_q;
    private final int n_q; // no. of conic constraints

    /* linear inequalities */
    private final Matrix A_l;
    private final Vector c_l;
    private final int n_l; // no. of linear inequalities

    /* linear equalities */
    private final Matrix A_u;
    private final Vector c_u;
    private final int n_u; // no. of linear equalities

    /* Data combined*/
    private Matrix[] A_full;
    private Vector[] c_full;
    //private Matrix[] A_q_full;
    //private Vector[] c_q_full;
    private Matrix[] A_l_full;
    private Vector[] c_l_full;

    /* Indicators */
    private boolean flag_u;
    private boolean flag_s; // problem structure

    private final Map<String, Object> cache = new ConcurrentHashMap<>();

    /**
     * Constructs a dual SOCP problem.
     * \[
     * \max_y \mathbf{b'y} \textrm{ s.t.,} \\
     * \mathbf{{A^q}_i'y + s_i = c^q_i}, s_i \in K_i, i = 1, 2, ..., q \\
     * \]
     *
     * @param b   \(b\)
     * @param A_q \({A^q}_i\)'s
     * @param c_q \(c^q_i\)'s
     */
    public SOCPDualProblem1(
        Vector b,
        Matrix[] A_q,
        Vector[] c_q
    ) {
        this(b, A_q, c_q, null, null, null, null);
        flag_s = true;
    }

    /**
     * Constructs a dual SOCP problem.
     * \[
     * \max_y \mathbf{b'y} \textrm{ s.t.,} \\
     * \mathbf{{A^q}_i'y + s_i = c^q_i}, s_i \in K_i, i = 1, 2, ..., q \\
     * \mathbf{{A^{\ell}}^T y + z^{\ell} = c^{\ell}} \\
     * \mathbf{{A^{u}}^T y = c^{u}} \\
     * \]
     *
     * @param b   \(b\)
     * @param A_q \({A^q}_i\)'s
     * @param c_q \(c^q_i\)'s
     * @param A_l \(A^{\ell}\)
     * @param c_l \(c^l\)
     * @param A_u \(A^u\)
     * @param c_u \(c^u\)
     */
    public SOCPDualProblem1(
        Vector b,
        Matrix[] A_q,
        Vector[] c_q,
        Matrix A_l,
        Vector c_l,
        Matrix A_u,
        Vector c_u
    ) {
        
        flag_s = false;
        
        this.m = b.size();
        this.b = b;

        this.n_q = A_q.length;

        // Add row dimension check
        ArgumentAssertion.assertTrue(
            A_q[0].nRows() == m,
            "A_q^all #rows (%d) must equal to b size (%d)",
            A_q[0].nRows(), m
        );

        this.A_q = new Matrix[n_q];
        this.c_q = new Vector[n_q];
        for (int i = 0; i < n_q; ++i) {
            ArgumentAssertion.assertTrue(
                A_q[i].nCols() == c_q[i].size(),
                "A_q[%d] #columns (%d) must equal to c_q[%d] size (%d)",
                i, A_q[i].nCols(), i, c_q[i].size()
            );
            this.A_q[i] = A_q[i];
            this.c_q[i] = c_q[i];
        }

        if (A_l != null && c_l != null) {
            ArgumentAssertion.assertTrue(
                A_l.nRows() == m,
                "A_l #rows (%d) must equal to b size (%d)",
                A_l.nRows(), m
            );
            ArgumentAssertion.assertTrue(
                A_l.nCols() == c_l.size(),
                "A_l #columns (%d) must equal to c_l size (%d)",
                A_l.nCols(), c_l.size()
            );
        }
        this.A_l = A_l;
        this.c_l = c_l;
        this.n_l = (A_l == null) ? 0 : A_l.nCols();

        this.A_u = A_u;
        this.c_u = c_u;
        this.n_u = (A_u == null) ? 0 : A_u.nCols();
        if (A_u != null && c_u != null) {
            ArgumentAssertion.assertTrue(
                A_u.nRows() == m,
                "A_u #rows (%d) must equal to b size (%d)",
                A_u.nRows(), m
            );
            ArgumentAssertion.assertTrue(
                A_u.nCols() == c_u.size(),
                "A_u #columns (%d) must equal to c_u size (%d)",
                A_u.nCols(), c_u.size()
            );
        }

        this.problem
            = newConstrainedOptimProblem(b, n_q, A_q, c_q, A_l, c_l, A_u, c_u);
    }

    /**
     * Copy constructor.
     *
     * @param that another {@linkplain SOCPDualProblem1}
     */
    public SOCPDualProblem1(SOCPDualProblem1 that) {
        this.problem = that.problem;
        this.b = that.b;
        this.m = that.m;
        this.A_q = that.A_q.clone();
        this.c_q = that.c_q.clone();
        this.n_q = that.n_q;
        this.A_l = that.A_l;
        this.c_l = that.c_l;
        this.n_l = that.n_l;
        this.A_u = that.A_u;
        this.c_u = that.c_u;
        this.n_u = that.n_u;
        this.A_full = that.A_full.clone();
        this.c_full = that.c_full.clone();
        this.A_l_full = that.A_l_full.clone();
        this.c_l_full = that.c_l_full.clone();
        this.flag_u = that.flag_u;
        this.flag_s = that.flag_s;
    }

    private ConstrainedOptimProblem newConstrainedOptimProblem(
        Vector b,
        int n_q,
        Matrix[] A_q,
        Vector[] c_q,
        Matrix A_l,
        Vector c_l,
        Matrix A_u,
        Vector c_u
    ) {

        flag_u = (A_u != null); // Indicate whether unrestriced blocks are involved; flag_u = true means yes.

        if (flag_u) {

            /**
             * when restricted blocks are involved, we convert it to linear
             * inequalities.
             * We convert linear equality constraints to two linear
             * inequalities, i.e.,
             * (A^u)^T \hat{y} + z_u^{+} = c^u, z_u^{+} \ge 0;
             * -(A^u)^T \hat{y} + z_u^{-} = - c^u, z_u^{-} \ge 0;
             * to ensure the positive-definiteness of matrix \mathcal{M}.
             *
             * In this case,
             * A_l_new = [A_l, A_u, -A_u]; A_full = [A_q, A_l_new]
             * c_l_new = [c_l; c_u; -c_u]; A_full = [A_q, A_l_new]
             * n_l_new = n_l + 2 * n_u; n = n_q + n_l_new
             *
             * x_u = x_u^{+} - x_u^{-};
             *
             * In order to ameliorate the ill-conditioned problem that might
             * arise,
             * we will modify x_u^{+}, x_u^{-}, z_u^{+}, z_u^{-} in each
             * iteration
             * using the following heuristics.
             *
             * x_u^{+} = x_u^{+} - 0.8 * min(x_u^{+}, x_u^{-});
             * x_u^{-} = x_u^{-} - 0.8 * min(x_u^{+}, x_u^{-});
             * z_u^{+} = z_u^{+} + 0.8 * max(z_u^{+}, z_u^{-});
             * z_u^{-} = z_u^{-} + 0.8 * max(z_u^{+}, z_u^{-});
             *
             * This follows Section 25.5.6 for computational efficiency.
             *
             */
            // Define data matrix array combined
            int n = n_q + ((A_l != null) ? 1 : 0) + 2; // size of array storing [A_q, (A_l,) A_u, -A_u]

            A_full = new Matrix[n];
            System.arraycopy(A_q, 0, A_full, 0, n_q);
            if (A_l != null) {
                A_full[n_q] = A_l;
            }
            A_full[n - 2] = A_u;
            A_full[n - 1] = A_u.scaled(-1);

            c_full = new Vector[n];
            System.arraycopy(c_q, 0, c_full, 0, n_q); // combine all c's
            if (c_l != null) {
                c_full[n_q] = c_l;
            }
            c_full[n - 2] = c_u;
            c_full[n - 1] = c_u.scaled(-1);

            // Define linear blocks combined
            int n_l_full = n - n_q;

            A_l_full = new Matrix[n_l_full];
            if (A_l != null) {
                A_l_full[0] = A_l;
            }
            A_l_full[n_l_full - 2] = A_u;
            A_l_full[n_l_full - 1] = A_u.scaled(-1);

            c_l_full = new Vector[n_l_full];
            if (c_l != null) {
                c_l_full[0] = c_l;
            }
            c_l_full[n_l_full - 2] = c_u;
            c_l_full[n_l_full - 1] = c_u.scaled(-1);

        } else { // when restriced blocks are not involved

            // Define data matrix array combined
            int n = n_q + ((A_l != null) ? 1 : 0); // size of array storing [A_q, (A_l)]

            A_full = new Matrix[n];
            System.arraycopy(A_q, 0, A_full, 0, n_q);
            if (A_l != null) {
                A_full[n_q] = A_l;
            }

            c_full = new Vector[n];
            System.arraycopy(c_q, 0, c_full, 0, n_q);
            if (c_l != null) {
                c_full[n_q] = c_l;
            }

            // Define linear blocks combined
            int n_l_full = n - n_q;
            A_l_full = new Matrix[n_l_full]; // TODO: how to simplify this? As A_l_full in this case is either null or just A_l
            if (A_l != null) {
                A_l_full[0] = A_l;
            }

            c_l_full = new Vector[n_l_full]; // TODO: how to simplify this? As c_l_full in this case is either null or just c_l   
            if (c_l != null) {
                c_l_full[0] = c_l;
            }

        }

        /*
        int n = n_q + ((A_l != null) ? 1 : 0) + ((A_u != null) ? 1 : 0);

        Matrix[] A = new Matrix[n]; // combine all A's
        System.arraycopy(A_q, 0, A, 0, n_q);
        if (A_l != null) {
            A[n_q] = A_l;
        }
        if (A_u != null) {
            A[n - 1] = A_u;
        }

        Vector[] c = new Vector[n]; // combine all c's
        System.arraycopy(c_q, 0, c, 0, n_q);
        if (c_l != null) {
            c[n_q] = c_l;
        }
        if (c_u != null) {
            c[n - 1] = c_u;
        }
         */
        return new ConstrainedOptimProblemImpl1(
            new RealScalarFunction() {
            @Override
            public Double evaluate(Vector x) {
                final int m = b.size();
                Vector y = new SubVectorRef(x, 1, m);
                double by = b.innerProduct(y);
                return by;
            }

            @Override
            public int dimensionOfDomain() {
                // int dim = b.size() + Stream.of(A).mapToInt(Matrix::nCols).sum();
                int dim = b.size() + Stream.of(A_full).mapToInt(Matrix::nCols).sum();
                return dim;
            }

            @Override
            public int dimensionOfRange() {
                return 1;
            }
        },
            new EqualityConstraints(b, A_full, c_full),
            // new EqualityConstraints(b, A, c),
            null
        );
    }

    @Override
    public int dimension() {
        return problem.dimension(); // Note that this is not the true dimension of the SOCPDualproblem
    }

    @Override
    public RealScalarFunction f() {
        return problem.f();
    }

    @Override
    public LessThanConstraints getLessThanConstraints() {
        return problem.getLessThanConstraints();
    }

    @Override
    public EqualityConstraints getEqualityConstraints() {
        return (EqualityConstraints) problem.getEqualityConstraints();
    }

    /**
     * Gets the dimension of the system, i.e., <i>m</i> = the dimension of
     * <i>y</i>.
     *
     * @return the dimension of the system
     */
    public int m() {
        return m;
    }

    /**
     * Gets the total number of \({A^q}_i\) matrices.
     *
     * @return the number of \({A^q}_i\) matrices
     */
    public int n_q() {
        return n_q;
    }

    /**
     * Gets <i>b</i>.
     *
     * @return <i>b</i>
     */
    public Vector b() {
        return b;
    }

    /**
     * Gets \(c^q_i\).
     *
     * @param i an index to the \(c^q_i\)'s, counting from 1
     * @return \(c^q_i\)
     */
    public Vector c_q(int i) {
        return c_q[i - 1];
    }

    /**
     * Gets \({A^q}_i\).
     *
     * @param i an index to the \({A^q}_i\)'s, counting from 1
     * @return \({A^q}_i\)
     */
    public Matrix A_q(int i) {
        return A_q[i - 1];
    }

    /**
     * Gets the number of columns of \({A^q}_i\).
     * i.e., n_i
     *
     * @param i an index to the \({A^q}_i\)'s, counting from 1
     * @return the number of columns of \({A^q}_i\)
     */
    public int n(int i) {
        Matrix Atemp = A_q[i - 1];
        return Atemp.nCols();
    }

    /**
     * Combine all A data as a matrix.
     *
     * @return [{A_q^i}_i^{n_q}, (A_l)] if flag_u = false;
     *         [{A_q^i}_i^{n_q}, (A_l), A_u, -A_u] if flag_u = true (remember to do
     *         heuristic step during iteration in this case.)
     */
    public Matrix A_full() {
        Matrix Aq_temp = MatrixFactory.cbind(A_q);
        Matrix Al_temp = MatrixFactory.cbind(A_l_full);
        return MatrixFactory.cbind(Aq_temp, Al_temp);

        //return (Matrix) cache.computeIfAbsent("A_full", t -> MatrixFactory.cbind(A_q, A_l_full));
    }

    public Vector c_full() {
        Vector cq_temp = VectorFactory.concat(c_q);
        Vector cl_temp = VectorFactory.concat(c_l_full);
        return VectorFactory.concat(cq_temp, cl_temp);

        //return (Vector) cache.computeIfAbsent("c_full", t -> VectorFactory.concat(c_q, c_l_full));
    }

    /**
     * A^q = [{A^q}_1, {A^q}_2, ... {A^q}_{n_q}]
     *
     * @return \({A^q}\)
     * @see "Andreas Antoniou, Wu-Sheng Lu, "Section 14.8.1, Assumptions and KKT
     * conditions, Practical Optimization: Algorithms and Engineering
     * Applications."
     */
    public Matrix A_q_full() {
        return (Matrix) cache.computeIfAbsent("A_q", t -> MatrixFactory.cbind(A_q));
    }

    public Vector c_q_full() {
        return (Vector) cache.computeIfAbsent("c_q", t -> VectorFactory.concat(c_q));
    }

    /**
     * Combine all linear blocks as a matrix.
     *
     * @return [(A_l)] if flag_u = false;
     *         [(A_l), A_u, -A_u] if flag_u = true (remember to do heuristic step
     *         during iteration in this case.)
     */
    public Matrix A_l_full() {
        return (Matrix) cache.computeIfAbsent("A_l_full", t -> MatrixFactory.cbind(A_l_full));
    }

    public Vector c_l_full() {
        return (Vector) cache.computeIfAbsent("c_l_full", t -> VectorFactory.concat(c_l_full));
    }

    /**
     * @return true if unrestricted blocks are involved and thus heuristic step
     *         is required.
     */
    public boolean flag_u() {
        return flag_u;
    }
    
    /**
     * @return true if the input problem is pure conic; otherwise it is conic + linear 
     */
    public boolean flag_s() {
        return flag_s;
    }

    public int n_l() {
        return n_l;
    }

    public Matrix A_l() {
        return A_l;
    }

    public Vector c_l() {
        return c_l;
    }

    public int n_u() {
        return n_u;
    }

    public Matrix A_u() {
        return A_u;
    }

    public Vector c_u() {
        return c_u;
    }

    /**
     * TODO: Perhaps we need another indicator to obtain problem structure,
     * i.e. 1) pure conic constraints
     * 2) conic + linear
     * 3) pure linear
     *              *
     * Currently we only use flag_u to check if A_u is involved and then convert
     * A_u to part of A_l_new.
     *
     * Later on, we should also allow user to choose if they want to keep A_u
     * intact and use LDL factorization etc.
     */
}
