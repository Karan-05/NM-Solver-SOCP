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
package SDPT3v4_1;
import dev.nm.algebra.linear.matrix.doubles.ImmutableMatrix;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
import dev.nm.algebra.linear.vector.doubles.operation.VectorFactory;
import dev.nm.number.doublearray.DoubleArrayMath;
import dev.nm.solver.IterativeSolution;
import dev.nm.solver.multivariate.constrained.ConstrainedMinimizer;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPDualProblem1;
import dev.nm.stat.descriptive.moment.Mean;
import static java.lang.Math.sqrt;
import java.util.ArrayList;

/**
 * Solves a Dual Second Order Conic Programming problem using the Primal Dual
 * Interior Point algorithm. The dual problem is of the form:
 * \[
 * \max_y \mathbf{b'y} \\ \textrm{ s.t.,} 
 * \mathbf{\hat{A}_i'y + s_i = \hat{c}_i} \\
 * s_i \in K_i, i = 1, 2, ..., q
 * \]
 * <p/>
 *
 * @author Weng Bo
 * @author Ding Hao
 * @see
 * <ul>
 * <li>"Andreas Antoniou, Wu-Sheng Lu, "Algorithm 14.5, Section 14.8.2, A primal-dual interior-point
 * algorithm," Practical Optimization: Algorithms and Engineering Applications."</li>
 * <li>"K. C. Toh, M. J. Todd, R. H. Tütüncü, "On the implementation and usage of SDPT3 - a MATLAB
 * software package for semidefinite-quadratic-linear programming, version 4.0," in Handbook on
 * Semidefinite, Cone and Polynomial Optimization: Theory, Algorithms, Software and Applications,
 * Anjos, M. and Lasserre, J.B., ED. Springer, 2012, pp. 715--754."</li>
 * </ul>
 *
 * //2014/1/9: This solver is tested up to 6000 variables and 26000 constraints. It took 5 minutes to return the result.
  
 */


/**
 * The SOCP dual problem we are solving here is :
 * 
 * \max {\bm b}^T \hat{\bm y} \\
 * {\rm s.t.} ({\bm A_i^q})^T \hat{\bm y} + {\bm z_i^q} = c_i^q，\ {\bm z_i^q}\in \mathcal{K}_q^{q_i},\ for i\in [n_q];\\
 *            ({\bm A^{\ell}})^T \hat{\bm y} + {\bm z}^{\ell} = c^{\ell}，\ {\bm z}^{\ell} \ge 0;\\
 *            ({\bm A^u})^T \hat{\bm y} = c^u;\\
 *            \hat{\bm y} \in \mathbb{R}^m;\ {\bm z}^{\ell}\in \mathbb{R}^{n_{\ell}};\ {\bm z}^u \in \mathbb{R}^{n_u}.
 *              
 * We implement Primal-Dual Predictor-Corrector Interior Point Method to solve it.
 * 
 * Reference: "K. C. Toh, M. J. Todd, R. H. Tütüncü, "On the implementation and usage of SDPT3 - a MATLAB
 * software package for semidefinite-quadratic-linear programming, version 4.0," in Handbook on
 * Semidefinite, Cone and Polynomial Optimization: Theory, Algorithms, Software and Applications,
 * Anjos, M. and Lasserre, J.B., ED. Springer, 2012, pp. 715--754."
 * 
 * @author SherryNi 
 */


public class PrimalDualInteriorPointMinimizer1 implements ConstrainedMinimizer<SOCPDualProblem1, IterativeSolution<PrimalDualSolution>> {

    /**
     * This is the solution to a Dual Second-Order Conic Programming problem using the 
     * Primal-Dual Predictor-Corrector Interior Point algorithm.
     */
    
    public class Solution implements IterativeSolution<PrimalDualSolution> {

//        private static final double MINIMUM_STEP_SIZE = 1e-6; // use epsilon instead
//        private static final double MAXIMUM_INFEASIBILITY = 1e8; // use 1/epsilon instead
        private static final double SCALE_FOR_SLOW_PROGRESS = 5;
        private static final double SLOW_PROGRESS_RATIO = 0.4;
        private final ArrayList<Double> mu_history = new ArrayList<>();
        private PrimalDualSolution soln;
        private final SOCPDualProblem1 problem;
        
        // private final ImmutableMatrix A_q; // for all conic blocks
        private final ImmutableMatrix A_l; // for all linear blocks (A_l and A_u)
        private final ImmutableMatrix A_full; // for [A_q, A_l_new]
               
        // private final Vector c_q; // for all conic blocks
        private final Vector c_l; // for all linear blocks (A_l and A_u)
        private final Vector c_full;
                
        private final int n_q; // no. of conic blocks; note that dim of conic variables are problem.n(i), where i = 1, ..., n_q
        private final int n_l; 
        // private final int n_u; // important for heuristic step
        private final int n; // column size of A_full
        
        // private final ImmutableMatrix A_l_new;
        // private final Vector c_l_new;
        // private final int n_l_new;
        
        private final Vector b; 
        private final int m;
        // private final boolean flag_U; // indicate whether x_u is involved
        
        private PrimalDualInteriorPointIterationStep1 impl;

        private Solution(SOCPDualProblem1 problem, PrimalDualInteriorPointIterationStep1 impl) {
            this.problem = problem;
                        
            this.m = problem.m();  // dimemsion of \hat{y}
            this.b = problem.b();  // vector in objective function
                       
            // this.A_q = new ImmutableMatrix(problem.A_q_full()); 
            this.A_l = new ImmutableMatrix(problem.A_l_full()); 
                       
            this.A_full = new ImmutableMatrix(problem.A_full()); 
            this.c_full = problem.c_full();
            
            this.n = A_full.nCols();
            
            // this.c_q = problem.c_q_full(); 
            this.c_l = problem.c_l_full();
                        
            this.n_q = problem.n_q();  // no. of SOC constraints; each conic variable has dimension q_i, i = 1, ..., n_q
            this.n_l = A_l.nCols();  // no. of linear constraints (A_l and A_u together)
            // this.n_u = problem.n_u();  // dim of unrestricted variables
                       
            // this.flag_U = problem.flag_u(); // true means there exists unrestricted variables
                         
            this.impl = impl;
            
            /** TODO: add pre-processing as in Section 25.3.8.
             * a) detect and remove nearly-dependent constraints
             * b) detect problem structures, e.g., pure conic/ pure linear/ conic + linear 
             * c) detect unrestricted variables --- DONE by flag_u      
             * d) choose between HKM and NT search direction. Default is HKM.
             * e) choose whether to do corrector step. Default is Do.
             * f) choose adaptive step length(gamma) or fixed step size(input). Default is adaptive step length.
             * g) choose between chol, sparse chol, densecolumnhandling, LDL when solving M y = h. (Future) 
             * 
             * TODO: error msg returned can be more specific.
            */
            
        }

        /**
         * {@inheritDoc}
         *
         * c*x is the value of the prime objective function, and b*y is the value of the dual
         * objective function. In theory c*x=b*y. For an SOCP problem, the prime objective function
         * is minimized, while the dual objective function is maximized. Therefore, the method
         * "minimum()" return c*x.
         * @return 
         */
        @Override
        public double minimum() { // minimum of the primal problem
            // return problem.c().innerProduct(soln.x);
            return c_full.innerProduct(soln.x); 
            // Note that this is the optimal value of the SOCP Primal/Dual problem, not that of
            // the original portfolio problem. 
        }

        @Override
        public PrimalDualSolution minimizer() {
            return soln;
        }

        @Override
        public void setInitials(PrimalDualSolution... initials) {
            soln = initials[0];
        }

        @Override
        public PrimalDualSolution search(PrimalDualSolution... initials) throws Exception {
            switch (initials.length) {
                case 0:
                    return search();
                case 1:
                    return search(initials[0]);
                default:
                    throw new IllegalArgumentException("Wrong number of parameters for initialization");
            }
        }

        /**
         * Searches for a solution that optimizes the objective function from the given starting
         * point.
         *
         * @param initial an initial guess
         * @return an (approximate) optimizer
         * @throws Exception when an error occurs during the search
         */
        public PrimalDualSolution search(PrimalDualSolution initial) throws Exception {
            setInitials(initial);

            for (; mu_history.size() < maxIterations;) {
                int Iter = mu_history.size() + 1;
                
                System.out.println();
                System.out.println("Current iteration is " + Iter);
                
                if (!step()) {
                    break;
                }
            }
            return minimizer();
        }

        /**
         * [Previous Version] Searches for a solution that optimizes the objective function from 
         * the starting point given by K. C. Toh, SDPT3 Version 3.0, p. 6.
         * 
         * [Current version] Searches for a solution using initialization suggested by SDPT3 
         * version 4. See Section 25.3.7 of the reference paper.
         * 
         * @return an (approximate) optimizer
         * @throws Exception when an error occurs during the search
         */
        
        
        
        public PrimalDualSolution search() throws Exception {
            
            // Define initial point for conic variables, x_i^q, and z_i^q
            Vector[] eq_x = new Vector[n_q];
            Vector[] eq_z = new Vector[n_q];
            
            for (int i = 0; i < n_q; i++) {
                
                // Define e_i^q as in Section 25.3.7, e_i^q is the first q_i-dimensional unit vector
                eq_x[i] = new DenseVector(problem.n(i + 1)); // all 0 vector
                eq_z[i] = new DenseVector(problem.n(i + 1));
                
                /** Define xi_i^q and eta_i^q as in Section 25.3.7
                 * 
                 * xi_q(i) = max {10, \sqrt(n_i), \sqrt(n_i) * max(\frac{1+abs(b_k}{1 + norm (A_i^q(k, :)}}
                 * eta_q(i) = max{10, sqrt(n_i), max{norm(c), norm(A(k:)}}
                 * 
                **/
                
                double xi_q = Math.max(10, sqrt(problem.n(i + 1)));
                double eta_q = Math.max(10, sqrt(problem.n(i + 1)));
                eta_q = Math.max(eta_q , problem.c_q(i + 1).norm()); // note that problem.c_q(i) is different from c_q above (which is c_q_full)
                
                for (int k = 1; k <= m; k++){
                    double norm_Aiqk = problem.A_q(i + 1).getRow(k).norm();  // Rmk: problem.A_q(i) means A_q^i
                    double term1 = sqrt(problem.n(i + 1)) * (1 + Math.abs(b.get(k))) / (1 + norm_Aiqk);
                    if (term1 > xi_q){
                        xi_q = term1;
                    }
                    if (norm_Aiqk > eta_q){
                        eta_q = norm_Aiqk;
                    }
                }
                
                eq_x[i].set(1, xi_q);  // set the first entry to be xi_q
                eq_z[i].set(1, eta_q); // set the first entry to be eta_q
            }
            
            Vector x0_q = VectorFactory.concat(eq_x);
            Vector z0_q = VectorFactory.concat(eq_z);
            

            // Define initial point for linear variables including "l" and "u"(converted)
                      
            // int n_l_act = (flag_U)? n_l_new: n_l; 
            // Matrix A_l_act =(flag_U)? A_l_new : A_l;
           
            
            Vector el_x = new DenseVector(n_l); // TODO: How to create an all-one vector?
            Vector el_z = new DenseVector(n_l);
            
            double xi_l = Math.max(10, sqrt(n_l));
            // double eta_l = Math.max(Math.max(10, sqrt(n_l)), c_l.norm());
            double eta_l = DoubleArrayMath.max(10, sqrt(n_l), c_l.norm());
            
            for (int k = 1; k <= m; k++){
                double norm_Alk = A_l.getRow(k).norm();
                double term2 = sqrt(n_l) * (1 + Math.abs(b.get(k))) / (1 + norm_Alk);
                if (term2 > xi_l){
                    xi_l = term2;
                }
                if (norm_Alk > eta_l){
                    eta_l = norm_Alk;
                }
            }
            
            for (int j = 1; j <= n_l; j++){  // TODO: any efficient way to create an all-one vector first to avoid for loop? 
                el_x.set(j, xi_l);
                el_z.set(j, eta_l);
            }
            
            
            Vector x0_l = el_x;
            Vector z0_l = el_z;
        
            
            /* Previous codes by DingHao
            
            Vector[] eq = new Vector[problem.q()];
            for (int i = 0; i < problem.q(); i++) {
                eq[i] = new DenseVector(problem.n(i + 1)); // different sizes
                eq[i].set(1, 1);
            }
            Vector e_q = VectorFactory.concat(eq); 

            
            double[] AColumnNorms = AColumnNorms();
            double[] AtColumnNorms = AtColumnNorms();
            
            //compute xi
            double xi = 1;
            for (int i = 1, act = 0; i <= problem.q(); act += problem.n(i++)) { // update 'act' before updating i
               // Section 3.4
               // 
                
                double term1 = sqrt(problem.n(i));
                if (term1 > xi){
                    xi = term1;
                    }
                for (int j = 1; j <= Math.min(problem.n(i), problem.b().size()); j++) {
                 // double term = (1 + problem.b().get(j)) / (1 + AColumnNorms[j + act]);
                    double term = sqrt(problem.n(i)) * (1 + Math.abs(problem.b().get(j))) / (1 + AColumnNorms[j + act]);
                    if (term > xi) {
                        xi = term;
                    }
                }
            }
            // paper SDPT3 v4.0
            xi = Math.max(xi, 10);

            // compute eta
            double normMax = problem.c().norm(); // norm of C
            for (int j = 1; j <= At.nCols(); j++) {
                double normAt = AtColumnNorms[j];
                if (normAt > normMax) {
                    normMax = normAt;
                }
            }
            // int N = this.A.nCols();
            // double eta = max(1, (1 + normMax) / sqrt(N));
           
            
            // paper SDPT3 v4.0
            double eta = Math.max(10, normMax);
            //double eta = normMax;
            for (int i = 1; i <= problem.q(); i++) {
                double term2 = sqrt(problem.n(i));
                if (term2 > eta){
                    eta = term2;
                    }
                }
            
            // the starting/initial values
            Vector x0_q = e_q.scaled(xi);
            Vector s0_q = e_q.scaled(eta);
            
            */
                
            
            // Combine conic and linear parts to consititue an initial point
            // Remark: we dont involve x0_u, z0_u here as they are converted to "l" part.
            
            Vector x0 = VectorFactory.concat(x0_q, x0_l); 
            Vector s0 = VectorFactory.concat(z0_q, z0_l); 
            Vector y0 = new DenseVector(m); // first equation, "Section 3.4 Initial iterates", page 13, reference [2]
                                   
            PrimalDualSolution soln0 = new PrimalDualSolution(x0, s0, y0);
            
            System.out.println("x0 = " + x0);
            System.out.println("y0 = " + y0);
            System.out.println("z0 = " + s0);
            
            return search(soln0);
        }

        @Override
        public Boolean step() {
            
            final Vector Ax = A_full.multiply(soln.x); // not correct cuz size not the same
            final Vector Aty = A_full.t().multiply(soln.y);
            final Vector Atys = Aty.add(soln.s); 
            final double by = b.innerProduct(soln.y);
            final double cx = c_full.innerProduct(soln.x);
            
            double mu = soln.x.innerProduct(soln.s) / n;
            
            double rel_gap = (mu * n) / (1 + Math.abs(cx) + Math.abs(by));
            double pinfeas = (Ax.minus(b)).norm() / (1 + b.norm());
            double dinfeas = (Atys.minus(c_full)).norm() / (1 + c_full.norm());
            
            /* @DingHao
            final Vector Ax = A.multiply(soln.x);
            final Vector Aty = At.multiply(soln.y);
            final Vector Atys = Aty.add(soln.s);
            final double by = problem.b().innerProduct(soln.y);
            final double cx = problem.c().innerProduct(soln.x);

            double mu = soln.x.innerProduct(soln.s) / A.nCols();

            double rel_gap = (mu * A.nCols()) / (1 + Math.abs(cx) + Math.abs(by)); // eq. 6, ref. [2]
            double pinfeas = (Ax.minus(problem.b())).norm() / (1 + problem.b().norm()); // eq. 6, ref. [2]
            double dinfeas = (Atys.minus(problem.c())).norm() / (1 + problem.c().norm()); // eq. 6, ref. [2]
            */
            
            double phi = DoubleArrayMath.max(rel_gap, pinfeas, dinfeas); // eq. 6, ref. [2]
            double P_infeas_ind = by / (Atys.norm());
            double D_infeas_ind = - cx / Ax.norm();
            
            
            boolean isSlow = isSlow(mu);
            
            System.out.println();
            System.out.println("Current Phi = " + phi);
            // System.out.println("Pinfeas = " + pinfeas);
            // System.out.println("Dinfeas = " + dinfeas);
            // System.out.println("Primal infeasibility indicator is = " + P_infeas_ind);
            // System.out.println("Dual infeasibility indicator is = " + D_infeas_ind);
            System.out.println("Slow progress indicator is = " + (mu < epsilon && phi < epsilon && isSlow));
            System.out.println("Small stepsize indicator is = " + (impl.stepSize() < epsilon));
            System.out.println();
            /*
             * The algorithm stops when relative gap is smaller than epsilon and infeasible measure is
             * smaller than epsilon,
             * or primal infeasibility is suggested,
             * or dual infeasibility is suggested,
             * or slow progress is detected,
             * or numerical problems are encountered (if the algorithm is trying to inverse a
             * singular matrix),
             * or step size is too small.
             * See "Section 3.3 Stopping criteria", page 12-13, reference [2].
             */
            if ((phi < epsilon)                                 // 1. desired accuracy
                || (P_infeas_ind > 1. / epsilon)                // 2. primal infeasibility
                || (D_infeas_ind > 1. / epsilon)                // 3. dual infeasibility
                || (mu < epsilon && phi < epsilon && isSlow)    // 4. detect slow progress. TODO: double check 
                || impl.stepSize() < epsilon)                   // 6. detect small step size
                                                                // 5. detect numerical problems, eg non-PD matrices, see Exception
                {
                return false;  // stop based on termination criteria
            }

            PrimalDualSolution soln1 = impl.iterate(soln, Ax, Aty, mu);
            
            if (soln1 != null && Double.compare(soln1.y.get(1), Double.NaN) != 0) {
                soln = soln1;
                return true;  // continue iteration
            }
            
            System.out.println("Final Phi = " + phi);
            
            return false; // stop due to unexpected reasons such as degeneracy
            
        }

        /**
         * The detection of slow progress is not documented in reference [2]. We use the MATLAB
         * program "sqlpmain.m" in SDPT3 package as the reference.
         *
         * If the number of iterations is larger than 2, then the converge rate of mu is
         * calculated and slow progression is checked. The converge rate is the mean of mu
         * ratios in the last 5 iterations.
         *
         */
        private boolean isSlow(double mu) {
            mu_history.add(mu);
            int n = mu_history.size();

            Mean average_convergence_rate = new Mean();
            if (mu_history.size() > 1) { 
                for (int i = 1; i <= 5; i++) {
                    if (n - i - 1 >= 0) {
                        double rate = mu_history.get(n - i) / mu_history.get(n - i - 1);
                        average_convergence_rate.addData(rate);
                    }
                }

                // The prime dual gap, mu, is considered as reduced only when
                // the rate mu(n)/mu(n-1) < SLOW_PROGRESS_RATIO.
                double rate = mu_history.get(n - 1) / mu_history.get(n - 2); // the smaller the better
                if ((rate < SLOW_PROGRESS_RATIO)) {
                    return false;
                }

                if (rate < SCALE_FOR_SLOW_PROGRESS * average_convergence_rate.value()) {
                    return false;
                }
            }

            return true; // default
        }

        /*
        private double[] AColumnNorms() {
            return rowNorms(At);
        }

        private double[] AtColumnNorms() {
            return rowNorms(A);
        }

        private double[] rowNorms(final Matrix A) {
            double[] norms = new double[A.nRows() + 1];
            // parallelize norms computation
            Arrays.parallelSetAll(norms, new IntToDoubleFunction() {
                @Override
                public double applyAsDouble(int j) {
                    if (j == 0) {
                        return 0.;
                    }
                    return A.getRow(j).norm();
                }
            });
            return norms;
        }
        */
        
    }

    private final double epsilon;
    private final int maxIterations;

    /**
     * Constructs a Primal Dual Interior Point minimizer to solve Dual Second Order Conic
     * Programming problems.
     *
     * @param epsilon       a precision parameter: when a number |x| &le; &epsilon;, it is
     *                      considered 0
     * @param maxIterations the maximum number of iterations
     */
    public PrimalDualInteriorPointMinimizer1(double epsilon, int maxIterations) {
        this.epsilon = epsilon;
        this.maxIterations = maxIterations;
    }

    @Override
    public Solution solve(SOCPDualProblem1 problem) throws Exception {
  //     PrimalDualInteriorPointIterationStep impl = new AntoniouLu2007(problem, sigma);
        
//       PrimalDualInteriorPointIterationStep impl = new SDPT3v4(problem);
       //PrimalDualInteriorPointIterationStep1 impl = null; // TODO: replace with an implementation that works on SOCPDualProblem1
       //PrimalDualInteriorPointIterationStep1 impl = new SDPT3v4_1(problem);
       //return solve(problem, impl);
       
       if (problem.flag_s()){ // pure conic
           PrimalDualInteriorPointIterationStep1 impl = new SDPT3v4(problem);
           return solve(problem, impl);
       }
       else{
           PrimalDualInteriorPointIterationStep1 impl = new SDPT3v4_1(problem);
           return solve(problem, impl);
       }
    }

    // for debugging
    Solution solve(SOCPDualProblem1 problem, PrimalDualInteriorPointIterationStep1 impl) throws Exception {
        return new Solution(problem, impl);
    }
}