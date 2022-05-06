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
package nm_socp;

import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPDualProblem1;
import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.algebra.linear.matrix.doubles.factorization.triangle.cholesky.Chol;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.BackwardSubstitution;
import dev.nm.algebra.linear.matrix.doubles.linearsystem.ForwardSubstitution;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.triangle.LowerTriangularMatrix;
import dev.nm.algebra.linear.matrix.doubles.operation.MatrixFactory;
import dev.nm.algebra.linear.matrix.doubles.operation.OuterProduct;
import dev.nm.algebra.linear.vector.doubles.SubVectorRef;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.algebra.linear.vector.doubles.dense.DenseVector;
// import dev.nm.misc.parallel.LoopBody;
// import dev.nm.misc.parallel.MultipleExecutionException;
// import dev.nm.misc.parallel.ParallelExecutor;
// import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPDualProblem;
import dev.nm.algebra.linear.matrix.doubles.factorization.eigen.EigenDecomposition;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.diagonal.DiagonalMatrix;
import dev.nm.algebra.linear.vector.doubles.operation.VectorFactory;
import java.util.stream.IntStream;

// import static java.lang.Math.sqrt;

/**
 * This implements Algorithm_IPC, the SOCP interior point algorithm in SDPT3
 * version 4.
 *
 * @author Ding Hao
 * @see "K. C. Toh, M. J. Todd, R. H. Tütüncü, "On the implementation and usage
 * of SDPT3 - a MATLAB software package for semidefinite-quadratic-linear
 * programming, version 4.0," in Handbook on Semidefinite, Cone and Polynomial
 * Optimization: Theory, Algorithms, Software and Applications, Anjos, M. and
 * Lasserre, J.B., ED. Springer, 2012, pp. 715--754."
 */
public class SDPT3v4_1 extends PrimalDualInteriorPointIterationStep1 {

    private double gamma = 0.9; // the scale parameter of the step size
    
    private final int[] indices; // offset of sub-matrices Ai
    private final Matrix A_q;
    private final Matrix A_l;
    
    private final boolean flag_u;
    
    
    /**
     * Constructs an instance of the SOCP interior point algorithm using the an
     * algorithm similar to that of SDPT3, version 4.
     */
    
    SDPT3v4_1(SOCPDualProblem1 problem) {
        super(problem);
                
        this.indices = new int[q]; // indices[i+1] denots column index of A_q^i
        for (int i = 1, index = 1; i <= q; i++) {
            indices[i - 1] = index;
            index += problem.n(i);
        }
        
        this.A_q = problem.A_q_full();
        this.A_l = problem.A_l_full();
        
        this.flag_u = problem.flag_u();
               
    }

    // In this implementation, we have both quadratic "q" term, namely A^q, x^q, and 
    // linear "l" term, namely A^l, x^l.
    // Our 's' is the 'z' in the reference.
    
    @Override
    public PrimalDualSolution iterate(PrimalDualSolution soln0, Vector Ax, Vector Aty, double mu) {
        // For testing
        // System.out.println("y = " + soln0.y);
       
        // Predictor Step 
        // Herein, sigma = 0; the following works for conic + linear model. 
       
            // TODO: flag for conic + linear/ pure conic/ pure linear.
            // TODO: choose between HKM direction and NT direction 
        
        // Define dinfeas if it is not passed here
        final Vector Atys = Aty.add(soln0.s); 
        Vector c = problem.c_full();
        double dinfeas = (Atys.minus(c)).norm() / (1 + c.norm());
        // TODO: can comment the above definition if dinfeas is passed from Minimizer
        
        
        Vector Rp = problem.b().minus(Ax);
        
        Vector zq = new SubVectorRef(soln0.s, 1, nq);
        Vector zl = new SubVectorRef(soln0.s, nq + 1, n);
        Vector Aqty = A_q.t().multiply(soln0.y);  
        Vector Alty = A_l.t().multiply(soln0.y);
        //Note that Aty = concat(Aqty, Alty)
        
        Vector Rd_q = problem.c_q_full().minus(zq).minus(Aqty);
        Vector Rd_l = problem.c_l_full().minus(zl).minus(Alty);
        
        Vector xq = new SubVectorRef(soln0.x, 1, nq);
        Vector xl = new SubVectorRef(soln0.x, nq + 1, n);
        Vector Rc_q = xq.scaled(-1);
        Vector Rc_l = xl.scaled(-1);
        
        // Define H terms
        Matrix[] Hq = H_q_HKM(soln0.x, soln0.s);  
        Matrix Hl = H_l(soln0.x, soln0.s);
        
        // Define M as in (25.14)
        Matrix Mq = problem.A_q(1).multiply(Hq[0]).multiply((problem.A_q(1)).t());
        for (int i = 2; i<= q; i++){
            Matrix Mqi = problem.A_q(i).multiply(Hq[i - 1]).multiply((problem.A_q(i)).t());
            Mq = Mq.add(Mqi);
        }
                
        Matrix Ml = A_l.multiply(Hl).multiply((A_l).t());       
        
        Matrix M = Mq.add(Ml);
        
            // Check if M is PD // catch exception?
            EigenDecomposition Eig = new EigenDecomposition(M, 1e-16);
            Matrix D_M = Eig.D(); // Eigenvalues of M
            for (int j = 1; j <= D_M.nRows(); j++){
                if (D_M.get(j, j) < 1e-8){
                    System.out.println("Zero or negative Eigenvalue of M is detected.");
                    System.out.println("M = " + M);
                    System.out.println("Its eigenvalues are " + D_M);
                }
            }
        
        // Define h as in (25.15)
        Vector HqRdq = multiplyHbyVector(Hq, Rd_q);
        Vector HlRdl = Hl.multiply(Rd_l);
        
        Vector AqRcq = A_q.multiply(Rc_q);
        Vector AlRcl = A_l.multiply(Rc_l);
        
        Vector AHR_d = (A_q.multiply(HqRdq)).add(A_l.multiply(HlRdl));
        
        // Vector hq = A_q.multiply((HqRdq.minus(Rc_q)));
        // Vector hl = A_l.multiply((HlRdl.minus(Rc_l)));
        
        // Vector AHR_d = hq.add(hl);
        
        Vector h = Rp.add(AHR_d).minus(AqRcq).minus(AlRcl);      
        
        // Solve M * dy = h for dy as in (25.13)
        LowerTriangularMatrix L;
        try {
            Chol chol = new Chol(M, true); 
            // TODO: sparse Choleksy (expected speed-up of two orders) 
            L = chol.L();
            M = null; // free up memory for the large M
        }
        catch (Exception ex) {
            // The algorithm stops when the matrix M is not positive definite.
            // See "Section 3.3 Stopping criteria", page 12-13.
            System.out.println("M is not PD. The Cholesky decomposition is null.");
            return null;
        }
        
        // Define search direction (dx, dy, dz), where x and z consist of "q" and "l" parts.
        Vector dy = dy(L, h); 
        
       // System.out.println("Predictor dy = " + dy);
        
        Vector dz_q = Rd_q.minus((A_q.t().multiply(dy)));
        Vector dz_l = Rd_l.minus((A_l.t().multiply(dy)));
        Vector dz = VectorFactory.concat(dz_q, dz_l);
        
        Vector dx_q = Rc_q.minus((multiplyHbyVector(Hq, dz_q)));
        Vector dx_l = Rc_l.minus((Hl.multiply(dz_l)));
        Vector dx = VectorFactory.concat(dx_q, dx_l);
        
       
        //  Compute step sizes
        double alpha_p = Rescale_gamma(soln0.x, dx);  // for x
        double beta_p = Rescale_gamma(soln0.s, dz);   // for z, y
        
        
        // Define paramters for Corrector Step
        double e = para_e(alpha_p, beta_p, mu);
        // double sigma0 = (soln0.x.add(dx.scaled(alpha_p))).innerProduct((soln0.s.add(dz.scaled(beta_p)))) / n / mu; // n denotes coln size of A_full // Q: why n?
        double sigma0 = (soln0.x.add(dx.scaled(alpha_p))).innerProduct((soln0.s.add(dz.scaled(beta_p)))) / mu;
        double sigma1 = Math.min(1, Math.pow(sigma0, e));
        
        
        //---------------------------------------------------
        /* Below are previous codes [pure conic version]
        
        Vector R_p = problem.b().minus(Ax); // first equation, page 24
        Vector R_d = problem.c().minus(soln0.s).minus(Aty); // third equation, page 24

        // predictor step, Algorithm IPC, page 37
        Vector R_c = soln0.x.scaled(-1); // seventh equation, page 24; sigma=0 and we do not have the v_i^q term

        // solving eq. 13.
        // Since we don't have the A^u term, eq. 13 reduces to M*dy = h.
        Matrix[] H = H(soln0.x, soln0.s); // eq. 12; keep diagonal blocks of H
        // System.out.println("x for computing H is " + soln0.x);
        // System.out.println("s for computing H is " + soln0.s);
        Matrix AH = computeAH(A, H); // multiplication with diagonal blocks for speed
        Matrix M = A.multiply(AH.t()); // eq. 14; M = AHA' where HA' = (AH)' as H is symmetric
        
        Matrix D_M0 = new EigenDecomposition(M, 1e-12).D();
        System.out.println("M = " + M);
        System.out.println("Eigenvalues of M are" + D_M0);
        /*
        // Check if M is PSD
        Matrix D_M = new EigenDecomposition(M, e-12).D();
        //System.out.println("M = " + M);
        //System.out.println("Eigenvalues of M are" + D_M);
       
        for (int j=1; j<= D_M.nRows(); j++){
            if (D_M.get(j,j) < 0){
                System.out.println("Negative eigenvalue of M is detected");
                System.out.println("Eigenvalues of M are" + D_M);
                System.out.println("M = " + M);
            }
        }
        
        LowerTriangularMatrix L;
         try {
//          DenseMatrix denseM = new DenseMatrix(M);
//          Chol chol = new Chol(denseM, true); // third paragraph, "Section 5.5 Solving the Schur complement equation", page 28
            Chol chol = new Chol(M, true); // third paragraph, "Section 5.5 Solving the Schur complement equation", page 28
            L = chol.L();
            M = null; // free up memory for the large M
//            denseM = null;
        }
      
        catch (Exception ex) {
            // The algorithm stops when the matrix M is singular, hence not positive semi-definite.
            // See "Section 3.3 Stopping criteria", page 12-13.
            
            System.out.println("M is not PD. The Cholesky decomposition is null.");
            
            return null;
        }
         
        Vector AHR_d = AH.multiply(R_d);
        AH = null; // free up memory
        Vector AR_c = A.multiply(R_c);
        Vector h = R_p.add(AHR_d).minus(AR_c);
        Vector dy_q = dy(L, h); // eq. 13
        Vector ds_q = R_d.minus(At.multiply(dy_q)); // first equation, page 25
        Vector dx_q = R_c.minus(multiplyHbyVector(H, ds_q)); // first equation, page 25

        
        //  "Section 5.9 Step-length computation"
        double alpha_p = scale(soln0.x, dx_q); // eq. 23
        double beta_p = scale(soln0.s, ds_q); // eq. 24
        double e = e(alpha_p, beta_p, mu);
        // second to last equation, page 37
        double sigma0 = soln0.x.add(dx_q.scaled(alpha_p))
                .innerProduct(soln0.s.add(ds_q.scaled(beta_p))) / N / mu;
        double sigma1 = Math.min(1, Math.pow(sigma0, e));
        
        
        /*
         * Corrector step, Algorithm IPC, page 37.
         * The calculation of the Mehrotra-corrector term is not documented in
         * reference [2]. We use lines 44-64 in "HKMrhsfun.m" in the SDPT3
         * package.
         */
        
        /*
        Vector corrector = Mehrotra(sigma1, mu, soln0.x, soln0.s, dx_q, ds_q);
        Vector Ac = A.multiply(corrector);
        Vector h_c = R_p.add(AHR_d).minus(Ac);
        Vector dy_c = dy(L, h_c);// eq. 13
        Vector ds_c = R_d.minus(At.multiply(dy_c)); // first equation, page 25
        Vector dx_c = corrector.minus(multiplyHbyVector(H, ds_c)); // first equation, page 25

        // Algorithm IPC, page 38
        gamma = gamma(alpha_p, beta_p);
        double alpha = scale(soln0.x, dx_c);
        double beta = scale(soln0.s, ds_c);
        gamma = gamma(alpha, beta);

        stepSize = Math.min(alpha, beta);

        // update x, y, s, Algorithm IPC, page 38
        Vector x = soln0.x.add(dx_c.scaled(alpha));
        Vector s = soln0.s.add(ds_c.scaled(beta));
        Vector y = soln0.y.add(dy_c.scaled(beta));
        */
        
        // Above are previous version codes
        //------------------------------------------------
        
        
        // Corrector Step in Algorithm IPC, Appendix
        // The implementation follows from HKMrhsfun.m in SDPT3 version 4.0 MATLAB source code.
        
        /** Step 1: Update h to h_corr.
         *  
         *  h_corr = Rp + A^q * (H^q(R_d^q) - R_c^q[sigma = 0]) + A^l * (H^l(R_d^l) - R_c^l[sigma = 0])
         *  h_corr = Rp + A^q * (H^q(R_d^q) - \hat{R_c^q}) + A^l * (H^l(R_d^l) - \hat(R_c^l))
         *  where \hat{R_c^q} = R_c^q[new sigma] - CorrectedTerm-q;
         *        \hat{R_c^l} = R_c^l[new sigma] - CorrectedTerm-l;
         *  
         *  Therefore, h_corr = h + A^q * (R_c^q[sigma = 0] - R_c^q[new sigma] + CorrectedTerm-q)
         *                        + A^l * (R_c^l[sigma = 0] - R_c^l[new sigma] + CorrectedTerm-l)
         */
        
       
        Vector Rcq_hat = Rcq_hat_HKM(sigma1, mu, soln0.x, soln0.s, dx, dz);
        Vector Rcl_hat = Rcl_hat(sigma1, mu, soln0.x, soln0.s, dx, dz);
        
        //Vector termq_corr = A_q.multiply(Rc_q.minus(Rcq_hat));
        //Vector terml_corr = A_l.multiply(Rc_l.minus(Rcl_hat));
        
        Vector termq_corr = A_q.multiply(Rcq_hat);
        Vector terml_corr = A_l.multiply(Rcl_hat);
        
        Vector h_corr = Rp.add(AHR_d).minus(termq_corr).minus(terml_corr);    
               
            
        /** Step 2: Obtain corrected search direction (dx_c, dy_c, dz_c).
         */
        
        Vector dy_c = dy(L, h_corr); // Because M is the same
        //System.out.println("Corrector dy = " + dy_c);
        
        Vector dzq_c = Rd_q.minus(A_q.t().multiply(dy_c));
        Vector dzl_c = Rd_l.minus(A_l.t().multiply(dy_c));
        Vector dz_c = VectorFactory.concat(dzq_c, dzl_c);
        
        Vector dxq_c = Rcq_hat.minus(multiplyHbyVector(Hq, dzq_c)); // should use corrected Rc_q
        Vector dxl_c = Rcl_hat.minus(Hl.multiply(dzl_c));
        // Vector dxq_c = Rc_q.minus(multiplyHbyVector(Hq, dzq_c)); // should use corrected Rc_q
        // Vector dxl_c = Rc_l.minus(Hl.multiply(dzl_c));
        Vector dx_c = VectorFactory.concat(dxq_c, dxl_c);
        
        
        // Update step length
        gamma = newgamma(alpha_p, beta_p);  // \bar{gamma} 
        double alpha = Rescale_gamma(soln0.x, dx_c);
        double beta = Rescale_gamma(soln0.s, dz_c);
        
        gamma = newgamma(alpha, beta); // \bar{gamma}^{+}

        stepSize = Math.min(alpha, beta);
        System.out.println("StepSize = " + stepSize);

        // update current iterate (x, y, z) to next (x^+, y^+，z^+) 
        Vector x = soln0.x.add(dx_c.scaled(alpha));
        Vector z = soln0.s.add(dz_c.scaled(beta));
        Vector y = soln0.y.add(dy_c.scaled(beta));
        
        
        /** Heuristic step if unrestricted variables (x_u）is involved.
         * 
         * In order to ameliorate the ill-conditioned problem that might arise, 
         * we will modify x_u^{+}, x_u^{-}, z_u^{+}, z_u^{-} in each iteration
         * using the following heuristics.
         * 
         * x_u^{+} = x_u^{+} - 0.8 * min(x_u^{+}, x_u^{-});
         * x_u^{-} = x_u^{-} - 0.8 * min(x_u^{+}, x_u^{-});
         * the update rule of z_u^{+} and z_u^{-} are more complicated
         * following the MATLAB code: Line 569-594, sqlpmain.m, version 4.0. 
         * 
         */
        
        if (flag_u){
            
            System.out.println();
            System.out.println("Before heuristic step, x = " + x);
            System.out.println("Before heuristic step, z = " + z);
            
            // The implementation follows from Line 568 - 595, sqlpmain.m in Matlab Source Code.
            
            int nu = problem.n_u();
            
            Vector xup = new SubVectorRef(x, n - nu - nu + 1, n - nu); // x_u^{+}
            Vector xun = new SubVectorRef(x, n - nu + 1, n);           // x_u^{-}
            Vector zup = new SubVectorRef(z, n - nu - nu + 1, n - nu); // z_u^{+}
            Vector zun = new SubVectorRef(z, n - nu + 1, n);           // z_u^{-}
            // Note: cannot set xup directly, as SubVectorRef is read-only     
            
            // Heuristic step for x 
            for(int i = 1; i <= nu; i++){           
                double x_temp = 0.8 * Math.min(xup.get(i), xun.get(i));              
                x.set(n - nu - nu + i, xup.get(i) - x_temp);
                x.set(n - nu + i,      xun.get(i) - x_temp);     
            }   
            System.out.println("After heuristic step, x = " + x);        
            // Heuristic step for z
            if (mu < 1e-4){
                for (int i = 1; i <= 2 * nu; i++){
                   z.set(n - nu - nu + i, 0.5 * mu / Math.max(1, x.get(i)));  // Line 580 Be careful with index
                }
            }
            else{
                for (int i = 1; i <= nu; i++){        
                    double z_temp = Math.min(1, Math.max(zup.get(i), zun.get(i)));
                    double z_rescale = 0; // important to set this scale factor to be 0 at later stage                
                    if (dinfeas > 1e-4 && stepSize < 0.2){ 
                        z_rescale = 0.3;
                    }                  
                    z.set(n - nu - nu + i, zup.get(i) + z_rescale * z_temp);
                    z.set(n - nu + i,      zun.get(i) + z_rescale * z_temp);                
                }           
            }
            
            // TODO: can combine for-loops
            System.out.println("After heuristic step, z = " + z);
            
        }
        
        System.out.println();
        System.out.println("y = " + y);    
        
        PrimalDualSolution soln1 = new PrimalDualSolution(x, z, y);
        return soln1;
    }

    
    // Define functions
    
    /** Compute H^q as in (25.12), Section 25.5.1 HKM Direction.
     * H^q = [H_i^q]_{i\in[q]}
     * H_i^q = - (x_i^q)^T * z_i^q / gamma(z_i^q)^2 * J_i^q + x_i^q * ((z_i^q)^(-1))^T + (z_i^q)^(-1) * (x_i^q)^T
     * 
     * @param xq
     * @param zq
     * @return H^q
     */
    
    private Matrix[] H_q_HKM(Vector x, Vector z){
        Matrix[] H_q_HKM = new Matrix[q];
        
        Vector xq_i;
        Vector zq_i;
        int act = 0;
                    
        for (int i = 1; i <= q; i++){
            final int ni = problem.n(i);
            xq_i = new SubVectorRef(x, act + 1, act + ni); // Define x_i^q
            zq_i = new SubVectorRef(z, act + 1, act + ni); // Define z_i^q
            
            // term 1 coefficient (to avoid defining J matrix)
            double term1coeff = xq_i.innerProduct(zq_i) / gammasq(zq_i); 
            
            // term 2: x_i^q * ((z_i^q)^(-1))^T
            Vector z_inv = z_inv(zq_i);
            Matrix term2 = new OuterProduct(xq_i, z_inv);
            
            // term 3: (z_i^q)^(-1) * (x_i^q)^T
            Matrix term3 = term2.t();
            
            // Combined
            Matrix Hi = term2.add(term3);
            Hi.set(1, 1, Hi.get(1, 1) - term1coeff);
            for (int k = 2; k <= ni; k++){
                Hi.set(k, k, Hi.get(k, k) + term1coeff);
            }
             
            H_q_HKM[i - 1] = Hi;
            
            act += ni;
            
                // Check if Hi is still PD // TODO: catch exception? 
                EigenDecomposition Eig = new EigenDecomposition(Hi, 1e-16);
                Matrix D_Hi = Eig.D(); // Eigenvalues of Hi
                for (int j = 1; j <= ni; j++){
                    if (D_Hi.get(j, j) < 1e-8){
                        System.out.println("Zero or negative Eigenvalue of Hi is detected.");
                        System.out.println("Current block index is " + i);
                        System.out.println("The ith block of H is " + Hi);
                        System.out.println("Its eigenvalues are " + D_Hi);
                    }
                }
        
        }
        return H_q_HKM;
    
    }
    
    /* TODO: NT direction
    
    private Matrix[] H_q_NT(Vector x, Vector z){
    } 
    */
    
     /** Compute H^l as in (25.12), Section 25.5.1.
     * H^l = Diag(xl) * Diag(zl)^(-1)
     *  
     * @param xl
     * @param zl
     * @return H^l
     */
    
    private Matrix H_l(Vector x, Vector z){
        
        // Obtain xl, zl first
        int nl = A_l.nCols();
        Vector xl = new SubVectorRef(x, n - nl + 1, n);
        Vector zl = new SubVectorRef(z, n - nl + 1, n);
        
        Vector xzinv = new DenseVector(nl);
        
        for (int i = 1; i <= nl; i++){
            xzinv.set(i, xl.get(i)/zl.get(i));
            if (zl.get(i) == 0) {
                System.out.println("zl" + i + "= 0"); // TODO: what to do if zi = 0?
            }
        }
        // TODO: pointwise inverse of vector like 1./A in matlab?
        
        DiagonalMatrix H_l = new DiagonalMatrix(xzinv.toArray());
                      
        return H_l;
    }
    
        /** Computes Hq(x).
     * Hq(x) is a long vector consisting of q blocks, each of which defined by
     * (25.12) in Section 25.5.1.
     * 
     * @param Hq (array)
     * @param x (vector)
     * @return Hq(x)
     */
    
      private Vector multiplyHbyVector(final Matrix[] H, final Vector x) {
        Vector[] y = new Vector[q];
        
        IntStream.rangeClosed(1, q).parallel().forEach(i -> {
            int ni = problem.n(i);
            int index = indices[i - 1];
            Vector Hixi = H[i - 1].multiply(new SubVectorRef(x, index, index + ni - 1));
            y[i - 1] = Hixi;
        });

        return VectorFactory.concat(y);
    }
    
     /**
    * Define gamma^2(u) = u1^2 - u(2:end)^T u(2:end).
    * Define u^(-1) = J_i^q * u / gamma^2(u), where J is negative identity matrix except (1,1)-entry = 1.
    * 
    * @param u
    * @return r(u)^2, u^(-1)
    */
    
    // Define functions
    
    private double gammasq(Vector u){
        double u_1 = u.get(1);
        Vector u_bar = new SubVectorRef(u, 2, u.size());
        double gammasq = u_1 * u_1 - u_bar.innerProduct(u_bar);
        if (gammasq == 0){
            System.out.println("Gamma square of " + u + "is 0.");
        }
        return gammasq;
    }
    
    private Vector z_inv(Vector u){
        Vector z_inv = new DenseVector(u.size()); // TODO: double check "Dense"   
        // TODO: any more efficient way to do this? like stacking a scalar with a vector? or replaceinplace for Vector?
        z_inv.set(1, u.get(1));
        for (int j = 2; j <= u.size(); j++){ 
            z_inv.set(j, - u.get(j));
        }   
        z_inv = z_inv.scaled(1./gammasq(u));
        
        return z_inv;
    }
            
    
    // For solving linear system 
    private Vector dy(LowerTriangularMatrix L, Vector h) {
        ForwardSubstitution fsub = new ForwardSubstitution();
        Vector Ltdy = fsub.solve(L, h);
        BackwardSubstitution bsub = new BackwardSubstitution();
        Vector dy = bsub.solve(L.t(), Ltdy);
        return dy;
    }
     
    // For corrector step sigma
    private double para_e(double alpha_p, double beta_p, double mu) {
        if (mu <= 1e-6) {
            return 1;
        }
        double min_ab = Math.min(alpha_p, beta_p);
        double e = Math.max(1, 3. * min_ab * min_ab);
        return e;
    }
     
    // For corrector terms 
    
    /** This class returns \hat{R_c^q} as in Corrector Step, Algorithm IPC.
     *  Rcq_hat = Rcq[new sigma] - CorrectedTerm_q
     * 
     *  We define it blockwisely.
     *  Note that this corrector term suits for HKM search direction.
     * 
     * @param sigma
     * @param mu
     * @param x
     * @param z
     * @param dx
     * @param dz
     * @return 
     */
    
    private Vector Rcq_hat_HKM(double sigma, double mu, Vector x, Vector z, Vector dx, Vector dz){
           
        Vector[] Rcq_hat = new Vector[q];
        int act = 0;
        
        //TODO: can do parallel computing? 
        for (int i = 1; i <= q; i++){ 
            
            int ni = problem.n(i);
            Vector x_p = new SubVectorRef(x, act + 1, act + ni); // extract x[i] corresponding to i-th conic block
            Vector z_p = new SubVectorRef(z, act + 1, act + ni);
            Vector dx_p = new SubVectorRef(dx, act + 1, act + ni);
            Vector dz_p = new SubVectorRef(dz, act + 1, act + ni);
            
            // Define R_c^q with new sigma = sigmamusiInv - xp
            Vector sigmamusiInv = z_inv(z_p).scaled(sigma * mu);
            
            Vector Rcq_newsigma = sigmamusiInv.minus(x_p);
            //Vector Rcq_newsigma = z_inv(z_p).scaled(sigma * mu).minus(x_p);
            
            // Define corrected term
            // Corrected term = rqp = hdzpM * hdxdz as in Page 2, "The calculation of Mehrotra-corrector term" document.
            
            double gamma2 = gammasq(z_p);
            double gamma1 = Math.sqrt(gamma2);
            Vector ffp = z_p.scaled(1./gamma1);
            double inProd_ffdx = ffp.innerProduct(dx_p);
            
            // Define hdxp (or hsxp in doc)
            Vector hdxp = new DenseVector(ni);  // TODO: what if hdxp is not dense? 
            hdxp.set(1, inProd_ffdx * gamma1);
            double coef_temp = (inProd_ffdx + dx_p.get(1)) / (1 + ffp.get(1));
            for (int j = 2; j <= ni; j++){
                hdxp.set(j, (dx_p.get(j) + ffp.get(j) * coef_temp) * gamma1);
            }       
            // TODO: can we have "ReplaceinPlace" for vector to avoid for loop?
            
//            Vector hdxp_temp1 = new SubVectorRef(dx_p, 2, ni); 
//            Vector hdxp_temp2 = new SubVectorRef(ffp, 2, ni);
//            double hdxp_temp3 = (inProd_ffdx + dx_p.get(1)) / (1 + ffp.get(1));
//            Vector hdxp_r = (hdxp_temp2.scaled(hdxp_temp3).add(hdxp_temp1)).scaled(gamma1);
            
            // Define hdzpM
            Vector ffpb = new SubVectorRef(ffp, 2, ni);
            Matrix ffpffpt = new OuterProduct(ffpb, ffpb).scaled(1./(1 + ffp.get(1)));
            
            Matrix hdzpM = new DenseMatrix(ni, ni); // TODO: what if not dense?
            
            hdzpM.set(1, 1, ffp.get(1));
            // MatrixFactory.replaceInPlace(hdzpM, 1, 1, 2, ni, ffpb); // TODO: fill vector in "replaceInplace"
            MatrixFactory.replaceInPlace(hdzpM, 2, ni, 2, ni, ffpffpt);
            
            for (int j = 2; j <= ni; j++){
                hdzpM.set(1, j, - ffpb.get(j - 1));
                hdzpM.set(j, 1, - ffpb.get(j - 1));
                hdzpM.set(j, j, hdzpM.get(j, j) + 1);
            }
           
            hdzpM = hdzpM.scaled(1. / gamma1);
            
            // Define hdzp
            Vector hdzp = hdzpM.multiply(dz_p);
            
            // Define hdxdz --- Arrow function
            Vector hdxdz = new DenseVector(ni); // TODO: what if not dense?
            hdxdz.set(1, hdxp.innerProduct(hdzp));
            for (int j = 2; j <= ni; j++){
                hdxdz.set(j, hdxp.get(1) * hdzp.get(j) + hdzp.get(1) * hdxp.get(j));
            }
            
            // Define rqp
            Vector rqp = hdzpM.multiply(hdxdz);
            
            // Combined
            Rcq_hat[i - 1] = Rcq_newsigma.minus(rqp); // 
                       
            act += ni;        
        }
        
        return VectorFactory.concat(Rcq_hat);
        
        }
    
        // TODO: Corrector for NT direction
    
        /* 
        private Matrix[] Rcq_hat_NT(double sigma, double mu, Vector x, Vector z, Vector dx, Vector dz){
        } 
        */
        
    
        private Vector Rcl_hat(double sigma, double mu, Vector x, Vector z, Vector dx, Vector dz){
        // This class returns \hat{R_c^l} as in Corrector Step, Algorithm IPC.
        // Rcl_hat = Rcl[new sigma] - CorrectedTerm_l
        
        // Note that this corrector term might suit for both HKM and NT search directions.
        // TODO: double check
            
        int nl = A_l.nCols();
        
        Vector Rcl_hat = new DenseVector(nl);
       
        Vector xl = new SubVectorRef(x, n - nl + 1, n);
        Vector zl = new SubVectorRef(z, n - nl + 1, n);
        Vector dxl = new SubVectorRef(dx, n - nl + 1, n);
        Vector dzl = new SubVectorRef(dz, n - nl + 1, n);
        
        for (int i = 1; i <= nl; i++) {
            Rcl_hat.set(i, sigma * mu / zl.get(i) - xl.get(i) - dxl.get(i) * dzl.get(i) / zl.get(i));        
        }
        
        return Rcl_hat;
        
        }
    
    // for iterate update
        
    private double newgamma(double a, double b) {
        double g = 0.9 + 0.09 * Math.min(a, b);
        return g;
    }

    private double Rescale_gamma(Vector x, Vector dx) {
        double inc = increment(x, dx);
        double s = Math.min(1, gamma * inc);
        return s;
    }
    //----------------------------------------------
    // Below are previous version of predictor step
    
    /**
     * Calculates H in eq. 12
     */
    /*
    private Matrix[] H(Vector x, Vector z) {
        Matrix[] H = new Matrix[problem.q()]; // diagonal blocks of H
        int act = 0; // the current working index to x_i[0] (the first entry in x_i) in x = [x_1, x_2, ... x_q]; this is the notation used in AntoniouLu2007
        for (int i = 1; i <= problem.q(); i++) {
            final int ni = problem.n(i);
            Vector xi = new SubVectorRef(x, act + 1, act + ni);
            Vector zi = new SubVectorRef(z, act + 1, act + ni);
            double gamma_z_2_inv = gamma2_inv(zi);
            // calculates the inverse of s, first equation above eq. 12
            Vector zi_inv = new DenseVector(ni); // z_i^{-1} = Jz/gamma(z)^2; J defined in eq. 10 whose entries are either 1 or -1 on the diagonal
            zi_inv.set(1, zi.get(1) * gamma_z_2_inv); // J entry = 1 (on [1,1])
            for (int j = 2; j <= ni; j++) {
                zi_inv.set(j, -zi.get(j) * gamma_z_2_inv); // J entry = -1 
            }

            // calculates the first term in H, eq. 12
            double xsjScale = xi.innerProduct(zi) * gamma_z_2_inv;
//            Matrix term1 = new DenseMatrix(ni, ni).ONE().scaled(xsjScale); // multiply by -J
//            term1.set(1, 1, -xsjScale); // multiply by J

            // calculates the second term in H, eq. 12
            Matrix term2 = new OuterProduct(xi, zi_inv);

            // calculates the third term in H, eq. 12
            Matrix term3 = term2.t(); // new OuterProduct(zi_inv, xi);

            Matrix Hi = term2.add(term3); // TODO: (A + A') is a symmetric matrix
            Hi.set(1, 1, Hi.get(1, 1) - xsjScale); // add term1 efficiently, just the diagonal
            for (int j = 2; j <= ni; ++j) {
                Hi.set(j, j, Hi.get(j, j) + xsjScale);
            }

            H[i - 1] = Hi;
            
            // Check if Hi is still PD
            
            EigenDecomposition Eig = new EigenDecomposition(Hi, 1e-16);
            Matrix D_Hi = Eig.D(); // Eigenvalues of Hi
            for (int j = 1; j <= ni; j++){
                if (D_Hi.get(j, j) < 1e-6){
                    System.out.println("Negative Eigenvalue of Hi is detected.");
                    System.out.println("Current block index is " + i);
                    System.out.println("The ith block is " + Hi);
                    System.out.println("Its eigenvalues are " + D_Hi);
                    }
                }
            
            act += ni;
        }

        return H;
    }
   
     private Vector multiplyHbyVector(final Matrix[] H, final Vector x) {
        Vector y = new DenseVector(N);
        IntStream.rangeClosed(1, q).parallel().forEach(i -> {
            int ni = problem.n(i);
            int index = indices[i - 1];
            Vector Hixi = H[i - 1].multiply(new SubVectorRef(x, index, index + ni - 1));
            for (int k = 1; k <= ni; ++k) {
                y.set(index + k - 1, Hixi.get(k));
            }
        });
        return y;
    }
    
    private Matrix computeAH(final Matrix A, final Matrix[] H) {
        int m = A.nRows();
        Matrix AH = new DenseMatrix(A.nRows(), A.nCols());
        IntStream.rangeClosed(1, q).parallel().forEach(i -> {
            int ni = problem.n(i);
            int index = indices[i - 1];
            Matrix AiHi = problem.A(i).multiply(H[i - 1]);
            MatrixFactory.replaceInPlace(AH, 1, m, index, index + ni - 1, AiHi);
        });
        return AH;
    }
    */
    
    /*
    // defined toward the bottom on p.2
    private double gamma2_inv(Vector u) { // gamma squared
        double u_1 = u.get(1);
        Vector u_bar = new SubVectorRef(u, 2, u.size());
        double gamma2 = u_1 * u_1 - u_bar.innerProduct(u_bar);
        double inv = 0;
        if (gamma2 != 0) {
            inv = 1. / gamma2;
        }
        return inv;
    }
    
    /**
     * Calculates <i>dy</i>. <i>M*dy = h</i>. Since <i>M</i> is a positive
     * definite matrix, we can avoid calculating the inverse of M. We
     * compute the Cholesky decomposition of <i>M</i> such that <i>LL'=M</i>,
     * where <i>L</i> is a lower triangular matrix. Then we use forward
     * substitution and backward substitution to calculate <i>dy</i>.
     *
     * @see "third paragraph, "Section 5.5 Solving the Schur complement
     * equation", page 28."
     */
     
     /*
    private Vector dy(LowerTriangularMatrix L, Vector h) {
        ForwardSubstitution fsub = new ForwardSubstitution();
        Vector Ltdy = fsub.solve(L, h);
        BackwardSubstitution bsub = new BackwardSubstitution();
        Vector dy = bsub.solve(L.t(), Ltdy);
        return dy;
    }
   
    // last equation, page 37
    private double e(double alpha_p, double beta_p, double mu) {
        if (mu < 1e-6) {
            return 1;
        }
        double min_ab = Math.min(alpha_p, beta_p);
        double e = Math.max(1, 3. * min_ab * min_ab);
        return e;
    }
    
    
    

    private double gamma(double a, double b) {
        double g = 0.9 + 0.09 * Math.min(a, b);
        return g;
    }

    private double scale(Vector x, Vector dx) {
        double inc = increment(x, dx);
        double s = Math.min(1, gamma * inc);
        return s;
    }

    */
    
    /**
     * Calculates R_c minus the Mehrotra-corrector term generated from dx_p,
     * ds_p, second to last line, page 37. The calculation of Mehrotra-corrector
     * term is not documented in reference [2]. We use instead lines 44-64,
     * "HKMrhsfun.m" in SDPT3 package is used as the reference.
     */
    
    /*
    private Vector Mehrotra(double sigma, double mu, Vector x, Vector z, Vector dx, Vector dz) {
        int act = 0; // index to xi[0] in x = [x1, x2, ... xq]
        Vector Rc = new DenseVector(N);

        for (int p = 1; p <= problem.q(); p++) {
            final int n_p = problem.n(p);
            Vector x_p = new SubVectorRef(x, act + 1, act + n_p);
            Vector z_p = new SubVectorRef(z, act + 1, act + n_p);
            Vector dx_p = new SubVectorRef(dx, act + 1, act + n_p);
            Vector dz_p = new SubVectorRef(dz, act + 1, act + n_p);

            double gamma2 = gamma2_inv(z_p);
            double gamz = Math.sqrt(1. / gamma2);
            // line 52, HKMrhsfun.m, SDPT3 package
            Vector ff_p = z_p.scaled(1. / gamz);

            // line 53, HKMrhsfun.m, SDPT3 package
            Vector hdx = qops_5(gamz, ff_p, dx_p);

            // calculates hdz, line 54, HKMrhsfun.m, SDPT3 package, and line 35-41, qops.m, SDPT3 package
            Vector hdz = qops_6(gamz, ff_p, dz_p);

            // line 55, HKMrhsfun.m, SDPT3 package
            Vector hdxdz = arrow(hdx, hdz);

            // line 56, HKMrhsfun.m, SDPT3 package
            Vector Rq = qops_6(gamz, ff_p, hdxdz);

            // sigmamusiInv equals sigma*mu*z_i_inv, eq. 11
            Vector sigma_mu_z_i_inv = new DenseVector(n_p);
            sigma_mu_z_i_inv.set(1, z_p.get(1) * sigma * mu * gamma2);
            for (int j = 2; j <= n_p; j++) {
                sigma_mu_z_i_inv.set(j, -z_p.get(j) * sigma * mu * gamma2);
            }
            // Rc - Rq = sigma*mu*z_i_inv - xi - Rq, where Rq is the Mehrotra-corrector term, eq. 11.
            Vector Rc_p = sigma_mu_z_i_inv.minus(x_p).minus(Rq);

            for (int j = 1; j <= n_p; j++) {
                Rc.set(j + act, Rc_p.get(j));
            }

            act += n_p;
        }

        return Rc;
    }

    private Vector qops_6(double w, Vector f, Vector u) {
        final int n = f.size();

        DenseMatrix Finv = new DenseMatrix(n, n);
        Finv.set(1, 1, f.get(1));
        for (int b = 2; b <= n; b++) {
            Finv.set(1, b, -f.get(b));
            Finv.set(b, 1, -f.get(b));
        }

        Vector fb = new SubVectorRef(f, 2, n);
        Vector fb_scaled = fb.scaled(Math.sqrt(1. / (1. + f.get(1))));
        Matrix fbfb = new OuterProduct(fb_scaled, fb_scaled);
        MatrixFactory.replaceInPlace(Finv, 2, n, 2, n, fbfb.add(fbfb.ONE()));
        Finv = Finv.scaled(1 / w);

        Vector hdz = Finv.multiply(u);

        return hdz;
    }

    private Vector qops_5(double w, Vector f, Vector u) {
        final int n = f.size();

        Vector Fu = new DenseVector(n); // line 53, HKMrhsfun.m, SDPT3 package
        double fu = f.innerProduct(u);
        Fu.set(1, fu);

        double alp = (fu + u.get(1)) / (1. + f.get(1));
        for (int b = 2; b <= n; b++) {
            Fu.set(b, u.get(b) + f.get(b) * alp);
        }

        Fu = Fu.scaled(w);

        return Fu;
    }

    */
    
    /*
     * Calculates Fx, where Fx(1) = <hdx,hdz> and
     * Fx(i)=hdx(1)*hdz(i)+hdx(i)*hdz(1), i=2,...,n.
     * See Arrow.m, SDPT3 package.
     */
    
    /*
    private Vector arrow(Vector hdx, Vector hdz) {
        Vector Fx = new DenseVector(hdx.size());
        Fx.set(1, hdx.innerProduct(hdz));
        for (int i = 2; i <= hdx.size(); i++) {
            Fx.set(i, hdx.get(1) * hdz.get(i) + hdz.get(1) * hdx.get(i));
        }
        return Fx;
    }
    
    */
    // Above are previous codes
    // ---------------------------------
    
    
}
