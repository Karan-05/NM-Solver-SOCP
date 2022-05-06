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
//import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPDualProblem1;
import dev.nm.algebra.linear.vector.doubles.SubVectorRef;
import dev.nm.algebra.linear.vector.doubles.Vector;
import dev.nm.number.doublearray.DoubleArrayMath;
//import dev.nm.solver.multivariate.constrained.convex.sdp.socp.problem.SOCPDualProblem;
import static java.lang.Math.sqrt;

/**
 * This is an interface to the implementation of an SOCP algorithm.
 *
 * @author Haksun Li
 */
abstract class PrimalDualInteriorPointIterationStep1 {

    double stepSize = 1; // For SDPT3v4 use
    final SOCPDualProblem1 problem;
    
    // final Matrix A;
    // final Matrix At;
    // final int N;
    
    // double r = 1; // step-length parameter  --- Gamma in SDPT3v4
    
    final int q;   // no. conic blocks
    
    final int nq;  // column size of A_q_full
    final int n;   // column size of A_full 

    public abstract PrimalDualSolution iterate(PrimalDualSolution soln0, Vector Ax, Vector Aty, double mu);

    PrimalDualInteriorPointIterationStep1(SOCPDualProblem1 problem) {
        this.problem = problem;
        // this.A = problem.A();
        // this.At = this.A.t();
        // this.N = this.A.nCols();
        
        this.q = problem.n_q(); 
        this.nq = problem.A_q_full().nCols();
        this.n = problem.A_full().nCols();
    }

    
    double stepSize() {
        return stepSize;
    }
    
    
    /**
     * eq. 14.126: find_alpha.m
     * Section "5.9 Step-length computation", reference [2].
     */
    
    /** The step-length computation follows Section 25.5.9; or Section 5.9 of the 17 July 2006 arxiv version
     * 
     * @param x
     * @param dx
     * @return maximal step size alpha allowed
     */
    
    double increment(Vector x, Vector dx) {
        
        double min;
        
        // compute alpha_i^q
        double[] alpha_q = new double[q]; // TODO: can we define a vector and use max(vector)?
        
        for (int i = 0, act = 0; i < q; i++){
            
            final int ni = problem.n(i + 1);
            
            double xq1 = x.get(act + 1); // first entry of x_i^q
            Vector xqr = new SubVectorRef(x, act + 2, act + ni); // x_i^q(2:end)
            
            double dxq1 = dx.get(act + 1);
            Vector dxqr = new SubVectorRef(dx, act + 2, act + ni);
            
            double ai = dxq1 * dxq1 - dxqr.innerProduct(dxqr);
            double bi = dxq1 * xq1 - dxqr.innerProduct(xqr);
            double ci = xq1 * xq1 - xqr.innerProduct(xqr);
            double di = bi * bi - ai * ci;
            
            double aqi;
            
            if ((ai < 0) || (bi < 0) && (ai <= bi * bi / ci)){
                aqi = ((-1) * bi - sqrt(di)) / ai;
            }
            else if((ai == 0) && (bi < 0)){
                aqi = (-0.5) * ci / bi;
            }
            else{
                aqi = 1e12; // Double.POSITIVE_INFINITY;  
            }
            
            alpha_q[i] = aqi;
            act += ni;                  
        }
        
        double min_q = DoubleArrayMath.min(alpha_q);
        
        
        // Compute alpha_k^l if applicable
        if (n != nq) {// There exists linear blocks
            
            final int nl = n - nq;
            double[] alpha_l = new double[nl];
            
            double min_l; 
            double alk;
            
            for(int k = 1; k <= nl; k++){
                if (dx.get(k + nq) < 0){
                    alk = - x.get(k + nq) / dx.get(k + nq); // l block
                } 
                else{
                    alk = 1e12; // follow MATLAB code //Double.POSITIVE_INFINITY;
                }
                alpha_l[k - 1] = alk;
            }
            
            min_l = DoubleArrayMath.min(alpha_l);
            
            min = DoubleArrayMath.min(min_q, min_l);
        }
        else{
            min = min_q;
        }
        
               
        /*
        double[] alpha = new double[problem.q()];

        for (int i = 0, act = 0; i < problem.q(); i++) {
            final int ni = problem.n(i + 1);

            double x1 = x.get(act + 1);
            Vector xr = new SubVectorRef(x, act + 2, act + ni);

            double dx1 = dx.get(act + 1);
            Vector dxr = new SubVectorRef(dx, act + 2, act + ni);

            // the first equation below eq. 22, page 31, reference [2]
            double ai = dx1 * dx1 - dxr.innerProduct(dxr);
            double bi = x1 * dx1 - xr.innerProduct(dxr);
            double ci = x1 * x1 - xr.innerProduct(xr);
            double di = bi * bi - ai * ci;
            double aiq = 1; // ?

            // the last equation on page 31, reference [2]
            if ((ai < 0 || (bi < 0)) && ai <= bi * bi / ci) {
                aiq = (-1 * bi - sqrt(di)) / ai;
            } else if (ai == 0 && bi < 0) {
                aiq = -1 * ci / 2 * bi;
            } else {
                aiq = Double.POSITIVE_INFINITY;
            }
            alpha[i] = aiq;
            act += ni;
        }

        double min = DoubleArrayMath.min(alpha);
        
        */
        
        return min;
        
        //TODO: double check the value of r (step lenth parameter)
    }
}
