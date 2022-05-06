/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nm_socp;

import dev.nm.algebra.linear.matrix.doubles.Matrix;
import dev.nm.algebra.linear.matrix.doubles.matrixtype.dense.DenseMatrix;
import dev.nm.algebra.linear.matrix.doubles.operation.Inverse;
import dev.nm.misc.license.License;


/**
 *
 * @author karanallagh
 */
public class NM_SOCP {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        License.setLicenseFile(new java.io.File("/Users/karanallagh/Desktop/NM_DEV/NM_SOCP/nm.lic"));
       // System.out.println(“Hello NM Dev”);
        System.out.println("Hello NM");
        Matrix A1 = new DenseMatrix(new double[][]{
        {1, 2, 1},
        {4, 5, 2},
        {7, 8, 1}
    });
    
    System.out.println(A1);
    
    Matrix B = new Inverse(A1);//compute the inverse of A1
    Matrix I = A1.multiply(B);//this should be the identity matrix
    System.out.println(String.format("%s * %s = %s (the identity matrix)",A1,B,I));


    }
    
}
/