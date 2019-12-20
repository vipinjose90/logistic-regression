/**
 *
 * @author Vipin Jose
 */
import java.io.*;
import java.util.*;

public class LogisticRegression {
    
    public static void main(String[] args) throws FileNotFoundException { 
        
        RunLogisticRegression rlr=new RunLogisticRegression();
        
        RunLogisticRegressionX2 rlrx2=new RunLogisticRegressionX2();
        
/* 4.4.a) Runs cross validation on parameters   */
        rlr.startCrossValidation();
        
/* 4.4.b) Runs Logistic Regresiion with best parameters   */
        rlrx2.startLogisticRegression();
        

    }
}
    
