
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;

class RunLogisticRegressionX2{
    
    //*******************************LogisticRegression SETTINGS************START************//
    //Mention the Training File without extension
    public static String trainFile="data/a5a.train";    
    //Mention the Test File here without extension
    public static String testFile="data/a5a.test"; 
    //Mention the starting value of rate(Yo)
    public static float rate=(float) 0.01;
    //Mention the value of C
    public static double C=50;
    //Mention the value of "number of epochs"
    public static int epochs= 20;
    //*******************************LogisticRegression SETTINGS*************END***********//
    
    
    public static float accuracy, accuracyTrain;
    public static float sd,sdTrain;
    public static int mistakesTrain;
    public static float avgMistakes;    

    int trainRow=0, trainCol=0;
    int testRow=0, testCol=0;
    int depth=0;
    boolean firstIter=true;
    String[][] trainTemp;
    String [][] testTemp;
    int[][] train;
    int[][] test;
    int[] trainLabels;
    int[] testLabels;
    int weightSize=0;
    double weights[];
    float bias=0;
    int mistakeCounter=0;


    RunLogisticRegressionX2() throws FileNotFoundException {  
        
        Scanner trainIn = new Scanner (new File (trainFile));
        Scanner testIn = new Scanner (new File (testFile));
        
        int v=0;

        while (trainIn.hasNextLine()){
            trainIn.nextLine();
            trainRow++;
        }
        trainTemp = new String[trainRow][];
        Scanner trainIn1 = new Scanner (new File (trainFile));        
        while (trainIn1.hasNextLine()){
            trainTemp[v++]=trainIn1.nextLine().split("\\s+");
        }
        
        v=0;
        while (testIn.hasNextLine()){
            testIn.nextLine();
            testRow++;
        }
        testTemp = new String[testRow][];
        Scanner testIn1 = new Scanner (new File (testFile));        
        while (testIn1.hasNextLine()){
            testTemp[v++]=testIn1.nextLine().split("\\s+");
        }
    }
    
    void startLogisticRegression(){
        preProcess();
        trainLogisticRegression();
        predictLogisticRegressiononTrain();
        predictLogisticRegressiononTest();
    }
    
    void preProcess(){
        int max=0;
        int val=0;
        String [] cellVal=new String[2];
        
        for(int i=0;i<trainRow;i++){
            for(int j=1;j<trainTemp[i].length;j++){
                cellVal=trainTemp[i][j].split(":");
                if((val=(Integer.parseInt(cellVal[0])))>max){
                    max=val;
                }
            }
        }
                
        trainCol=max+2;
        train=new int[trainRow][trainCol];
        
        for(int i=0;i<trainRow;i++){
            for(int j=0;j<trainCol;j++){
                train[i][j]=0;
            } 
        }
        
        for(int i=0;i<trainRow;i++){
            for(int j=0;j<trainTemp[i].length;j++){
                if(j==0){
                    train[i][j]=Integer.parseInt(trainTemp[i][j]);
                }
                else{
                    cellVal=trainTemp[i][j].split(":");
                    val=Integer.parseInt(cellVal[0]);
                    train[i][val+1]=Integer.parseInt(cellVal[1]);     
                }        
            }
        }

        trainLabels=new int[trainRow];
        for(int i=0;i<trainRow;i++){
            trainLabels[i]=train[i][0];
            train[i][0]=1;
        }
        

        
        weightSize=max+2;
        weights=new double[weightSize];
        for(int i=0;i<weightSize;i++){
            weights[i]=0;
        }
        
        
        max=0;
        val=0;
        for(int i=0;i<testRow;i++){
            for(int j=1;j<testTemp[i].length;j++){
                cellVal=testTemp[i][j].split(":");
                if((val=(Integer.parseInt(cellVal[0])))>max){
                    max=val;
                }
            }
        }
       
        testCol=max+2;
        test=new int[testRow][testCol];
        

        for(int i=0;i<testRow;i++){
            for(int j=0;j<testCol;j++){
                test[i][j]=0;
            } 
        }
        
        for(int i=0;i<testRow;i++){
            for(int j=0;j<testTemp[i].length;j++){
                if(j==0){
                    test[i][j]=Integer.parseInt(testTemp[i][j]);
                }
                else{
                    cellVal=testTemp[i][j].split(":");
                    val=Integer.parseInt(cellVal[0]);
                    test[i][val+1]=Integer.parseInt(cellVal[1]);     
                }        
            }
        }
        
        testLabels=new int[testRow];
        for(int i=0;i<testRow;i++){
            testLabels[i]=test[i][0];
            test[i][0]=1;
        }
        
//        for(int i=0;i<testRow;i++){
//            System.out.print(testLabels[i]+"\t");
//            for(int j=0;j<testCol;j++){
//                System.out.print(test[i][j]+"  ");
//            }
//            System.out.println();
//        }
    }
    
    void trainLogisticRegression(){
        mistakeCounter=0;
        float Y=rate;
        int t=1;
        System.out.println("Objective For Each Epoch:");
        System.out.println("\nEpoch\tObjective");
        for(int e=0;e<epochs;e++){
            shuffle();
            for(int i=0;i<trainRow;i++){
            //     Y=rate/(1+((rate*(t++))/(float)C));
                float yp=0;
                for(int j=0;j<trainCol;j++){
                    yp=(float) (yp+(weights[j]*(float)train[i][j]));
                }

                for(int k=0;k<trainCol;k++){
                    weights[k]=weights[k]-Y*((2*weights[k]/((float)C*C))-((((float)trainLabels[i]*(float)train[i][k]))/(1+Math.exp((float)trainLabels[i]*(float)yp))));
                }

            }
            calculateLogLikelihood(e);
        }
    }
    
    void predictLogisticRegressiononTrain(){
        mistakeCounter=0;
        int yp;
        for(int i=0;i<trainRow;i++){
            double ytemp=0;
            for(int j=0;j<trainCol && j<weightSize;j++){
                ytemp=ytemp+(weights[j]*train[i][j]);
            }
            yp=sgn(ytemp);
            if(yp!=trainLabels[i]){
                mistakeCounter++;
            }
        }
        accuracyTrain=((trainRow-mistakeCounter)/(float)trainRow);
        printTestStatsonTrain();
    }
    
    void predictLogisticRegressiononTest(){
        mistakeCounter=0;
        int yp;
        for(int i=0;i<testRow;i++){
            double ytemp=0;
            for(int j=0;j<testCol && j<weightSize;j++){
                ytemp=ytemp+(weights[j]*test[i][j]);
            }
            yp=sgn(ytemp);
            if(yp!=testLabels[i]){
                mistakeCounter++;
            }
        }
        accuracy=((testRow-mistakeCounter)/(float)testRow);
        mistakesTrain=mistakeCounter;
        printTestStatsonTest();
    }

    
    void printTrainStats(){
        System.out.print("\n\n***** Training on data : "+trainFile+" *****");
        System.out.print("\nRate used(r) = "+rate);
        System.out.print("\nC used = "+C);
        System.out.print("\nWeight vector(w):\n");
        for(int i=0;i<weightSize;i++){
            System.out.print(weights[i]+"  ");
        }
        System.out.print("\nBias(b) = "+bias);
        System.out.print("\nNumber of mistakes made in training = "+mistakeCounter);
        System.out.print("\nTotal input = "+trainRow);
        System.out.println("\nAccuracy = "+((trainRow-mistakeCounter)/(float)trainRow)*100+" %");
    }    
    
    void printTestStatsonTrain(){    
        System.out.print("\n***** Testing on Train data : "+trainFile+" *****");
        System.out.print("\nValue of C used= "+C);
        System.out.print("\nInitial Value of Learning Rate(Y) = "+rate);
        System.out.print("\nNumber of epochs used= "+epochs);
        System.out.print("\nNumber of incorrect predictions = "+mistakeCounter);
        System.out.print("\nTotal input = "+trainRow);
        System.out.println("\nAccuracy = "+accuracyTrain*100+" %");        
    }
    
    void printTestStatsonTest(){    
        System.out.print("\n***** Testing on Test data : "+testFile+" *****");
        System.out.print("\nValue of C used= "+C);
        System.out.print("\nInitial Value of Learning Rate(Y) = "+rate);
        System.out.print("\nNumber of epochs used= "+epochs);
        System.out.print("\nNumber of incorrect predictions = "+mistakeCounter);
        System.out.print("\nTotal input = "+testRow);
        System.out.println("\nAccuracy = "+accuracy*100+" %");
        
    }
    
    int sgn(double n){
        if((n)<0)
            return (-1);
        else
            return (+1);
    }
  
    void shuffle(){ 
        for(int i=0;i<trainRow;i++){
            Random rnum = new Random();
            int rand = rnum.nextInt(trainRow);
            for(int j=0;j<trainCol;j++){
                int temp=train[rand][j];
                train[rand][j]=train[i][j];
                train[i][j]=temp;
            }
            int tempL = trainLabels[rand];
            trainLabels[rand]=trainLabels[i];
            trainLabels[i]=tempL;
        }
    }
    
    void calculateLogLikelihood(int e){
        
        double logLike=0, weightProd=0,objective=0;
        
        for(int i=0;i<trainRow;i++){
            double yp=0;
            for(int j=0;j<trainCol;j++){
                yp= (yp+(weights[j]*(float)train[i][j]));
            }
            logLike=(float) (logLike+ (Math.log(1+Math.exp(-1*trainLabels[i]*yp))));   
        }
        for(int j=0;j<trainCol;j++){
            weightProd=weightProd+(weights[j]*weights[j]);
        }
        weightProd=weightProd/(C*C);
        objective=weightProd+logLike;
        System.out.println((e+1)+"\t"+objective);
    }
}

