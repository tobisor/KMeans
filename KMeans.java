package KMeansProject;


import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;


public class KMeans {

    public int m_K;
    public Instance[] m_CentroidsArray;
    public int m_iterations = 40;

    private double currentAvgErr;

    /**
     * set K
     *
     * @param K
     */
    void setK(int K) {
        m_K = K;
    }

    /**
     * set number of iterations
     *
     * @param x
     */
    void setIterations(int x) {
        m_iterations = x;
    }

    /**
     * Input: Instances object
     * This method is building the KMeans object.
     * It should initialize centroids (by calling initializeCentroids) and run the
     * K-Means algorithm(which means to call findKMeansCentroids methods).
     */
    void buildClusterModel(Instances data) {
        currentAvgErr = Double.MAX_VALUE;
        initializeCentroids(data);
        findKMeansCentroids(data);
    }


    /**
     * Input: Instances
     * Initialize the centroids by selecting k random instances
     * from the training set and setting the centroids to be those instances.
     */

    void initializeCentroids(Instances data) {
        m_CentroidsArray = new Instance[m_K];
        Random random = new Random();
        int size = data.size();

        int counter = 0; //number of instances in centroid array
        while (counter < m_K) {

            boolean toAdd = true;
            //picking a random instance
            Instance temp = data.instance(random.nextInt(size));
            //checking if he has already assigned

            for (int j = 0; j < counter; j++) {
                if (equals(temp, m_CentroidsArray[j])) {
                    toAdd = false;
                    break;
                }
            }

            if (toAdd) m_CentroidsArray[counter++] = new DenseInstance(temp);

        }
    }

    /**
     * check if two instances are equal
     *
     * @param a
     * @param b
     * @return
     */
    private boolean equals(Instance a, Instance b) {
        for (int i = 0; i < a.numAttributes(); i++) {
            if (a.value(i) != b.value(i)) return false;
        }
        return true;
    }

    /**
     * Input: Instances
     * Should find and store the centroids according to the KMeans algorithm.
     * Your stopping condition for when to stop iterating can be either when the centroids
     * have not moved much from their previous location, the cost function did not change much,
     * or you have reached a preset number of iterations.
     * In this assignment we will only use the preset number option.
     * A good preset number of iterations is 40.
     * Use one or any combination of these methods to determine when to stop iterating.
     */
    void findKMeansCentroids(Instances data) {
        int i;
        for (i = 0; i < m_iterations; i++) {

            createNewCentroids(data);
            //if k = 5 print output
            if (m_K == 5) {

                System.out.println("AvgWSSSE is " + calcAvgWSSSE(data) + " .iteration number: " + i + 1);
            }
        }

    }

    /**
     *
     */
    private void createNewCentroids(Instances data) {
        double[][] rgbAvgArray = new double[m_K][4];
        double[] clusterSizes = new double[m_K];
        int size = data.size();

        //for each instance:add his RGB values to the RGB sums and increment the relevant counter.
        for (int i = 0; i < size; i++) {
            Instance temp = data.instance(i);
            int clusterIndex = findClosestCentroid(temp);
            rgbAvgArray[clusterIndex][0] += temp.value(0);
            rgbAvgArray[clusterIndex][1] += temp.value(1);
            rgbAvgArray[clusterIndex][2] += temp.value(2);
            rgbAvgArray[clusterIndex][3] += temp.value(3);
            clusterSizes[clusterIndex]++;
        }

        //divide all RGB sum values to create averages
        for (int i = 0; i < m_K; i++) {
            rgbAvgArray[i][0] /= clusterSizes[i];
            rgbAvgArray[i][1] /= clusterSizes[i];
            rgbAvgArray[i][2] /= clusterSizes[i];
            rgbAvgArray[i][3] /= clusterSizes[i];
        }

        //set new centroids
        for (int i = 0; i < m_K; i++) {
            m_CentroidsArray[i].setValue(0, rgbAvgArray[i][0]);
            m_CentroidsArray[i].setValue(1, rgbAvgArray[i][1]);
            m_CentroidsArray[i].setValue(2, rgbAvgArray[i][2]);
            m_CentroidsArray[i].setValue(3, rgbAvgArray[i][3]);
        }
    }

    /**
     * calculating squared distance between instance and its centroid.
     *
     * @param instance
     * @param centroid
     * @return
     */

    double calcSquaredDistanceFromCentroid(Instance instance, Instance centroid) {
        double sum = 0;
        //iterating each attribute to calculate distance
        for (int i = 0; i < instance.numAttributes(); i++) {
            sum += Math.pow(instance.value(i) - centroid.value(i), 2);
        }
        return sum;
    }

    /**
     * @param pixel
     * @return
     */
    int findClosestCentroid(Instance pixel) {
        double min_dist = Double.MAX_VALUE;
        int closestCentroid = 0;
        //iterate each centroid and save for the minimal distance
        for (int i = 0; i < m_K; i++) {
            double temp_dist = calcSquaredDistanceFromCentroid(pixel, m_CentroidsArray[i]);
            if (temp_dist < min_dist) {
                min_dist = temp_dist;
                closestCentroid = i;
            }
        }
        return closestCentroid;
    }

    /**
     * this method create new set of instances from the data where each instance
     * is replced with his centroid.
     *
     * @param data
     * @return quantized data
     */
    Instances quantize(Instances data) {
        Instances quantizedData = new Instances(data);
        for (int i = 0; i < data.size(); i++) {
            quantizedData.set(i, m_CentroidsArray[findClosestCentroid(data.instance(i))]);
        }

        return quantizedData;
    }

    /**
     * @param data
     * @return the average within set sum of squared errors.
     */
    double calcAvgWSSSE(Instances data) {
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            sum += calcSquaredDistanceFromCentroid(data.instance(i),
                    m_CentroidsArray[findClosestCentroid(data.instance(i))]);
        }
        return sum / (double) data.size();
    }

}
