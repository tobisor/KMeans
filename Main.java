package KMeansProject;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;

import javax.imageio.ImageIO;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Main {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    private static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static Instances convertImgToInstances(BufferedImage image) {
        Attribute attribute1 = new Attribute("alpha");
        Attribute attribute2 = new Attribute("red");
        Attribute attribute3 = new Attribute("green");
        Attribute attribute4 = new Attribute("blue");
        ArrayList<Attribute> attributes = new ArrayList<Attribute>(4);
        attributes.add(attribute1);
        attributes.add(attribute2);
        attributes.add(attribute3);
        attributes.add(attribute4);
        Instances imageInstances = new Instances("Image", attributes, image.getHeight() * image.getWidth());

        int[][] result = new int[image.getHeight()][image.getWidth()];
        int[][][] resultARGB = new int[image.getHeight()][image.getWidth()][4];

        for (int col = 0; col < image.getWidth(); col++) {
            for (int row = 0; row < image.getHeight(); row++) {
                int pixel = image.getRGB(col, row);

                int alpha = (pixel >> 24) & 0xff;
                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = (pixel) & 0xff;
                result[row][col] = pixel;
                resultARGB[row][col][0] = alpha;
                resultARGB[row][col][1] = red;
                resultARGB[row][col][2] = green;
                resultARGB[row][col][3] = blue;

                Instance iExample = new DenseInstance(4);
                iExample.setValue((Attribute) attributes.get(0), alpha);// alpha
                iExample.setValue((Attribute) attributes.get(1), red);// red
                iExample.setValue((Attribute) attributes.get(2), green);// green
                iExample.setValue((Attribute) attributes.get(3), blue);// blue
                imageInstances.add(iExample);
            }
        }

        return imageInstances;

    }

    public static BufferedImage convertInstancesToImg(Instances instancesImage, int width, int height) {
        final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                Instance instancePixel = instancesImage.instance(index);
                int pixel = ((int) instancePixel.value(0) << 24) | (int) instancePixel.value(1) << 16
                        | (int) instancePixel.value(2) << 8 | (int) instancePixel.value(3);
                image.setRGB(col, row, pixel);
                index++;
            }
        }
        return image;
    }

    /**
     * @param a
     * @param b
     * @return
     */
    static double calcAvgDistance(Instances a, Instances b) {
        double averageDistance = 0;
        int size = a.size();
        for (int i = 0; i < size; i++) {
            averageDistance += calcDistance(a.instance(i), b.instance(i));
        }
        return averageDistance / size;
    }

    /**
     * calculating euclidean distance between two instances.
     *
     * @param a
     * @param b
     * @return
     */

    static double calcDistance(Instance a, Instance b) {
        double sum = 0;
        for (int i = 0; i < a.numAttributes(); i++) {
            sum += Math.pow(Math.abs(a.value(i) - b.value(i)), 2);
        }
        return Math.pow(sum, .5f);
    }

    public static void main(String[] args) throws Exception {
        /******************----------------K-MEAN-------------********************/
        //reading image and turn it into a data set
        BufferedImage image = ImageIO.read(new File("messi.jpg"));
        Instances data = convertImgToInstances(image);

        //initializing array of K values
        int[] arrK = new int[]{2, 3, 5, 10, 25, 50, 100, 256};
        KMeans kMeans = new KMeans();

        for (int i = 0; i < arrK.length; i++) {
            int k = arrK[i];

            kMeans.setK(k);
            kMeans.buildClusterModel(data);

            //turning quantized data set to image
            BufferedImage imageII = convertInstancesToImg(kMeans.quantize(data),
                    image.getWidth(), image.getHeight());
            File outputfile = new File("messi" + k + ".jpg");
            ImageIO.write(imageII, "jpg", outputfile);
        }

        /******************----------------PCA-------------********************/

        Instances pcaData = loadData("libras.txt");
        PrincipalComponents pca = new PrincipalComponents();

        for (int i = 13; i < 91; i++) {
            //for each number of principal components, preform :
            //1. Run PCA on the instances and transform them back to the original space
            //2.Measure the average Euclidean distance of the new data set from
            //  the original data set and print this average distance to the console.

            pca.setNumPrinComponents(i);
            pca.setTransformBackToOriginal(true);
            pca.buildEvaluator(pcaData);
            Instances transformedData = pca.transformedData(pcaData);
            double dist = calcAvgDistance(pcaData, transformedData);

            System.out.println("using " + i + " principal components provides" +
                    " average distance of: " + dist);
        }


    }
}

