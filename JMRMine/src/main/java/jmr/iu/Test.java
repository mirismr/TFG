package jmr.iu;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

import javax.swing.*;
import jmr.descriptor.Comparator;
import jmr.descriptor.color.SingleColorDescriptor;
import jmr.descriptor.label.Classifier;
import jmr.descriptor.label.LabelDescriptor;
import jmr.descriptor.label.SingleLabelDescriptor;
import jmr.media.JMRBufferedImage;
import jmr.video.FrameCollection;
import jmr.video.FrameCollectionIO;
import jmr.video.KeyFrameDescriptor;
import jmr.video.MinMinComparator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class Test {

    boolean packFrame = false;

    /**
     * Construct and show the application.
     */
    public Test() {
        TestLabel();
    }

    private void TestColor() {
        JMRBufferedImage img = null;
        SingleColorDescriptor d = new SingleColorDescriptor(img);
        System.out.println("Color: " + d.getColor());
        SingleColorDescriptor d2 = new SingleColorDescriptor(Color.LIGHT_GRAY);
        System.out.println("Color: " + d2.getColor());
        SingleColorDescriptor d3 = new SingleColorDescriptor(Color.WHITE);
        System.out.println("Color: " + d3.getColor());

        Object o;
        o = d2.compare(d3);
        System.out.println("Output: " + o + " de tipo " + o.getClass());
        Comparator<SingleColorDescriptor, Double> c = (a, b) -> 1.0;
        d2.setComparator(c);
        o = d2.compare(d3);
        System.out.println("Output: " + o + " de tipo " + o.getClass());
        Comparator<SingleColorDescriptor, Color> c2 = (a, b) -> Color.blue;
        d2.setComparator(c2);
        o = d2.compare(d3);
        System.out.println("Output: " + o + " de tipo " + o.getClass());
    }

    private void TestVideo() {
        File file = new File("C:\\Users\\Jesús\\Documents\\_JMR_TestImages\\video");
        FrameCollection fc = FrameCollectionIO.read(file);
        KeyFrameDescriptor kfd = new KeyFrameDescriptor(fc, jmr.descriptor.color.SingleColorDescriptor.class);

        System.out.println("KFD: \n" + kfd);

        file = new File("C:\\Users\\Jesús\\Documents\\_JMR_TestImages\\036.jpg");
        FrameCollection fc2 = FrameCollectionIO.read(file);
        KeyFrameDescriptor kfd2 = new KeyFrameDescriptor(fc2, jmr.descriptor.color.SingleColorDescriptor.class);
        System.out.println("KFD: \n" + kfd2);

        Double dist = kfd.compare(kfd2);
        System.out.println(dist);

        Comparator<KeyFrameDescriptor, Double> ckfd = new MinMinComparator();
        kfd.setComparator(ckfd);
        dist = kfd.compare(kfd2);
        System.out.println(dist);

    }

    private Double miApply(String s) {
        return 2.0;
    }

    private void TestLabel() {
        /*String path = "D:\\Descargas\\imagenesTFG\\dog_cat.jpg";
        File f = new File(path);
        BufferedImage img;
        try {
            img = ImageIO.read(f);
            
            IO io = new IO();
            MultiLayerNetwork network = io.loadModel("D:\\Descargas\\pesos\\vgg16_mine_sigmoid.h5");
            Map<Integer, String> classMap = io.loadClassMap("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\my_class_index.json");
            KerasClassifier clasificador = new KerasClassifier(network, new Dimension(224,224), classMap, 0.5);
            LabelDescriptor descriptor = new LabelDescriptor(img, clasificador);
            System.out.println(descriptor);
        } catch (IOException ex) {
            Logger.getLogger(Test.class.getName()).log(Level.SEVERE, null, ex);
        }*/

        String array[] = new String[]{"hola"};

        LabelDescriptor<BufferedImage> mlabel3 = new LabelDescriptor(array[0]);
        mlabel3.setWeights(1.7);
        System.out.println(mlabel3);

        /*SingleLabelDescriptor<BufferedImage> slabel1 = new SingleLabelDescriptor("Hola");
        SingleLabelDescriptor<BufferedImage> slabel2 = new SingleLabelDescriptor("Hola");
        System.out.println(slabel1 + "\n" + slabel2 + "\n Distancia:" + slabel1.compare(slabel2));

        slabel1 = new SingleLabelDescriptor(img);
        System.out.println(slabel1 + "\n" + slabel2 + "\n Distancia:" + slabel1.compare(slabel2));

        LabelDescriptor<BufferedImage> mlabel1 = new LabelDescriptor("Hola", "Adios");
        LabelDescriptor<BufferedImage> mlabel2 = new LabelDescriptor("Adios", "Hola");
        System.out.println(mlabel1 + "\n" + mlabel2 + "\n Distancia:" + mlabel1.compare(mlabel2));

        mlabel1 = new LabelDescriptor(img);
        System.out.println(mlabel1 + "\n" + mlabel2 + "\n Distancia:" + mlabel1.compare(mlabel2));

        LabelDescriptor<BufferedImage> label1 = new LabelDescriptor(img);
        LabelDescriptor<BufferedImage> label2 = new LabelDescriptor("Adios");
        System.out.println(label1 + "\n" + label2 + "\n Distancia:" + label1.compare(label2));

        Classifier<String, Double> c = new SingleLabelDescriptor.DefaultClassifier();
        System.out.println(c.apply("hola"));

        c = (s) -> s.length() * 2.0;
        System.out.println(c.apply("hola"));*/
//         label1.setClassifier(new SingleLabelDescriptor.DefaultClassifier());
//         label1.setSource(img);
//         System.out.println(label1+"\n"+label2+"\n Distancia:"+label1.compare(label2));
//         
//         label1.setClassifier(new MultipleLabelDescriptor.DefaultClassifier());
//         label1.setSource(img);
//         System.out.println(label1+"\n"+label2+"\n Distancia:"+label1.compare(label2));
    }

    /**
     * Application entry point.
     *
     * @param args String[]
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new Test();
            }
        });
    }
}
