package jmr.iu;

import java.awt.*;
import java.io.File;

import javax.swing.*;
import jmr.descriptor.Comparator;
import jmr.descriptor.color.SingleColorDescriptor;
import jmr.media.JMRBufferedImage;
import jmr.video.FrameCollection;
import jmr.video.FrameCollectionIO;
import jmr.video.KeyFrameDescriptor;
import jmr.video.MinMinComparator;


public class Test {
    boolean packFrame = false;

    /**
     * Construct and show the application.
     */
    public Test() {
        
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
        Comparator<SingleColorDescriptor, Double> c = (a,b)->1.0;
        d2.setComparator(c);
        o = d2.compare(d3);
        System.out.println("Output: " + o + " de tipo " + o.getClass());
        Comparator<SingleColorDescriptor, Color> c2 = (a,b)->Color.blue;
        d2.setComparator(c2);
        o = d2.compare(d3);
        System.out.println("Output: " + o + " de tipo " + o.getClass());   
        
        TestVideo();
    }
    
    private void TestVideo(){
        File file = new File("C:\\Users\\Jesús\\Documents\\_JMR_TestImages\\video");
        FrameCollection fc = FrameCollectionIO.read(file);
        KeyFrameDescriptor kfd = new KeyFrameDescriptor(fc,jmr.descriptor.color.SingleColorDescriptor.class);
        
        
                
        System.out.println("KFD: \n"+kfd);
        
        file = new File("C:\\Users\\Jesús\\Documents\\_JMR_TestImages\\036.jpg");
        FrameCollection fc2 = FrameCollectionIO.read(file);
        KeyFrameDescriptor kfd2 = new KeyFrameDescriptor(fc2,jmr.descriptor.color.SingleColorDescriptor.class);
        System.out.println("KFD: \n"+kfd2);
        
        Double dist = kfd.compare(kfd2);
        System.out.println(dist);
        
        
        Comparator<KeyFrameDescriptor, Double> ckfd = new MinMinComparator();
        kfd.setComparator(ckfd);
        dist = kfd.compare(kfd2);
        System.out.println(dist);
        
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
