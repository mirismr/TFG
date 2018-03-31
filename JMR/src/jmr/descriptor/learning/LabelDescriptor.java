/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jmr.descriptor.learning;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import jmr.descriptor.Comparator;
import jmr.descriptor.MediaDescriptorAdapter;

/**
 *
 * @author mirismr
 */
public class LabelDescriptor extends MediaDescriptorAdapter<BufferedImage> implements Serializable{
    
    protected String label;
    
    /**
     * Constructs a learning label descriptor, initializes it from the image 
     * given by parameter and set as comparator the default one.
     * 
     * @param image the image source
     */
    public LabelDescriptor(BufferedImage image) {
        super(image, new DefaultComparator()); //Implicit call to init      
    }   
    
    /**
     * Constructs a learning label descriptor, initializes it with the given 
     * label and set as comparator the default one.
     * 
     * @param label the label to be set
     */
    public LabelDescriptor(String label) {
        super(null, new DefaultComparator()); //Implicit call to init
        this.label = label;
    }
    
     /**
     * Returns the label associated to this descriptor
     * @return the label associated to this descriptor
     */
    public String getLabel(){
        return this.label;
    } 
    
    @Override
    public void init(BufferedImage media){
        this.label = media.toString();
    }
    
    /**
     * Returns a string representation of this descriptor.
     * 
     * @return a string representation of this descriptor 
     */
    @Override
    public String toString(){
        return "LearningLabelDescriptor: ["+this.label+"]";
    }
    
    /**
     * Functional (inner) class implementing a comparator between learning label descriptors
     */
    static class DefaultComparator implements Comparator<LabelDescriptor, Boolean> {
        @Override
        public Boolean apply(LabelDescriptor t, LabelDescriptor u) {
            String label_1 = t.label, label_2 = u.label;
 
            return label_1.equals(label_2);
        }    
    }
}
