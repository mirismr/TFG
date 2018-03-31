/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jmr.descriptor.learning;

import java.awt.image.BufferedImage;

/**
 *
 * @author mirismr
 */
public abstract class LearnedLabelDescriptor extends LabelDescriptor{

    public LearnedLabelDescriptor(BufferedImage image) {
        super(image); //Implicit call to init      
    }   
    
    @Override
    public final void init(BufferedImage media){
        this.learnLabel(media);
    }
    
    public abstract void learnLabel(BufferedImage image);
}
