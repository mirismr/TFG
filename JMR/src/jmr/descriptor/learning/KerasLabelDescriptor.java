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
public class KerasLabelDescriptor extends LearnedLabelDescriptor{

    //private ModelKeras model;
    
    public KerasLabelDescriptor(BufferedImage image) {
        super(image);
    }

    @Override
    public void learnLabel(BufferedImage image) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        /*this.label = this.model.predict(image);*/
    }

}
