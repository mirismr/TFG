package jmr.descriptor.label;

import java.io.Serializable;
import jmr.descriptor.Comparator;
import jmr.descriptor.MediaDescriptorAdapter;

/**
 * A descriptor representing a single label associated to a visual media.
 * 
 * @param <T> the type of media described by this descriptor.
 * 
 * @author Jesús Chamorro Martínez (jesus@decsai.ugr.es)
 */
public class SingleLabelDescriptor<T> extends MediaDescriptorAdapter<T> implements Serializable {   
    /**
     * Label associated to this descriptor.
     */
    private String label;
    /**
     * A classifier used for labeling a given media. It uses a standard
     * functional interface, allowing lambda expressions.
     */
    private Classifier<T, String> classifier = null;
     
    /**
     * Constructs a single label descriptor using the default classifier to 
     * label the given image, and set as comparator the default one.
     * 
     * @param media the media source
     */
    public SingleLabelDescriptor(T media) {
        this(media, new DefaultClassifier()); 
    }   
    
    /**
     * Constructs a single label descriptor using the given classifier to label
     * the given image, and set as comparator the default one.
     *
     * @param media the media source
     * @param classifier the classifier used for labeling a given media. The
     * result type of the classifier must be <code>String</code>.
     */
    public SingleLabelDescriptor(T media, Classifier classifier) {
        super(media, new DefaultComparator()); //Implicit call to init 
        // The previous call does not initialize the label since the classifier
        // has not been assigned yet. Therefore, in the following sentences the
        // classifier data member is initialize and then used for obtaining the
        // label of this descriptor
        this.classifier = classifier;
        this.init(media); //Second call, but needed (see init method)
    }   
    
    /**
     * Constructs a single label descriptor, initializes it with the given 
     * label and set as comparator and classifier the default ones.
     * 
     * @param label the label to be set
     */
    public SingleLabelDescriptor(String label) {
        this((T)null); //Default comparator and classifier; null source
        this.label = label;
    }
        
    /**
     * Initialize the descriptor by using the classifier.
     *
     * @param media the media used for initializating this descriptor
     */
    @Override
    public void init(T media) {
        label = media!=null && classifier!=null ? classifier.apply(media) : null;
        // When this method is called from the superclass constructor, the local
        // member data, and particularly the classifier, are not initialized 
        // yet. Thus, in the construction process, the previous code always 
        // initializes the label to null. For this reason, after the super() 
        // call in the constructor, we have to (1) initialize the rest of the 
        // descriptor (particularly the classifier) and (2) to calculte the
        // label again (for example, calling this init method again).
        //
        // Note that this method is not only called from the constructor, it is 
        // also called from the setSource method (which allow to chage de media
        // and, consequently, it changes the label using the current classidier
    }
       
    /**
     * Returns the label associated to this descriptor
     * @return the label associated to this descriptor
     */
    public String getLabel(){
        return label;
    } 
    
    /**
     * Set the classifier for this descriptor.
     *
     * @param classifier the new classifier. The result type of the classifier
     * must be <code>String</code>
     */
    public void setClassifier(Classifier<T, String> classifier){
        this.classifier = classifier;
    }
    
    /**
     * Returns the classifier of this descriptor. 
     * 
     * @return the classifier of this descriptor. 
     */
    public Classifier getClassifier(){
        return classifier;
    }
    
    /**
     * Returns a string representation of this descriptor.
     * 
     * @return a string representation of this descriptor 
     */
    @Override
    public String toString(){
        return this.getClass().getSimpleName()+": ["+label+"]";
    }
    
    /**
     * Functional (inner) class implementing a comparator between single label
     * descriptors. It returns 0.0 if the labels are different and 1.0 if they 
     * are equals (ignoring upper cases).
     */
    static class DefaultComparator implements Comparator<SingleLabelDescriptor, Double> {
        @Override
        public Double apply(SingleLabelDescriptor t, SingleLabelDescriptor u) {
            int equal = t.label.compareToIgnoreCase(u.label);
            return equal == 0 ? 0.0 : 1.0;
        }
    }
    
    /**
     * Functional (inner) class implementing a default classifier. This
     * implementation labels the media by the (simple) name of its class.
     */
    static public class DefaultClassifier<T> implements Classifier<T, String> {
        @Override
        public String apply(T t) {
            return (t!=null) ? t.getClass().getSimpleName() : "";
        }
    }
}
