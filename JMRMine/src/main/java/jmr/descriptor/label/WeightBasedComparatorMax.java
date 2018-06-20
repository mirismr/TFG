/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jmr.descriptor.label;

import java.io.Serializable;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;
import jmr.descriptor.Comparator;

/**
 *
 * @author mirismr
 */
public class WeightBasedComparatorMax implements Comparator<LabelDescriptor, Double>, Serializable {
        private boolean only_inclusion;
        
        
        public WeightBasedComparatorMax() {
            this(false);
        }
        
        public WeightBasedComparatorMax(boolean only_inclusion) {
            this.only_inclusion = only_inclusion;
        }
        
        /**
     * Returns a value related to the distance in which the this descriptor is
     * included in the one given by parameter. This method is used in comparator
     * inner classes.
     *
     * @param u the second label descriptor.
     * @param op_init the unary operator used to initialize the distance
     * accumulator (as a function of the fisrt distance).
     * @param op_aggregation the binary operator used to aggregate a new
     * distance to the previous ones.
     * @return a value related to the degree in which the first descriptor is
     * included in the second one (Double.POSITIVE_INFINITY if some label is not
     * included)
     */
    private Double inclusionDistance(LabelDescriptor t, LabelDescriptor u) {
        int equal;
        String label_i;
        Double dist = null, dist_ij= null;
        
        for (int i = 0; i < t.size(); i++) {
            label_i = t.getLabel(i);
            equal = 1;
            // We search the same label
            for (int j = 0; j < u.size() && equal != 0; j++) {
                equal = label_i.compareToIgnoreCase(u.getLabel(j)); // 0 if equals                  
                if (equal == 0) {
                    //We assume that the distance is given by the first coincidence
                    dist_ij = Math.abs(t.getWeight(i) - u.getWeight(j));
                }
            }
            if (equal != 0) {
                return Double.POSITIVE_INFINITY; //Same label not found
            } else {
                dist = dist==null ? dist_ij : Math.max(dist, dist_ij);
            }
        }
        return dist;
    }
    
        /**
         * Applies this comparator to the given arguments.
         *
         * @param t the first function argument
         * @param u the second function argument
         * @return the function result
         */
        @Override
        public Double apply(LabelDescriptor t, LabelDescriptor u) {
            if(!only_inclusion && t.size() != u.size()){
                return Double.POSITIVE_INFINITY;
            }
            // If the size is equal, and the labels are the same, the distance 
            // between t and u will be given by the inclusion of t in u (which 
            // will be the same that the inclusion of u in t). If the labels are
            // different, the inclusion will be Double.POSITIVE_INFINITY
            Double output = this.inclusionDistance(t, u);
          
            return output;
        }
        
        
}
