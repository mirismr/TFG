package jmr.db;

import java.io.Serializable;
import java.net.URL;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jmr.descriptor.DescriptorList;
import jmr.descriptor.MediaDescriptor;
import jmr.descriptor.MediaDescriptorFactory;
import jmr.result.ResultMetadata;

/**
 * Class representing a database stored as a list of descriptors in the main 
 * memory.
 *
 * @param <T> the media type of this database
 *
 * @author Jesús Chamorro Martínez (jesus@decsai.ugr.es)

 */
public class ListDB<T> implements Serializable{
    /**
     * List of database records
     */
    private ArrayList<Record> database = null;
    /**
     * List of the descriptor classes associated to this database
     */
    private Class descriptorClasses[] = null;
    /**
     * Reference for record ordering.
     */
    private Record orderReference = null;
    
    /**
     * Constructs an empty database.
     * 
     * @param descriptorClasses the list of descriptor classes that will
     * determine the set of descriptor associated to this database. Each
     * descriptor class have to provide, at least, a constructor with a single
     * parameter of <code>T</code> type.
     */
    public ListDB(Class... descriptorClasses){
        this.descriptorClasses = descriptorClasses;
        database = new ArrayList();
    }
    
    /**
     * Appends the specified record to the end of this database.
     *
     * @param record record to be appended to this database
     * @return <tt>true</tt> (as specified by 
     * {@link java.util.Collection#add(java.lang.Object) })
     * @throws InvalidParameterException if the new record does not share the
     * database structure.
     */
    public boolean add(Record record) {
        if (!record.isCompatible()) {
            throw new InvalidParameterException("The new record does not share the database structure.");
        }
        return database.add(record);
    }
    
    /**
     * Appends a new record to the end of this database.
     *
     * @param media media from which the new record is calculated
     * @return <tt>true</tt> (as specified by 
     * {@link java.util.Collection#add(java.lang.Object) })
     * @throws InvalidParameterException
     */
    public boolean add(T media){
        Record record = new Record(media);
        return database.add(record);
    }
    
    public boolean add(T media, URL locator){
        Record record = new Record(media, locator);
        return database.add(record);
    }

    /**
     * Inserts the specified record at the specified position in this database
     * Shifts the record currently at that position (if any) and any subsequent
     * records to the right (adds one to their indices).
     *
     * @param index index at which the specified record is to be inserted
     * @param record record to be inserted
     * @throws IndexOutOfBoundsException {@inheritDoc}
     * @throws InvalidParameterException
     */
    public void add(int index, Record record) {
        if (!record.isCompatible()) {
            throw new InvalidParameterException("The new record does not share the data base structure.");
        }
        database.add(index, record);
    }
    
    /**
     * Inserts a new record at the specified position in this database.
     * Shifts the record currently at that position (if any) and any subsequent
     * records to the right (adds one to their indices).
     *
     * @param index index at which the specified record is to be inserted
     * @param media media from which the new record is calculated
     * @throws IndexOutOfBoundsException {@inheritDoc}
     * @throws InvalidParameterException
     */
    public void add(int index, T media) {
        Record record = new Record(media);
        database.add(index, record);
    }

    /**
     * Replaces the record at the specified position in this database with the
     * specified record.
     *
     * @param index index of the record to replace
     * @param record record to be stored at the specified position
     * @return the record previously at the specified position
     * @throws IndexOutOfBoundsException {@inheritDoc}
     * @throws InvalidParameterException
     */
    public Record set(int index, Record record) {
        if (!record.isCompatible()) {
            throw new InvalidParameterException("The new record does not share the data base structure.");
        }
        return database.set(index, record);
    }

    /**
     * Replaces the record at the specified position in this database with a
     * new record calculated from the given media.
     *
     * @param index index of the record to replace
     * @param media media from which the new record is calculated
     * @return the record previously at the specified position
     * @throws IndexOutOfBoundsException {@inheritDoc}
     * @throws InvalidParameterException
     */
    public Record set(int index, T media) {
        Record record = new Record(media);
        return database.set(index, record);
    }
    
    /**
     * Returns the record at the specified position in this database.
     *
     * @param index index of the record to return
     * @return the record at the specified position in this database
     * @throws IndexOutOfBoundsException {@inheritDoc}
     */
    public Record get(int index) {
        return database.get(index);
    }

    /**
     * Removes the record at the specified position in this database. Shifts any
     * subsequent record to the left (subtracts one from their indices).
     *
     * @param index the index of the record to be removed
     * @return the record that was removed from the database
     * @throws IndexOutOfBoundsException {@inheritDoc}
     */
    public Record remove(int index) {
        return database.remove(index);
    }

    /**
     * Removes all of the record from this database. The database will be empty
     * after this call returns.
     */
    public void clear() {
        database.clear();
    }

    /**
     * Returns the number of records in this database.
     *
     * @return the number of records in this database
     */
    public int size() {
        return database.size();
    }

    /**
     * Returns <tt>true</tt> if this database contains no record.
     *
     * @return <tt>true</tt> if this database contains no record
     */
    public boolean isEmpty() {
        return database.isEmpty();
    }
    
    /**
     * Returns the list of descriptor classes associated to this database.
     * 
     * @return the list of descriptor classes
     */
    public List<Class> getDescriptorClasses(){ 
        return Arrays.asList(this.descriptorClasses);
    }
    
    /**
     * Set a record as reference for record ordering.
     * 
     * The record order is based on the distance to a reference record: a record 
     * will be less than, equal to, or greater than other if its distance to the 
     * reference is less than, equal to, or greater than the distance of the
     * other record to the same reference.
     * 
     * The reference record can be stored in the database or not. If not, it 
     * have to share the database structure.
     * 
     * @param orderReference the new reference
     * throws InvalidParameterException
     */
    public void setOrderReference(Record orderReference){
        if (!orderReference.isCompatible()) {
            throw new InvalidParameterException("The new record does not share the data base structure.");
        }
        this.orderReference = orderReference;
    }
    
    /**
     * Returns the record used as reference for record ordering.
     * 
     * @return the record used as reference for record ordering.
     */
    Record getOrderReference(){
        return this.orderReference;
    }
    
    /**
     * Returns the records of this databa ordered on the basis of its distance
     * to the given query.
     * 
     * @param queryRecord the query record
     * @return a list of records ordered by distante to <code>queryRecord</code>
     */
    public List<Record> query(Record queryRecord){
        if (!queryRecord.isCompatible()) {
            throw new InvalidParameterException("The query record does not share the data base structure.");
        }
        List output = (List)database.clone(); // Shallow copy 
        // The output will be ordered by using the query record as reference  
        Record actual_orderReference = orderReference;
        this.orderReference = queryRecord;
        output.sort(null);
        this.orderReference = actual_orderReference;
        
        return output;
    }
    
    /**
     * Returns a subset of this databa corresponding to the nearest records to 
     * the given query one. The output is sorted on the basis of the distance 
     * to the query.
     * 
     * @param queryRecord the query record
     * @param size the size of the output subset 
     * @return the nearest records sorted by distance
     */
    public List<Record> query(Record queryRecord, int size){
        if (!queryRecord.isCompatible()) {
            throw new InvalidParameterException("The query record does not share the data base structure.");
        }
        List output = query(queryRecord);
        return output.subList(0, size);
        
        //Record actual_orderReference = orderReference;
        //this.orderReference = queryRecord;         
        //size = Math.min(Math.max(size,0), database.size());
        //TreeSet<Record> output = new TreeSet(database.subList(0,size)); //Sorted    
        //Record current, last = output.last(); // Highest element currently in output
        //for(int i=size; i<database.size(); i++){
        //    current = database.get(i);  
        //    if(last.compareTo(current)>0) {
        //        output.remove(last);                
        //        output.add(database.get(i)); // Added in a sorted way
        //        last = output.last();
        //    }
        //}
        //this.orderReference = actual_orderReference;
        //return new ArrayList(output);
        // NOTE: the previous implementation is faster than the easiest following
        // code: (1) call to query(queryRecord) and (2) return the sublist by
        // calling subList(0,size). If the size of the database (n) is very high,  
        // the sorting in the query method (O(n·logn)) have more cost than the  
        // insertion made in this implementation (O(n·m), with m<<n) For example,
        // if m=10 (the size parameter), this implementaion is faster for n>1024.
    }
    
    /**
     * Returns metadata info of the records of this databa ordered on the basis 
     * of ist distance to the given query.
     * 
     * The returned info is a list o pairs [distance, source] encapsulated in a 
     * {@link jmr.result.ResultMetadata} object, where the result type is a 
     * {@link java.lang.Double} representing the distance to the query, and the
     * metadata type is a <code>T</code> object representing the source 
     * associated to that distance.
     * 
     * @param queryRecord the query record
     * @return a list of ordered metadata
     */
    public List<ResultMetadata<Double,T>> queryMetadata(Record queryRecord){ 
        if (!queryRecord.isCompatible()) {
            throw new InvalidParameterException("The query record does not share the data base structure.");
        }
        List output = new ArrayList<>();
        Object distance;
        for(Record r: database){
            distance = queryRecord.compare(r);
            output.add(new ResultMetadata(distance, r.getSource()));
        }
        output.sort(null);
        return output;
    }
    
    /**
     * Returns a string representation of this database
     * .
     * @return a string representation of this database 
     */
    @Override
    public String toString(){
        String output ="";
        output += this.database.toString();
        return output;
    }
    
    
    /**
     * Inner class representing a single item (record) in the data base. In this
     * case, a record will correspond to the list of descriptors associated to
     * the media of the item.
     *
     */
    public class Record extends DescriptorList<T> implements Comparable<Record>{
        /**
         * Locator of the media associated to this record (null if not available).
         * 
         * It is useful when the database has been opened from a file and, then,
         * the media source objects are not available (in main memory) from the 
         * descriptors; in that case, if we need the source (for example, to 
         * show it), the locator will be useful for opening tasks;
         * 
         */
        private URL locator=null;
        
        /**
         * Constructs a record associated to the given media, initializing its
         * list of descriptors.
         *
         * The descriptors will be automatically calculated on the basis of the  
         * list of descriptor classes provided by the database (each descriptor 
         * class have to provide, at least, a constructor with a single  
         * parameter of type <code>T</code>). The comparator will be set to the 
         * default one.
         *
         * @param media the source media of this record
         */
        public Record(T media) {
            super(media);
            this.initDescriptors(descriptorClasses);
        }
        
        public Record(T media, URL locator) {
            this(media);
            this.locator = locator;
        }
        
        public URL getLocator(){
            return locator;
        }
        
        /**
         * Initializes the list of descriptors associated to this record.
         *
         * @param descriptorClasses the list of descriptor classes that will
         * determine the set of descriptor associated to this record. Each
         * descriptor class have to provide a constructor with a single
         * parameter of the query type.
         */
        private void initDescriptors(Class... descriptorClasses) {
            MediaDescriptor<T> descriptor;
            for (Class c : descriptorClasses) {
                descriptor = MediaDescriptorFactory.getInstance(c, this.source);
                this.add(descriptor);
            }
        }
        
        /**
         * Check if this record is compatible with the database. 
         * 
         * It is compatible if the descriptors are the same (in number and 
         * class) and in the same order.
         * 
         * @return <tt>true</tt> if the record is compatible
         */
        public boolean isCompatible() {
            boolean compatible = (this.size() == descriptorClasses.length);
            for (int i = 0; i < this.size() && compatible; i++) {
                if (this.get(i).getClass() != descriptorClasses[i])
                    compatible = false;
            }
            return compatible;
        }                

        /**
         * Compares this record with the given parameter for order. Returns a
         * negative integer, zero, or a positive integer as this record is less
         * than, equal to, or greater than the given parameter. 
         * 
         * The order is based on the distance to a reference record in the 
         * database: this record will be less than, equal to, or greater than 
         * the given parameter if its distance to the reference is less than, 
         * equal to, or greater than the distance of the given parameter to the
         * reference.
         *
         * @param o the record to be compared
         * @return the comparision results
         */
        @Override
        public int compareTo(Record o) {
            if (orderReference == null) {
                return 0;
            }
            Double d1 = this.compare(orderReference);
            Double d2 = o.compare(orderReference);
            return d1.compareTo(d2);
        }
    } // end inner class
}
