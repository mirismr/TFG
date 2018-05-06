package com.mirismr;


import com.sun.org.apache.xpath.internal.operations.Mult;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class App 
{
    private static Logger log = LoggerFactory.getLogger(App.class);
    public static void main(String[] args)
    {
        try {
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\final_model_exported_fold_4.h5");
            System.out.println(model.getLayerNames());
            int height = 224;
            int width = 224;
            int channels = 3;
            //File file = new File("D:\\Descargas\\n02121808_252.JPEG");
            //File file = new File("D:\\Descargas\\n02085374_945.JPEG");
            File file = new File("D:\\Descargas\\photo6014978389993172495.jpg");
            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
            INDArray image = loader.asMatrix(file);

            INDArray result = model.output(image, false);
            System.out.println(result);
            boolean encontrado = false;
            int label = -1;
            for (int i = 0; i < result.length() && !encontrado; i++) {
                if( String.valueOf(result.getColumn(i)).compareTo("1.00") == 0) {
                    encontrado = true;
                    label = i;
                }
            }
            JSONParser parser = new JSONParser();
            JSONObject diccionarioImagenes = (JSONObject) parser.parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Keras\\models\\vgg16\\class_dictionary.json"));
            String labelPredict = (String) diccionarioImagenes.get(String.valueOf(label));
            System.out.println("Diccionario: "+diccionarioImagenes);
            System.out.println("Imagen predecida: "+labelPredict);

        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}
