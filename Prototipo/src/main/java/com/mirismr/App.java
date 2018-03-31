package com.mirismr;


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
            MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights("D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\model_exported.h5", "D:\\Mega\\Universidad\\Cuarto\\TFG\\Prototipo\\src\\files\\weights_exported.json", false);
            //System.out.println(model.getLayerNames());

            int height = 64;
            int width = 64;
            int channels = 3;

            //File file = new File("D:\\Mega\\Universidad\\Cuarto\\TFG\\Keras\\data\\predict\\n01443537\\n01443537_203.JPEG");
            File file = new File("D:\\Mega\\Universidad\\Cuarto\\TFG\\Keras\\data\\predict\\n01629819_409.JPEG");
            //File file = new File("D:\\Mega\\Universidad\\Cuarto\\TFG\\Keras\\data\\predict\\n02094433_494.JPEG");
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
            JSONObject diccionarioImagenes = (JSONObject) parser.parse(new FileReader("D:\\Mega\\Universidad\\Cuarto\\TFG\\Keras\\class_dictionary.json"));
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
