package tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class JsonMerger {
    private final static List<String> modelNames = new ArrayList<>();


    public static void main(String[] args) {

        modelNames.addAll(Arrays.asList("alexnet", "bert", "googlenet", "inceptionv3", "resnet50", "squeezenet", "vgg19"));
        for (String modelName : modelNames) {
            try {
                JsonNode cpu = new ObjectMapper().readTree(new File("./trace/" + modelName + "_CPU.json"));
                JsonNode gpu = new ObjectMapper().readTree(new File("./trace/" + modelName + "_GPU.json"));

                ArrayNode merge = new ObjectMapper().createArrayNode();
                for(JsonNode n : cpu) {
                    merge.add(n);
                }
                for(JsonNode n : gpu) {
                    merge.add(n);
                }
                OutputStream os = new FileOutputStream("./trace/" + modelName + ".json");
                new ObjectMapper().writeValue(os, merge);
            } catch (Exception e) {
                e.printStackTrace();
            }


        }
    }
}
