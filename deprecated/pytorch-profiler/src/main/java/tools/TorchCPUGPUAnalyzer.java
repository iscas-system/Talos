package tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.protobuf.MapEntry;

import java.io.File;
import java.io.FileOutputStream;
import java.util.*;

public class TorchCPUGPUAnalyzer {
    public static void main(String[] args) throws Exception {
        HashMap<String, Double> cpuMap = new HashMap<>();
        HashMap<String, Double> gpuMap = new HashMap<>();
        HashMap<String, Integer> cpuCount = new HashMap<>();
        HashMap<String, Integer> gpuCount = new HashMap<>();


        HashMap<String, Double> speedup = new HashMap<>();

        Trace[] traces = new Trace[]{new Trace("cpu", args[1]), new Trace("gpu", args[3])};
        String outputFile = args[5];
        for(Trace trace : traces) {
            JsonNode dataSet = new ObjectMapper().readTree(
                    new File(trace.getFilePath()));

            List<JsonNode> firstLayerNodes = new ArrayList<>();

            Iterator<JsonNode> iter = dataSet.iterator();

            Double endTime = 0.0;

            while(iter.hasNext()) {
                JsonNode node = iter.next();
                String name = node.get("name").asText();
                String ts = node.get("ts").asText();
                String dur = "";
                String pid = "";
                if(node.get("dur") != null) {
                    dur = node.get("dur").asText();
                }
                if(node.get("pid") != null) {
                    pid = node.get("pid").asText();
                }

                if (trace.getType().equals("cpu") && Double.parseDouble(ts) >= endTime) {
                    firstLayerNodes.add(node);
                    endTime = Double.parseDouble(ts) + Double.parseDouble(dur);
                    cpuMap.put(name, cpuMap.getOrDefault(name, 0.0) + Double.parseDouble(dur));
                    cpuCount.put(name, cpuCount.getOrDefault(name, 0) + 1);
                }

                if (trace.getType().equals("gpu") && !dur.equals("") && pid.equals("CUDA functions") && Double.parseDouble(ts) >= endTime) {
                    firstLayerNodes.add(node);
                    endTime = Double.parseDouble(ts) + Double.parseDouble(dur);
                    gpuMap.put(name, gpuMap.getOrDefault(name, 0.0) + Double.parseDouble(dur));
                    gpuCount.put(name, gpuCount.getOrDefault(name, 0) + 1);
                }


            }


            if (trace.getType().equals("gpu")) {
                System.out.println("end at: " + endTime);
            }

            int count = 0;
            FileOutputStream fos = new FileOutputStream(new File("./" + trace.getName() + ".json"));

            fos.write("[".getBytes());
            for(int i = 0; i < firstLayerNodes.size(); i++) {
                if(i > 0) {
                    fos.write(",".getBytes());
                }
                fos.write((firstLayerNodes.get(i) + "").getBytes());
            }
            fos.write("]".getBytes());
            fos.flush();
            fos.close();

            if(trace.getType().equals("cpu")) {
                System.out.println(trace.getType());
                for(String key : cpuMap.keySet()) {
                    System.out.println(key + " , " + cpuMap.get(key) + " , " + cpuMap.get(key) / cpuCount.get(key));
                }
            }

            else if (trace.getType().equals("gpu")) {
                System.out.println(trace.getType());
                for(String key : gpuMap.keySet()) {
                    System.out.println(key + " , " + gpuMap.get(key) + " , " + gpuMap.get(key) / gpuCount.get(key));
                }
            }



            System.out.println(firstLayerNodes.size());
        }


        FileOutputStream fos = new FileOutputStream(new File(outputFile));
//        fos.write("operator, speedup, cpu_tol_time, gpu_tol_time, cpu_avg_time, gpu_avg_time\n".getBytes());
        for(String key : cpuMap.keySet()) {
            if (gpuMap.containsKey(key)) {
                speedup.put(key, cpuMap.get(key) * gpuCount.get(key) / (cpuCount.get(key) * gpuMap.get(key)));
            }
        }
//
//
//
        List<Map.Entry<String, Double>> list = new ArrayList<>(speedup.entrySet());

        list.sort(new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                if (o1.getValue() < o2.getValue()) {
                    return -1;
                }
                else if (o1.getValue().equals(o2.getValue())) {
                    return 0;
                }

                return 1;
            }
        });
        ArrayNode root = new ObjectMapper().createArrayNode();
        for(Map.Entry<String, Double> e : list) {
            ObjectNode node = new ObjectMapper().createObjectNode();
            String key = e.getKey();
            node.put(key, cpuMap.get(key) * gpuCount.get(key) / (cpuCount.get(key) * gpuMap.get(key)));
            root.add(node);
        }
        fos.write(root.toString().getBytes());
        fos.flush();
        fos.close();

//        for(Map.Entry<String, Double> e : list) {
//            String key = e.getKey();
//            fos.write((key + ", " + // operator
//                    cpuMap.get(key) * gpuCount.get(key) / (cpuCount.get(key) * gpuMap.get(key)) + ", " + // speedup
//                    cpuMap.get(key) + ", " +
//                    gpuMap.get(key) + ", " +
//                    cpuMap.get(key) / cpuCount.get(key) + ", " +
//                    gpuMap.get(key) / gpuCount.get(key) + "\n")
//                    .getBytes());
//        }
//        fos.flush();
//        fos.close();



    }
}
