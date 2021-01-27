/**
 * Copyright (2019, ) Institute of Software, Chinese Academy of Sciences
 */
package tools;

import java.io.File;
import java.io.FileOutputStream;
import java.util.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import sqlclient.SqlClient;
import sqlclient.SqlUtils;
import com.mysql.cj.jdbc.Driver;

/**
 * @author wuheng@otcaix.iscas.ac.cn
 *
 * @version 2.3.0
 * @since 2020.2.15
 *
 **/
public class TorchAnalyzer {
    /*****************************************************************************************
     *
     * Main
     *
     *****************************************************************************************/

    public static final String LABEL_OPERATOR_NAME = "#OP_NAME#";
    public static final String LABEL_DURATION = "#DUR#";

    public static final String LABEL_TABLE = "#TABLE#";
    public static final String LABEL_CPU_TABLE = "#CPU_TABLE#";
    public static final String LABEL_GPU_TABLE = "#GPU_TABLE#";
    public static final String LABEL_MERGE_TABLE = "#MERGE_TABLE#";

    public static final String CREATE_TABLE = "CREATE TABLE #TABLE# (operator_name varchar(50), dur float)ENGINE=MEMORY DEFAULT CHARSET=utf8mb4;";
    public static final String INSERT_INTO_TABLE = "INSERT INTO #TABLE# VALUES (#OP_NAME#, #DUR#)";
    public static final String CREATE_MERGE_TABLE =
            "CREATE TABLE #MERGE_TABLE# " +
                    "(SELECT #CPU_TABLE#.operator_name, AVG(#CPU_TABLE#.dur) / AVG(#GPU_TABLE#.dur) AS speedup,  AVG(#CPU_TABLE#.dur) AS cpu_dur, AVG(#GPU_TABLE#.dur) AS gpu_dur FROM #CPU_TABLE#, #GPU_TABLE# WHERE #CPU_TABLE#.operator_name = #GPU_TABLE#.operator_name " +
                    "GROUP BY operator_name ORDER BY speedup DESC)";

    public static final String DEFAULT_MYSQL_ADDRESS = "jdbc:mysql://localhost:32769";
    public static final String DEFAULT_DATABASE = "speedup";

    public static final String DEFAULT_USER = "root";
    public static final String DEFAULT_PASSWORD = "yangchen";


    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            System.out.println("usage: -cpu [cpu trace path] -gpu [gpu trace path] -merge [table name]");
            return;
        }
        Trace[] traces = new Trace[]{new Trace("cpu", args[1]), new Trace("gpu", args[3])};

        String mergeFileName = args[5];

        SqlClient client = new SqlClient(SqlUtils.createConnection(Driver.class.getName(),
                DEFAULT_MYSQL_ADDRESS, DEFAULT_USER, DEFAULT_PASSWORD), DEFAULT_DATABASE);


        HashMap<String, Double> cpuMap = new HashMap<>();
        HashMap<String, Double> gpuMap = new HashMap<>();

        HashMap<String, Integer> cpuCount = new HashMap<>();
        HashMap<String, Integer> gpuCount = new HashMap<>();


        HashMap<String, Double> speedup = new HashMap<>();


        for (Trace trace : traces) {
//            String path = "./" + trace.getFilePath() + "/train/plugins/profile/";
//            path += new File(path).list()[0];
//
//            for (String fileName : new File(path).list()) {
//                if (fileName.endsWith(".gz")) {
//                    path += "/" + fileName;
//                }
//            }
//            System.out.println(path);
//
//            GzipHelper gzip = new GzipHelper();
//            gzip.deCompress(path, path.replace(".gz", ""));
//
//            path = path.replace(".gz", "");
//            System.out.println(path);
            String path = "./" + trace.getFilePath();
            JsonNode dataSet = new ObjectMapper().readTree(
                    new File(path));
            Iterator<JsonNode> iter = dataSet.iterator();

            String tableName = trace.getFilePath().split("\\.")[0];
            System.out.println(tableName);
//            System.out.println(dataSet);
            client.exec(DEFAULT_DATABASE, CREATE_TABLE.replace(LABEL_TABLE, tableName));
            client.getConn().checkState();
            long count = 0;
            while (iter.hasNext()) {

                JsonNode node = iter.next();
                String dur = "";
                String name = "";
                String pid = "";

                JsonNode durNode = node.get("dur");
                JsonNode nameNode = node.get("name");
                JsonNode pidNode = node.get("pid");
                if (durNode != null) {
                    dur = durNode.toPrettyString();
                }
                if (nameNode != null) {
                    name = nameNode.toPrettyString();
                }
                if (pidNode != null) {
                    pid = pidNode.toPrettyString();
                }

                if(trace.getType() == "cpu") {
                    System.out.println(name + " " + pid + " " + dur);
                }

                if(trace.getType() == "gpu") {
                    System.out.println(name + " " + pid + " " + dur);
                }

                if(trace.getType().equals("cpu") && !name.equals("") && !dur.equals("") && pid.equals("\"CPU functions\"")) {
                    count++;
                    cpuMap.put(name, cpuMap.getOrDefault(name, 0.0) + Double.parseDouble(dur));
                    cpuCount.put(name, cpuCount.getOrDefault(name, 0) + 1);
                    //                if (name != "" && dur != "") {
//                    client.exec(DEFAULT_DATABASE, INSERT_INTO_TABLE
//                            .replace(LABEL_TABLE, tableName)
//                            .replace(LABEL_OPERATOR_NAME, name)
//                            .replace(LABEL_DURATION, dur));

                }

                if(trace.getType().equals("gpu") && !name.equals("") && !dur.equals("") && pid.equals("\"CUDA functions\"")) {
                    count++;
                    gpuMap.put(name, gpuMap.getOrDefault(name, 0.0) + Double.parseDouble(dur));
                    gpuCount.put(name, gpuCount.getOrDefault(name, 0) + 1);
//                    client.exec(DEFAULT_DATABASE, INSERT_INTO_TABLE
//                            .replace(LABEL_TABLE, tableName)
//                            .replace(LABEL_OPERATOR_NAME, name)
//                            .replace(LABEL_DURATION, dur));
                }


                // To insert
//                String operatorName = "";
//                String dur = "";
//
//                JsonNode durNode = node.get("dur");
//                if (durNode != null) {
//                    dur = durNode.toPrettyString();
//                }
//
//                if (trace.getType() == "cpu") {
//                    JsonNode nameNode = node.get("name");
//                    if (nameNode != null){
//                        operatorName = nameNode.toPrettyString();
//                    }
//                } else if (trace.getType() == "gpu") {
//                    JsonNode argsNode = node.get("args");
//                    if (argsNode != null) {
//                        // GPU
//                        JsonNode longNameNode = argsNode.get("long_name");
//                        JsonNode annotationNode = argsNode.get("annotation");
//                        if (longNameNode != null) {
//                            operatorName = longNameNode.toPrettyString();
//                        }
//                        if (annotationNode != null) {
//                            operatorName = annotationNode.toPrettyString();
//                        }
//                    }
//                }
//
//                if (name != "" && dur != "") {
//                    client.exec(DEFAULT_DATABASE, INSERT_INTO_TABLE
//                            .replace(LABEL_TABLE, tableName)
//                            .replace(LABEL_OPERATOR_NAME, name)
//                            .replace(LABEL_DURATION, dur));
            }
            System.out.println(count);
        }

        cpuMap.replaceAll((k, v) -> cpuMap.get(k) / cpuCount.get(k));

        gpuMap.replaceAll((k, v) -> gpuMap.get(k) / gpuCount.get(k));

        for(String key : gpuMap.keySet()) {
            if(cpuMap.containsKey(key)) {
                speedup.put(key, cpuMap.get(key) / gpuMap.get(key));
            }
        }
        List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(speedup.entrySet()); //转换为list

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
        FileOutputStream fos = new FileOutputStream(new File(mergeFileName));
        fos.write(("operator,speedup,cpu_avg_dur,gpu_avg_dur,cpu_tol_dur,gpu_tol_dur").getBytes());
        fos.write("\n".getBytes());
        for(Map.Entry<String, Double> e : list) {
            fos.write((e.getKey().replaceAll("\"", "") + "," + e.getValue() + "," + cpuMap.get(e.getKey()) + "," + gpuMap.get(e.getKey()) + "," + cpuMap.get(e.getKey()) * cpuCount.get(e.getKey()) + "," + gpuMap.get(e.getKey()) * gpuCount.get(e.getKey())).getBytes());
            fos.write("\n".getBytes());
        }
        fos.flush();
        fos.close();

//        client.exec(DEFAULT_DATABASE, CREATE_MERGE_TABLE
//                .replace(LABEL_MERGE_TABLE, mergeTableName)
//                .replace(LABEL_CPU_TABLE, traces[0].getFilePath().split("\\.")[0])
//                .replace(LABEL_GPU_TABLE, traces[1].getFilePath().split("\\.")[0]));
    }

}
