/**
 * Copyright (2019, ) Institute of Software, Chinese Academy of Sciences
 */
package tools;

import java.io.File;
import java.util.Iterator;

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
public class TFAnalyzer {
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

    public static final String CREATE_TABLE    = "CREATE TABLE #TABLE# (operator_name varchar(1000), dur float);";
    public static final String INSERT_INTO_TABLE   = "INSERT INTO #TABLE# VALUES (#OP_NAME#, #DUR#)";
    public static final String CREATE_MERGE_TABLE =
            "CREATE TABLE #MERGE_TABLE# " +
                    "(SELECT #CPU_TABLE#.operator_name, AVG(#CPU_TABLE#.dur) / AVG(#GPU_TABLE#.dur) AS speedup,  AVG(#CPU_TABLE#.dur) AS cpu_dur, AVG(#GPU_TABLE#.dur) AS gpu_dur FROM #CPU_TABLE#, #GPU_TABLE# WHERE #CPU_TABLE#.operator_name = #GPU_TABLE#.operator_name " +
                    "GROUP BY operator_name ORDER BY speedup DESC)";

    public static final String DEFAULT_MYSQL_ADDRESS = "jdbc:mysql://localhost:32769";
    public static final String DEFAULT_DATABASE = "speedup";

    public static final String DEFAULT_USER = "root";
    public static final String DEFAULT_PASSWORD = "onceas2020";


    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            System.out.println("usage: -cpu [cpu trace path] -gpu [gpu trace path] -merge [table name]");
            return;
        }
        Trace[] traces = new Trace[]{new Trace("cpu", args[1]), new Trace("gpu", args[3])};

        String mergeTableName = args[5];

        SqlClient client = new SqlClient(SqlUtils.createConnection(Driver.class.getName(),
                DEFAULT_MYSQL_ADDRESS, DEFAULT_USER, DEFAULT_PASSWORD), DEFAULT_DATABASE);

        for (Trace trace : traces) {
            String path = "./" + trace.getFilePath() + "/train/plugins/profile/";
            path += new File(path).list()[0];

            for (String fileName : new File(path).list()) {
                if (fileName.endsWith(".gz")) {
                    path += "/" + fileName;
                }
            }
            System.out.println(path);

            GzipHelper gzip = new GzipHelper();
            gzip.deCompress(path, path.replace(".gz", ""));

            path = path.replace(".gz", "");
            System.out.println(path);

            JsonNode dataSet = new ObjectMapper().readTree(
                    new File(path)).get("traceEvents");

            Iterator<JsonNode> iter =  dataSet.iterator();

            String tableName = trace.getFilePath();

            client.exec(DEFAULT_DATABASE, CREATE_TABLE.replace(LABEL_TABLE, tableName));
            client.getConn().checkState();
            while (iter.hasNext()) {
                JsonNode node = iter.next();
                // To insert
                String operatorName = "";
                String dur = "";

                JsonNode durNode = node.get("dur");
                if (durNode != null) {
                    dur = durNode.toPrettyString();
                }

                if (trace.getType() == "cpu") {
                    JsonNode nameNode = node.get("name");
                    if (nameNode != null){
                        operatorName = nameNode.toPrettyString();
                    }
                } else if (trace.getType() == "gpu") {
                    JsonNode argsNode = node.get("args");
                    if (argsNode != null) {
                        // GPU
                        JsonNode longNameNode = argsNode.get("long_name");
                        JsonNode annotationNode = argsNode.get("annotation");
                        if (longNameNode != null) {
                            operatorName = longNameNode.toPrettyString();
                        }
                        if (annotationNode != null) {
                            operatorName = annotationNode.toPrettyString();
                        }
                    }
                }

                if (operatorName != "" && dur != "") {
                    client.exec(DEFAULT_DATABASE, INSERT_INTO_TABLE
                            .replace(LABEL_TABLE, tableName)
                            .replace(LABEL_OPERATOR_NAME, operatorName)
                            .replace(LABEL_DURATION, dur));
                }
            }
        }
        client.exec(DEFAULT_DATABASE, CREATE_MERGE_TABLE
                .replace(LABEL_MERGE_TABLE, mergeTableName)
                .replace(LABEL_CPU_TABLE, traces[0].getFilePath())
                .replace(LABEL_GPU_TABLE, traces[1].getFilePath()));
    }
}
