package tools;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

public class GzipHelper {


    public void deCompress(String gzipFile, String outputFile) {
        try {
            System.out.println(gzipFile  +"," + outputFile);
            byte[] buffer = new byte[1024];
            FileInputStream fis = new FileInputStream(gzipFile);
            GZIPInputStream gis = new GZIPInputStream(fis);
            FileOutputStream fos = new FileOutputStream(outputFile);
            int size;
            while((size = gis.read(buffer)) > 0) {
                System.out.println(size);
                fos.write(buffer, 0, size);
            }
            fis.close();
            gis.close();
            fos.close();


        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static void main(String args[]) {
        GzipHelper helper = new GzipHelper();
        helper.deCompress("./keras-cpu/train/plugins/profile/2020_07_04_10_43_52iZ2ze86eplnjdkjfil6oahZ.trace.json.gz",
                "./keras-cpu/train/plugins/profile/2020_07_04_10_43_52iZ2ze86eplnjdkjfil6oahZ.trace.json");
    }
}

