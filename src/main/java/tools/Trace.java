package tools;

public class Trace {
    public String type;
    public String filePath;
    Trace(String type, String filePath) {
        this.type = type;
        this.filePath = filePath;
    }
    public String getType() {
        return type;
    }
    public String getFilePath() {
        return filePath;
    }
}