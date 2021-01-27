package tools;

public class Trace {
    public String type;
    public String filePath;
    public String name;

    Trace(String type, String filePath) {
        this.type = type;
        this.filePath = filePath;
        System.out.println(this.filePath);
        this.name = this.filePath.split("\\.")[0];

    }
    public String getType() {
        return type;
    }
    public String getFilePath() {
        return filePath;
    }
    public String getName() { return name; }
}