public class Span {
    public Double duration;
    public Double startPoint;
    public Double endPoint;
    public int number;
    public String sType;
    public String taskName;

    @Override
    public String toString() {
        return "Span{" +
                "duration=" + duration +
                ", startPoint=" + startPoint +
                ", endPoint=" + endPoint +
                ", number=" + number +
                ", sType='" + sType + '\'' +
                ", taskName='" + taskName + '\'' +
                '}';
    }

    public Span(double duration, double startPoint, double endPoint, String sType, String taskName, int number) {
        this.duration = duration;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.sType = sType;
        this.taskName = taskName;
        this.number = number;
        System.out.println(this);
    }

    public Span(){

    }
}
