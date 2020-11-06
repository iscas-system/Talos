import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class SpeedupResolver {
    private Double oldRuntime = 0.0;
    private Double gpuWaittime = 0.0;
    private Double cpuWaittime = 0.0;
    private Double newTaskstime = 0.0;

    public class RencentOperation{
        public int cpuI;
        public Double alreadyTime;

        public RencentOperation(int cpuI, Double alreadyTime) {
            this.cpuI = cpuI;
            this.alreadyTime = alreadyTime;
        }
    }

    private List<Span> newCPUOperations =new ArrayList<Span>();
    private List<Span> newGPUOperations =new ArrayList<Span>();

    private DataReader dr;

    public SpeedupResolver(DataReader dr) {
        this.dr = dr;
        this.newTaskstime = this.dr.getOldCPUTasktime()+this.dr.getOldGPUTasktime();
        this.oldRuntime = this.dr.getOldCPUTasktime()+this.dr.getOldGPUTasktime();
    }

    public RencentOperation finRecentCPUOperations(Double oGpuStartpoint,int gpuI){
        int cpuI = 0;
        double alreadyCPUMixTIme =0.0;
        for(Span temp1 : newCPUOperations){
            if(temp1.number >= cpuI && temp1.taskName.equals("cpuStask")){
                cpuI = temp1.number;
            }
            if(temp1.taskName.equals("cpuTask")){
                alreadyCPUMixTIme +=temp1.duration;
            }
        }
        for(Span temp1 : newGPUOperations){
            if(temp1.number >= cpuI && temp1.taskName.equals("cpuTask")){
                cpuI = temp1.number;
            }
            if(temp1.taskName.equals("cpuTask")){
                alreadyCPUMixTIme +=temp1.duration;
            }
        }
        if(this.newCPUOperations.size() == 0 && this.newGPUOperations.size() == 0){
            return new RencentOperation(0,0.0);
        }
        if(cpuI+1 < dr.getCpuTask_cpu().length && dr.getCpuTask_cpu()[cpuI+1]+alreadyCPUMixTIme >= oGpuStartpoint){
            if(oGpuStartpoint+this.dr.getGpuTask_gpu()[gpuI] >= alreadyCPUMixTIme+dr.getCpuTask_cpu()[cpuI]){
                this.newCPUOperations.add(new Span(dr.getCpuTask_cpu()[cpuI],alreadyCPUMixTIme,
                        alreadyCPUMixTIme+dr.getCpuTask_cpu()[cpuI],"cpu","cpuTask",cpuI));
                return new RencentOperation(cpuI+1,alreadyCPUMixTIme);
            }
            return new RencentOperation(cpuI,alreadyCPUMixTIme);
        }
        if(cpuI +1 >dr.getCpuTask_gpu().length ){
            System.out.println("at the end of CPUtask");
            return new RencentOperation(cpuI,alreadyCPUMixTIme);
        }
        return new RencentOperation(-1,-1.0);
    }

    public Double getOldRuntime() {
        return oldRuntime;
    }

    public void setOldRuntime(Double oldRuntime) {
        this.oldRuntime = oldRuntime;
    }
    public Double getGpuWaittime() {
        return gpuWaittime;
    }

    public void setGpuWaittime(Double gpuWaittime) {
        this.gpuWaittime = gpuWaittime;
    }

    public Double getCpuWaittime() {
        return cpuWaittime;
    }

    public void setCpuWaittime(Double cpuWaittime) {
        this.cpuWaittime = cpuWaittime;
    }

    public Double getNewTaskstime() {
        return newTaskstime;
    }

    public void setNewTaskstime(Double newTaskstime) {
        this.newTaskstime = newTaskstime;
    }

    public List<Span> getNewCPUOperations() {
        return newCPUOperations;
    }

    public void setNewCPUOperations(List<Span> newCPUOperations) {
        this.newCPUOperations = newCPUOperations;
    }

    public List<Span> getNewGPUOperations() {
        return newGPUOperations;
    }

    public void setNewGPUOperations(List<Span> newGPUOperations) {
        this.newGPUOperations = newGPUOperations;
    }

    public DataReader getDr() {
        return dr;
    }

    public void setDr(DataReader dr) {
        this.dr = dr;
    }

    public void resolve(){
        Double oGpuStartpoint = 0.0;
        int cpuI = 0;
        for(int gpuI = 0;gpuI<dr.getGpuTask_gpu().length;gpuI++){
            System.out.println("gpu operation O"+gpuI+": "+ dr.getGpuTask_cpu()[gpuI] +" / "+ dr.getGpuTask_gpu()[gpuI]);
            Double ifCPUTime = dr.getGpuTask_cpu()[gpuI];
            RencentOperation ro = this.finRecentCPUOperations(oGpuStartpoint,gpuI);
            Double ifGPUTime = 0.0;
            Double cpuTime =0.0;
            cpuI = ro.cpuI;
            if(cpuI == -1){
                continue;
            }
            System.out.println("Related CPU operation: O"+ ro.cpuI+"; already time:"+ ro.alreadyTime);
            for(cpuI = ro.cpuI;cpuI < dr.getCpuTask_gpu().length-1 ;cpuI++){
                System.out.println("CPUI before:"+cpuI);
                ifGPUTime+=dr.getCpuTask_gpu()[cpuI];
                cpuTime+=dr.getCpuTask_cpu()[cpuI];
                if(cpuI == dr.getCpuTask_gpu().length-1 && ifGPUTime+dr.getCpuTask_gpu()[cpuI+1] <= ifCPUTime){
                    cpuI++;
                    cpuTime+=dr.getCpuTask_cpu()[cpuI+1];
                    break;
                }
                if(ifGPUTime <= ifCPUTime && ifGPUTime+dr.getCpuTask_gpu()[cpuI+1] > ifCPUTime){
                    break;
                }
                if(ifGPUTime > ifCPUTime){
                    if(dr.getCpuTask_cpu()[cpuI]+dr.getGpuTask_gpu()[gpuI] > ifGPUTime+ifCPUTime){
                        break;
                    }
                    else{
                        break;
                    }
                }
            }
            System.out.println("CPUI after:"+cpuI);
            System.out.println(ifCPUTime+","+ifGPUTime);
            if(ifCPUTime - ifGPUTime >=0){
                gpuWaittime += (ifCPUTime - ifGPUTime);
            }
            else{
                gpuWaittime +=ifGPUTime-this.dr.getGpuTask_gpu()[gpuI];
            }
            cpuWaittime += oGpuStartpoint - ro.alreadyTime;
            System.out.println(gpuWaittime+","+cpuWaittime);
            Double currentTaskstime = this.dr.getOldCPUTasktime()+this.dr.getOldGPUTasktime() +
                    gpuWaittime+cpuWaittime+(ifCPUTime-dr.getCpuTask_gpu()[gpuI]) - (cpuTime - ifGPUTime);
            System.out.println("currentTaskstime:"+currentTaskstime+", ");
            if(currentTaskstime < this.dr.getOldCPUTasktime()+this.dr.getOldGPUTasktime() && currentTaskstime < this.newTaskstime){
                this.newTaskstime = currentTaskstime;
                if(oGpuStartpoint - ro.alreadyTime >0.0){
                    this.newCPUOperations.add(new Span(oGpuStartpoint - ro.alreadyTime, ro.alreadyTime,
                            oGpuStartpoint,"blank","cpuTask",-1));
                }
                if(oGpuStartpoint - ro.alreadyTime < 0.0){

                }
                if(oGpuStartpoint - ro.alreadyTime == 0.0){

                }
                this.newCPUOperations.add(new Span(ifCPUTime,oGpuStartpoint,ifCPUTime+oGpuStartpoint,"gpu-cpu","gpuTask",gpuI));
                Double cpuToGpuOperationStartPoint = oGpuStartpoint;
                for(int tempI = ro.cpuI;tempI <=cpuI;tempI++){
                    this.newGPUOperations.add(new Span(this.dr.getCpuTask_gpu()[tempI],cpuToGpuOperationStartPoint,
                            cpuToGpuOperationStartPoint+this.dr.getCpuTask_gpu()[tempI],"cpu-gpu","cpuTask",tempI));
                    cpuToGpuOperationStartPoint+=this.dr.getCpuTask_gpu()[tempI];
                }
                if(ifCPUTime - ifGPUTime > 0.0){
                    this.newGPUOperations.add(new Span(ifCPUTime - ifGPUTime,ifGPUTime,ifCPUTime ,"blank","gpuTask",-1));
                }
                if(ifCPUTime - ifGPUTime < 0.0){

                }
                if (ifCPUTime - ifGPUTime == 0.0){

                }
                oGpuStartpoint+=ifCPUTime;
                this.newTaskstime = currentTaskstime;
            }
            else{
                for(int i = 0;i<=cpuI;i++){
                    double startPoint = 0;
                    boolean isScheduled = false;
                    for(Span temp1 : newCPUOperations){
                        if(temp1.number ==i ){
                            isScheduled = true;
                        }
                        if(temp1.taskName.equals("cpuTask")){
                            startPoint+=temp1.duration;
                        }
                    }
                    for(Span temp1 : newGPUOperations){
                        if(temp1.number ==i ){
                            isScheduled = true;
                        }
                        if(temp1.taskName.equals("cpuTask")){
                            startPoint+=temp1.duration;
                        }
                    }
                    if(!isScheduled){
                        this.newCPUOperations.add(new Span(this.dr.getCpuTask_cpu()[i],startPoint,
                                startPoint+this.dr.getCpuTask_cpu()[i],"cpu","cpuTask",i));
                    }
                }
                this.newGPUOperations.add(new Span(dr.getGpuTask_gpu()[gpuI],oGpuStartpoint,dr.getGpuTask_gpu()[gpuI]+oGpuStartpoint,
                        "gpu","gpuTask",gpuI));

                oGpuStartpoint+=dr.getGpuTask_gpu()[gpuI];
            }
        }
    }

    public static void main(String[] args) {
        DataReader dr = new DataReader();
        SpeedupResolver sr = new SpeedupResolver(dr);
        sr.resolve();
        System.out.println("old runtime :"+ sr.getOldRuntime());
        System.out.println("new runtime :"+ sr.getNewTaskstime());
        DecimalFormat df = new DecimalFormat("00.00%");

        System.out.println("Improve percentage:" + df.format((sr.getOldRuntime()-sr.getNewTaskstime())/sr.getOldRuntime()));
        System.out.println("Improve percentage2:" + df.format((dr.getOldAllGPUtime()-sr.getNewTaskstime())/dr.getOldAllGPUtime()));
        System.out.println("Improve percentage3:" + df.format((dr.getTotalImmigrationTime()-sr.getNewTaskstime())/dr.getTotalImmigrationTime()));
    }
}
