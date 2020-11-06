public class DataReader {
    private Double[] gpuTask_gpu;
    private Double[] gpuTask_cpu;
    private Double oldGPUTasktime = 0.0;
    private Double oldCPUTasktime = 0.0;

    public Double getTotalImmigrationTime() {
        return totalImmigrationTime;
    }

    public void setTotalImmigrationTime(Double totalImmigrationTime) {
        this.totalImmigrationTime = totalImmigrationTime;
    }

    private Double totalImmigrationTime = 0.0;

    public Double getOldAllGPUtime() {
        return oldAllGPUtime;
    }

    public void setOldAllGPUtime(Double oldAllGPUtime) {
        this.oldAllGPUtime = oldAllGPUtime;
    }

    private Double oldAllGPUtime = 0.0;
    private Double[] cpuTask_gpu;
    private Double[] cpuTask_cpu;

    public Double getOldGPUTasktime() {
        return oldGPUTasktime;
    }

    public void setOldGPUTasktime(Double oldGPUTasktime) {
        this.oldGPUTasktime = oldGPUTasktime;
    }

    public Double getOldCPUTasktime() {
        return oldCPUTasktime;
    }

    public void setOldCPUTasktime(Double oldCPUTasktime) {
        this.oldCPUTasktime = oldCPUTasktime;
    }

    public DataReader(Double[] gpuTask_gpu, Double[] gpuTask_cpu, Double[] cpuTask_gpu, Double[] cpuTask_cpu) {
        this.gpuTask_gpu = gpuTask_gpu;
        this.gpuTask_cpu = gpuTask_cpu;
        this.cpuTask_gpu = cpuTask_gpu;
        this.cpuTask_cpu = cpuTask_cpu;
    }

    public DataReader(){
        this.test();
    }

    public void test(){
        this.gpuTask_gpu = new Double[]{14400.625, 980.375, 2705.5, 3795.125, 3429.0, 2738.125, 3102.25, 1150.5, 3086.5, 4467.5, 1889.625, 6272.75, 13034.25, 7132.375, 7831.625, 3154.875, 1799.5, 3052.125, 1241.125, 1261.375, 3585.875, 1513.875, 1927.125, 1128.5, 1198.0};
        this.gpuTask_cpu = new Double[]{34685.658999999985, 10297.23999999999, 56478.840000000084, 35559.40699999989, 32904.36100000003, 26197.219999999972, 24296.91100000008, 12243.158999999985, 22724.962000000058, 40544.77799999993, 14678.645000000019, 54940.75099999993, 128807.34100000001, 71074.39800000004, 32484.459000000264, 14318.176999999676, 10904.990000000224, 17146.989999999758, 17949.21299999999, 17458.149999999907, 19933.05000000028, 15219.518999999855, 18367.851999999955, 10056.763000000268, 13484.898000000045};
        this.cpuTask_gpu = new Double[]{2757.625, 2643.0, 3518.375, 3194.625, 2568.125, 1955.375, 4155.625, 4362.75, 2303.5, 3738.25, 2607.0, 3164.25, 3058.875, 2538.25, 2899.875, 4952.25, 3193.75, 9542.625, 3616.5, 2154.75, 3280.875, 4333.625, 4757.625, 4390.875, 5332.5, 4602.0, 5290.25, 2322.375, 6645.75, 12137.75, 1561.375, 2707.125, 4714.375, 12091.875, 1536.0, 2682.875, 18910.375, 4295.5, 8097.75};
        this.cpuTask_cpu = new Double[]{40382.513000000035, 21304.403999999864, 19083.597999999998, 24550.94299999997, 13655.780000000028, 16578.079000000143, 25052.267999999924, 18487.17299999995, 12725.523999999976, 13878.533999999985, 11819.529000000097, 10933.180999999866, 10127.34300000011, 13077.64599999995, 11002.780999999959, 37838.97400000016, 12841.154000000097, 62044.53500000015, 17766.768999999855, 10013.560000000056, 15741.79600000009, 17428.429000000004, 16868.668000000063, 14326.628999999724, 19866.20200000005, 31860.30299999984, 19791.48400000017, 10146.268000000156, 26901.899999999907, 38124.63899999997, 10645.479999999981, 21688.709999999963, 44894.90599999996, 38511.91200000001, 10756.691000000108, 22059.66000000015, 99398.71099999966, 43184.57800000021, 10041.550999999978};
//        this.gpuTask_gpu = new Double[]{1.0,1.0,1.0,1.0};
//        this.gpuTask_cpu = new Double[]{1.0,2.0,2.0,5.0};
//        this.cpuTask_cpu = new Double[]{1.0,3.0,3.0,2.0};
//        this.cpuTask_gpu = new Double[]{1.0,1.0,1.0,1.0};
        for(Double d : this.gpuTask_gpu){
            this.oldAllGPUtime+=d;
            this.oldGPUTasktime+=d;
        }
        this.oldAllGPUtime*=2;
        for(Double d: this.cpuTask_gpu){
            this.oldAllGPUtime+=d;
        }
        for(Double d: this.cpuTask_cpu){
            this.oldCPUTasktime+=d;
        }
        this.totalImmigrationTime = this.oldGPUTasktime;
        double temp = 0.0;
        int i = 0;
        for(i= 0;i<this.cpuTask_cpu.length;i++){
            temp+=this.cpuTask_cpu[i];
            if(temp > this.oldGPUTasktime){
                System.out.println(i+"iiiii,"+this.cpuTask_cpu.length);
                break;
            }
        }
        for(int j = i;j<this.cpuTask_gpu.length;j++){
            temp+=this.cpuTask_gpu[j];
        }
        this.totalImmigrationTime += temp;
    }

    public Double[] getGpuTask_gpu() {
        return gpuTask_gpu;
    }

    public void setGpuTask_gpu(Double[] gpuTask_gpu) {
        this.gpuTask_gpu = gpuTask_gpu;
    }

    public Double[] getGpuTask_cpu() {
        return gpuTask_cpu;
    }

    public void setGpuTask_cpu(Double[] gpuTask_cpu) {
        this.gpuTask_cpu = gpuTask_cpu;
    }

    public Double[] getCpuTask_gpu() {
        return cpuTask_gpu;
    }

    public void setCpuTask_gpu(Double[] cpuTask_gpu) {
        this.cpuTask_gpu = cpuTask_gpu;
    }

    public Double[] getCpuTask_cpu() {
        return cpuTask_cpu;
    }

    public void setCpuTask_cpu(Double[] cpuTask_cpu) {
        this.cpuTask_cpu = cpuTask_cpu;
    }
}
