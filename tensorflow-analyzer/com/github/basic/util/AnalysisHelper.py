import  os

basepath = "/tmp/TFAnalysisResult/"

def pathTest():
    print(os.getcwd()) #获取当前工作目录路径
    print(os.path.abspath('.')) #获取当前工作目录路径

def get_abs_base_path(sub_dir,file_name):
    return basepath + sub_dir+"/"+file_name