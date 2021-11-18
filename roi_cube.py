from  useful_class import DataLoading


for i in range(1,5):
    data_name = 'g_'+str(i)
    data_path = r'D:\Solar panels\20211026\\'
    save_path = r"D:\Solar panels\roi\\"
    
    DataLoading.grab_sample_cube(data_name, data_path, save_path)






