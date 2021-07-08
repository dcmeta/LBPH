import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os

class LBPH():
    def __init__(self,kernel=3,x_grid=8,y_grid=8,basewidth=120,pad_size=2):
        self.kernel = kernel
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.pad_size = pad_size
        self.basewidth = basewidth
        self.x_trained_hist = []
        self.y_trained_hist = []
        self.label_map = {}
        self.dist_list = {}
    def resize(self, img):
        wpercent = (self.basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((self.basewidth,hsize), Image.ANTIALIAS)
        return img
    def padding(self,img):
        pad_img = np.zeros((int(img.shape[0])+self.pad_size, int(img.shape[1])+self.pad_size))
        pad_img[1:img.shape[0]+1,1:img.shape[1]+1] = img
        return pad_img
    def get_binary_lbp(self,pad_img):
        temp_kernel = np.copy(pad_img)
        threshold = temp_kernel[self.kernel//2,self.kernel//2]
        new_kernel = temp_kernel.flatten()
        new_kernel[new_kernel>=threshold]="1"
        new_kernel[new_kernel!=1]="0"
        new_kernel = np.delete(new_kernel,(self.kernel*self.kernel)//2)
        new_kernel = new_kernel.astype(int)
        binary = ''.join(new_kernel.astype(str))
        binary = int(binary, 2)
        return binary
    def conv_to_hist(self,img,new_img):
        x_grid_range = np.linspace(0,img.shape[1]-1,self.x_grid)
        y_grid_range = np.linspace(0,img.shape[0]-1,self.y_grid)
        x_grid_range = np.delete(x_grid_range,0)
        y_grid_range = np.delete(y_grid_range,0)
        histogram = np.zeros((self.x_grid,self.y_grid,256))
        x_before,y_before=0,0
        i,j=0,0
        for x in x_grid_range:
            j=0
            x = math.floor(x)
            for y in y_grid_range:
                y = math.floor(y)
                flatten_array = new_img[x_before:x+1,y_before:y+1].flatten()
                for ar in flatten_array:
                    histogram[i,j,math.floor(ar)]+=1
                j+=1
                y_before = y
            i+=1
            x_before = x
        return histogram
    def save_hist_csv(self,hist,label,file_name):
        path = 'temp/'+label+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(path+file_name+".csv", hist.flatten(), delimiter=",", fmt="%s")
        
    def calculate_lbp(self,img_path):
        img = Image.open(img_path).convert('L')
        img = self.resize(img)
        img = np.asarray(img)
        pad_img = self.padding(img)
        new_img = np.zeros((img.shape[0],img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if j < pad_img.shape[1]-self.pad_size and i < pad_img.shape[0]-self.pad_size:
                    binary = self.get_binary_lbp(pad_img[i:i+3,j:j+3])
                    new_img[i,j]=binary
        hist = self.conv_to_hist(img,new_img)
        return hist
    def calculate_lbp_for_prediction(self,img):
        img = self.resize(img)
        img = np.asarray(img)
        pad_img = self.padding(img)
        new_img = np.zeros((img.shape[0],img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if j < pad_img.shape[1]-self.pad_size and i < pad_img.shape[0]-self.pad_size:
                    binary = self.get_binary_lbp(pad_img[i:i+3,j:j+3])
                    new_img[i,j]=binary
        hist = self.conv_to_hist(img,new_img)
        return hist
    def train_lbp(self,img_path,label,file_name):
        hist = self.calculate_lbp(img_path)
        self.save_hist_csv(hist,label,file_name)
    def validate_dataset(self,input_path):
        print("Validate Dataset")
        count=0
        for _, dirs, files in os.walk(input_path):
            for dir2 in dirs:
                for _, _, files2 in os.walk(input_path+dir2+"/"):
                    for f in files2:
                        if f != ".DS_Store":
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                                count+=1
                            else:
                                return "Dataset is not valid! There is a file which is not an image"
        return count
                            
    def train(self,input_path):
        count = 0
        total = str(self.validate_dataset(input_path))
        loading_bar = ""
        print("Total images: "+total)
        for _, dirs, files in os.walk(input_path):
            for dir2 in dirs:
                for _, _, files2 in os.walk(input_path+dir2+"/"):
                    for f in files2:
                        if f != ".DS_Store":
                            count+=1
                            prog=(count/int(total))*50
                            loading_bar=""
                            for _ in range(math.floor(prog)): loading_bar+="="
                            for _ in range(50-math.floor(prog)) : loading_bar+=" "
                            print('Progress: [%s] %s%%' % (str(loading_bar), round(prog*2,2)),end="\r")
                            f_name = f.split('.')[0]
                            self.train_lbp(input_path+"/"+dir2+"/"+f,dir2,f_name)
        self.load_trained_data()
    
    def load_trained_data(self):
        print("Load Dataset...")
        path = 'temp/'
        self.x_trained_hist = []
        self.y_trained_hist = []
        for _, dirs, _ in os.walk(path):
            for dir2 in dirs:
                for _, _, files2 in os.walk(path+"/"+dir2+"/"):
                    for f in files2:
                        data = np.loadtxt(path+"/"+dir2+"/"+f, delimiter=",")
                        self.x_trained_hist.append(data)
                        self.y_trained_hist.append(f)
                        self.label_map[f] = dir2
    
    def predict_label(self,x_hist):
        self.dist_list = {}
        for x in range(len(self.x_trained_hist)):
            self.dist_list[x] = np.linalg.norm(x_hist - self.x_trained_hist[x])
        minimum = min([z for k,z in self.dist_list.items()])
        for k,v in self.dist_list.items():
            if v == minimum:
                label_id = self.y_trained_hist[k]
                break
#         neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', metric="euclidean")
#         neigh.fit(self.x_trained_hist, self.y_trained_hist
        res = self.label_map[label_id]
        return res,label_id
       
    def calculate_confident(self,x_hist,label_id,label):
        path = 'temp/'
        maximum = max([z for k,z in self.dist_list.items()])
        data = np.loadtxt(path+label+"/"+label_id, delimiter=",")
        dist = np.linalg.norm(x_hist - data)
        confident = (1 - (dist/maximum)) * 100
        return confident
    def predict(self,img):
        x_hist = self.calculate_lbp_for_prediction(img)
        x_hist = x_hist.flatten()
        res,label_id = self.predict_label(x_hist)
        dist = self.calculate_confident(x_hist,label_id,res)
        return res,dist
