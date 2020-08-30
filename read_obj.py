import os
import numpy as np
import json
class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
           
            for line in f:
                line_ = line.split(' ')
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    #vertex_ =np.array ([round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2)])
                    self.vertices.append(vertex)
              
                
                elif line[0] == "f":

                    string = line.replace("//", "/")
                    ##
                    
                    
         
                    face = np.zeros(3)
                    
                    
                    for i in range(3):
                        temp = line_[i+1].split('/')
                        face[i] = int(temp[0])
                    self.faces.append(face)

                    
            f.close()
            self.faces = np.asarray(self.faces).astype('int')
            self.vertices = np.asarray(self.vertices).astype('float')
        except IOError:
            print(".obj file not found.")


def save_obj(pc, edge, filename):
    with open(filename, 'w') as fp:
        for v in pc:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in edge:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



if __name__ == '__main__':
    train_data_path = './data/train/data_info/train_set.json'
    real_train = './train_data.json'
    files = []
    with open(train_data_path,'r') as load_f:
        load_dict = json.load(load_f)
    print('total:', len(load_dict))
    for key in load_dict:
        path = './data/train/model/' + key['model'] +'.obj'
        print('........................')
        print(key['model'])

        obj = ObjLoader(path)
        if obj.vertices.shape[0]>2048:
            files.append(key)
    print("available data:", len(files))
    json_data = json.dumps(files)
    with open(real_train,'w') as f:
        f.write(json_data)


