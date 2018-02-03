import h5py
import numpy as np

def string_to_array(array,dtype='f4'):
    arr = np.array(array,dtype=dtype)
    return arr.astype(dtype)


def to_hd5f(input_fname,output_hd5f,nd):
    labels = []
    
    with h5py.File(output_hd5f,'w') as foo:
        ds = foo.create_dataset('d100',shape=(2,100),maxshape=(None,100),dtype='f4')
        with open(input_fname,'r') as fii:
            for i,l in enumerate(fii.readlines()):
                if i%500 == 0:
                    print "loop",i
                if i == ds.shape[0]:
                    ds.resize(ds.shape[0]*ds.shape[0],axis=0)
                s = l.split()
                #labels.append((' '.join(format(ord(x), 'b') for x in s[0]),i))
                labels.append(s[0])
                arr =  string_to_array(s[1:])
                #print arr
                ds[i] = arr
                #if i > 10000:
                #    break
        print "Adding header"
        print "X", max(len(x) for x in labels)
        dl = foo.create_dataset('d100_label',shape=(len(labels),),dtype='S20')
        dl[...] = np.array(labels)

        #print labels
        print ds

def show_ds(fname):
    with h5py.File(fname,'r') as fii:
        print fii.keys()
        print fii['d100']
        print type(fii['d100'][0][0])
        print fii['d100_label']


if __name__ == '__main__':
    fname = '/Volumes/MacHD/DeepLearning/GloVe_WordVectors/glove.6B.100d.txt'
    to_hd5f(fname,'test.h5')
    show_ds('test.h5')
