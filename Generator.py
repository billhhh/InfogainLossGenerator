import sys
import caffe
import numpy as np

# param setting
total_N = 78468.0 # total number of samples
L = 2 # dimentions

def main():
    p_N = sys.argv[1]
    print("Positive num: " + p_N);
    print("Total num: " + str(total_N));

    H = np.eye(L, dtype='f4')
    H[0, 0] = total_N/float(p_N);
    H[1, 1] = total_N / (total_N-float(p_N));

    blob = caffe.io.array_to_blobproto(H.reshape((1, 1, L, L)))
    with open('infogainH.binaryproto', 'wb') as f:
        f.write(blob.SerializeToString())

if __name__ == "__main__":
    main()
