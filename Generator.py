import sys
import caffe
import numpy as np

# param setting
total_N = 78468.0 # total number of samples
L = 2 # dimentions

def main():
    p_N = sys.argv[1]
    print("Total num: " + str(total_N));

    H = np.eye(L, dtype='f4')

    fp_N = float(p_N);
    if fp_N < 1:
        # p_N is probs
        H[0, 0] = fp_N;
        H[1, 1] = 1 - fp_N;
    else:
        H[0, 0] = fp_N/total_N;
        H[1, 1] = (total_N-fp_N)/total_N;
		
	print("Positive probs: " + H[0, 0]);
	print("Negative probs: " + H[1, 1]);	

    blob = caffe.io.array_to_blobproto(H.reshape((1, 1, L, L)))
    with open('infogainH.binaryproto', 'wb') as f:
        f.write(blob.SerializeToString())

if __name__ == "__main__":
    main()
