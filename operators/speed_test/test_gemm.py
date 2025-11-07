import pandas as pd
import os
import sys
import subprocess
from tqdm import tqdm
stuck_shape_list =[
    # M, K, N, transA, transB
    (8, 4096, 1024, 0, 0),
    (8, 4096, 7680, 0, 0),
    (2048, 4096, 1024, 0, 0),
    (2048, 4096, 7680, 0, 0),
    (8, 4096, 1024, 0, 1),
    (8, 4096, 7680, 0, 1),
    (2048, 4096, 1024, 0, 1),
    (2048, 4096, 7680, 0, 1),
    (8, 4096, 1024, 1, 1),
    (8, 4096, 7680, 1, 1),
    (2048, 4096, 1024, 1, 1),
    (2048, 4096, 7680, 1, 1),
]

if __name__ == '__main__':
    fname = sys.argv[1]
    datatype = int(sys.argv[2])
    validate = bool(int(sys.argv[3]))

    df = pd.read_csv(fname)

    if not validate:
        df['time'] = None
        df['score'] = None


    for i in tqdm(df.index):
        m = df['M'][i]
        k = df['K'][i]
        n = df['N'][i]
        trans1 = df['transA'][i]
        trans2 = df['transB'][i]
        print(m, k, n, trans1, trans2)
        if (m, k, n, trans1, trans2) in stuck_shape_list:
            continue
        out = subprocess.check_output(
            ['./dlc_ops/test_deeplink_mm', str(m), str(k), str(n), str(trans1), str(trans2), str(datatype)])
        try:
            time = float(out.decode())
        except ValueError:
            raise ValueError('Failed to decode the output.')
        time = format(time / 1000, '.3f')
        if not validate:
            df.at[i, 'baseline'] = time
        else:
            df.at[i, 'time'] = time
            df.at[i, 'score'] = round(
                float(df.at[i, 'baseline']) / float(time), 2)
            print(f"Score: {df.at[i, 'score']}, Time: {time}")

    df.to_csv(fname, index=False)
