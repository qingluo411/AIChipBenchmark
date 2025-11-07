import pandas as pd
import os
import sys
import subprocess
from tqdm import tqdm
stuck_shape_list = [
    # N, C, H, W, OutC, kw, kh, pw, ph, sw, sh, datatype
    ('512', '3', '224', '224', '64', '7', '7', '3', '3', '2', '2', '16')
]

if __name__=='__main__':
    fname=sys.argv[1]
    datatype=int(sys.argv[2])
    validate=bool(int(sys.argv[3]))

    df=pd.read_csv(fname)

    if not validate:
        df['time'] = None
        df['score'] = None

    for i in tqdm(df.index):
        print(['./dlc_ops/test_deeplink_conv', 
                                     str(df['N'][i]), str(df['C'][i]), 
                                     str(df['H'][i]), str(df['W'][i]),
                                     str(df['OutC'][i]), str(df['kw'][i]),
                                     str(df['kh'][i]), str(df['pw'][i]),
                                     str(df['ph'][i]), str(df['sh'][i]),
                                     str(df['sv'][i]),
                                     str(datatype)])
        out=subprocess.check_output(['./dlc_ops/test_deeplink_conv', 
                                     str(df['N'][i]), str(df['C'][i]), 
                                     str(df['H'][i]), str(df['W'][i]),
                                     str(df['OutC'][i]), str(df['kw'][i]),
                                     str(df['kh'][i]), str(df['pw'][i]),
                                     str(df['ph'][i]), str(df['sh'][i]),
                                     str(df['sv'][i]),
                                     str(datatype)])
        try:
            total_time = float(out.decode())
        except ValueError:
            raise ValueError('Failed to decode the output.')
        total_time = total_time / 1000
        if not validate:
            df.at[i, 'baseline'] = format(total_time, '.3f')
        else:
            df.at[i, 'time'] = total_time
            df.at[i, 'score'] = round(
                float(df.at[i, 'baseline']) / float(total_time), 2)
            print('Score:', df.at[i, 'score'], 'Time:', total_time)

    avg_score = df['score'].mean()

    df.to_csv(fname, index=False)
