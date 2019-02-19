from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------

from sdas.core.client.SDASClient import SDASClient
from sdas.core.SDAStime import TimeStamp

client = SDASClient('baco.ipfn.tecnico.ulisboa.pt', 8888)

def get_data(shot, channel):
    info = client.getData(channel, '0x0000', shot)
    data = info[0].getData()
    t0 = TimeStamp(tstamp=info[0]['events'][0]['tstamp']).getTimeInMicros()
    t1 = info[0].getTStart().getTimeInMicros()
    t2 = info[0].getTEnd().getTimeInMicros()
    dt = float(t2-t1)/float(len(data))
    time = np.arange(t1-t0, t2-t0, dt, dtype=data.dtype)*1e-6
    return data, time

# -------------------------------------------------------------------------

#0, 0cm 
shot = 43649
# #0, 3cm
# shot =  43650
# #0, 5cm 
#shot = 43653
# shot = 43655

# #90, 5cm
# shot = 43659
# #90, 7cm
# shot = 43657

# #180, 5cm 
# shot = 43660

# #270, 5cm
# shot = 43661 

# #315, 5cm 
# shot = 43662

# #45, 5cm
# shot = 43663

# #135, 5cm
# shot = 43664

# #225o, 5cm
# shot = 43665

shot = 43872


channels = [
    # top camera
    # 'MARTE_NODE_TOMO.DataCollection.Channel_011',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_012',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_009',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_014',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_015',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_008',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_010',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_013',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_005',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_004',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_007',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_003',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_001',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_000',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_006',
    # 'MARTE_NODE_TOMO.DataCollection.Channel_002',
    'MARTE_NODE_TOMO.DataCollection.Channel_012',
    'MARTE_NODE_TOMO.DataCollection.Channel_011',
    'MARTE_NODE_TOMO.DataCollection.Channel_014',
    'MARTE_NODE_TOMO.DataCollection.Channel_009',
    'MARTE_NODE_TOMO.DataCollection.Channel_008',
    'MARTE_NODE_TOMO.DataCollection.Channel_015',
    'MARTE_NODE_TOMO.DataCollection.Channel_013',
    'MARTE_NODE_TOMO.DataCollection.Channel_010',
    'MARTE_NODE_TOMO.DataCollection.Channel_004',
    'MARTE_NODE_TOMO.DataCollection.Channel_005',
    'MARTE_NODE_TOMO.DataCollection.Channel_003',
    'MARTE_NODE_TOMO.DataCollection.Channel_007',
    'MARTE_NODE_TOMO.DataCollection.Channel_000',
    'MARTE_NODE_TOMO.DataCollection.Channel_001',
    'MARTE_NODE_TOMO.DataCollection.Channel_002',
    'MARTE_NODE_TOMO.DataCollection.Channel_006',
    # front camera
    'MARTE_NODE_TOMO.DataCollection.Channel_028',
    'MARTE_NODE_TOMO.DataCollection.Channel_027',
    'MARTE_NODE_TOMO.DataCollection.Channel_030',
    'MARTE_NODE_TOMO.DataCollection.Channel_025',
    'MARTE_NODE_TOMO.DataCollection.Channel_024',
    'MARTE_NODE_TOMO.DataCollection.Channel_031',
    'MARTE_NODE_TOMO.DataCollection.Channel_029',
    'MARTE_NODE_TOMO.DataCollection.Channel_026',
    'MARTE_NODE_TOMO.DataCollection.Channel_020',
    'MARTE_NODE_TOMO.DataCollection.Channel_021',
    'MARTE_NODE_TOMO.DataCollection.Channel_019',
    'MARTE_NODE_TOMO.DataCollection.Channel_023',
    'MARTE_NODE_TOMO.DataCollection.Channel_016',
    'MARTE_NODE_TOMO.DataCollection.Channel_017',
    'MARTE_NODE_TOMO.DataCollection.Channel_018',
    'MARTE_NODE_TOMO.DataCollection.Channel_022',]


signals_data = []
signals_time = []

for channel in channels:
    print('channel:', channel)
    data, time = get_data(shot, channel)
    i0 = np.argmin(np.fabs(time))
    data -= np.mean(data[:i0])
    n = 200
    data = np.cumsum(data, axis=0)
    data = (data[n:]-data[:-n])/n
    data = data[::n]
    data = np.clip(data, 0., None)
    time = time[n//2::n]
    time = time[:data.shape[0]]
    signals_data.append(data)
    signals_time.append(time)
    plt.plot(time, data,label = channel[-3:])
    print('max: ', np.max(data))
    if channel == channels[15]:
        plt.title('signals (top camera)')
        plt.xlabel('t (s)')
        plt.legend()
        plt.savefig('./images/signals-top.png',dpi = 300)
        plt.show()
    if channel == channels[31]:
        plt.title('signals (front camera)')
        plt.xlabel('t (s)')
        plt.legend()
        plt.savefig('./images/signals-front.png',dpi = 300)
        plt.show()

# -------------------------------------------------------------------------

signals_data = np.array(signals_data)
signals_time = np.array(signals_time)

print('signals_data:', signals_data.shape, signals_data.dtype)
print('signals_time:', signals_time.shape, signals_time.dtype)

# -------------------------------------------------------------------------

fname = 'signals_data.npy'
print('Writing:', fname)
np.save(fname, signals_data)

fname = 'signals_time.npy'
print('Writing:', fname)
np.save(fname, signals_time)
