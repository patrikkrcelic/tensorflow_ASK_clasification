import numpy as np
import matplotlib.pyplot as plt

## Add datadirectory with data_ids and labeles
lutfile=''

Ndata=3032

f=open(lutfile,'r')

bask=[' ', ' ', ' ', ' ', ' ', ' ']
frame=[' ']
cl=np.zeros((Ndata, 5))

br1=0;

for line in f:
    inner_list = [elt.strip() for elt in line.split(' ')]
    br2=-1
    if br1>=Ndata-1:
        break
    for s in inner_list:
        if s != '':
            if br2 == -1:
                bask=s
                br2=br2+1
            else:
                cl[br1,br2]=int(s)
                br2=br2+1

    frame.append(bask)
    br1=br1+1

frame=frame[1:-1]

f.close()

shuffle_ix = np.arange(0,Ndata-2)
np.random.shuffle(shuffle_ix)


cl=np.array(cl)[shuffle_ix,:]
frame=np.array(frame)[shuffle_ix]

br1=0
br2=1
br3=1


f=open('test_multi_list.txt','w')

for fr in frame:
    line=frame[br1]+' '+str(cl[br1,0]) + ' ' +str(cl[br1,1]) + ' '+str(cl[br1,2]) + ' '+str(cl[br1,3]) + ' '+str(cl[br1,4])
    if br1<1000:
        f.write(line)
        f.write('\n')
    elif br2>1000:
        f.close()
        imfil='imfil_multi_list_'+str(br3)+'.txt'
        f=open(imfil, 'w')
        f.write(line)
        f.write('\n')
        br2=1
        br3=br3+1
    else:
        f.write(line)
        f.write('\n')
    br2=br2+1
    br1=br1+1

f.close()
