import numpy as np
import neurolab as nl
import matplotlib.pyplot as mp

charset='omandig'
x,y = [],[]
with open('/Users/liwenbo/Desktop/人工智能/day01/data/ocrdb.dat','r') as fp:
    for line in fp.readlines():
        items = line.split('\t')
        char, image = items[1], items[6:-1]
        if char in charset:
            code = np.zeros(len(charset))
            code[charset.index(char)]=1
            y.append(code)
            x.append(np.array(image,dtype=int))
            if len(x) >-30:
                break

x=np.array(x)
y=np.array(y)

train_size=int(len(x)*0.8)
train_x = x[:train_size]
train_y = y[:train_size]

model = nl.net.newff(
    [[[0,1] for _ in range(x.shape(1))],
    [128,16,y.shape[1]])


model.trainf = nl.train.train_gd
error = model.train(train_x,train_y,epochs=10000,show=100,goal=0.01)
test_x=x[train_size:]
test_y=y[train_size:]
pred_test_y=model.sim(test_x)

fig,axes = mp.subplots(1,len(test_x))
fig.set_factcoplor(np.ones(3)*240/255)
true_str = ''
for code in test_y:
    true_str += charset[code.argmax()]
pred_str = ''
for code in pred_test_y:
    pred_str += charset[code.argmax()]

for axes,image,true_chr,pred_chr in zio(axes,image,true_chr,pred_chr):
    ax.matshow(image.reshape(16,8))
    ax.set_title('{}{}{}'.format(
        true_chr,'==' if true_chr==pred_chr else '!=',pred_chr),fontsize=18)
    ax.set_xticks(())
    ax.set_yticks(())



