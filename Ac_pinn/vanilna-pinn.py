import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
from torch import autograd
import time
# 记录开始时间
start_time = time.time()

class Net(nn.Module):
    def __init__(self,inport_size,hidden_size,output_size):
        super(Net,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(inport_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,output_size),
            #多项式f(x)用线性变化，用x即可
        )
    def forward(self,x):
        return self.net(x)



device=torch.device('cuda')
net=Net(2,100,1)
net.to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error 均方误差求
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # 优化器

def initial_fun(x):
    a=torch.cos((torch.pi)*x)*(x**2)
    a=autograd.Variable(a.float(),requires_grad=True).to(device)
    return a

#x,t的输入都需要为(n,1)
def pde(x,t):
    combin=torch.cat((x,t),dim=1)
    lenx=len(x)
    lent=len(t)
    u=net(combin)
    du=autograd.grad(u,combin,grad_outputs=torch.ones_like(net(combin)),create_graph=True)[0]#grad_outputs=torch.ones_like(net(combined_input))：这里的 grad_outputs 设置为 torch.ones_like(net(combined_input))，表示对于每个输出，
    # 初始的梯度为 1（即每个输出的偏导数为 1）。通常你会在多输出的情况下使用这个选项，在单输出时可以省略。
    u_x=du[:,0].view(lenx,1)
    u_t=du[:,1].view(lent,1)
    u_xx=autograd.grad(u_x,combin,grad_outputs=torch.ones_like((net(combin))),create_graph=True)[0]
    u_xx=u_xx[:,1].view(lenx,1)
    rate=0.0001
    return u_t-rate*u_xx+5*(u**3-u)


#function loss
def lossr_f(n=1):
    rx = np.random.uniform(low=-1, high=1, size=(nr, 1))
    rx = autograd.Variable(torch.from_numpy(rx).float(), requires_grad=True)
    rt = np.random.uniform(low=i * dt, high=(i + 1) * dt, size=(nr, 1))
    rt = autograd.Variable(torch.from_numpy(rt).float(), requires_grad=True)
    rx = rx.to(device)
    rt = rt.to(device)
    # define function loss
    pde_result = pde(rx, rt)
    reference_result = torch.zeros(nr, 1).to(device)
    lossr = mse_cost_function(pde_result, reference_result)
    return lossr

    # define boundary loss
def lossb_f(n=1):
    bx = (-1) * np.ones([nb, 1], dtype=np.float32)
    bx = autograd.Variable(torch.from_numpy(bx).float(), requires_grad=True).to(device)
    bx1 = np.ones([nb, 1], dtype=np.float32)
    bx1 = autograd.Variable(torch.from_numpy(bx1).float(), requires_grad=True).to(device)
    bt = np.random.uniform(low=i * dt, high=(i + 1) * dt, size=(nb, 1))
    bt = autograd.Variable(torch.from_numpy(bt).float(), requires_grad=True).to(device)
    bu = net(torch.cat((bx, bt), dim=1))
    bu1 = net(torch.cat((bx1, bt), dim=1))  # compute boundary result

    dbx = autograd.grad(bu, bx, grad_outputs=torch.ones_like((net(torch.cat((bx1, bt), dim=1)))), create_graph=True)[0]
    dbx1 = autograd.grad(bu1, bx1, grad_outputs=torch.ones_like((net(torch.cat((bx1, bt), dim=1)))), create_graph=True)[0]

    dbu = net(torch.cat((dbx, bt), dim=1))
    dbu1 = net(torch.cat((dbx1, bt), dim=1))

    bu_dbu = torch.cat((bu, dbu), dim=0)
    bu1_dbu1 = torch.cat((bu1, dbu1), dim=0)
    lossb = mse_cost_function(bu_dbu, bu1_dbu1)
    return lossb

#initial loss
def lossi_f(n=1):
    if i == 0:
        initial_rel_result = initial_fun(ix)
    else:
        initial_rel_result = initial_rel_result1
    initial_net_result = net(torch.cat((ix, it), dim=1))
    lossi = mse_cost_function(initial_rel_result, initial_net_result)
    return lossi

losses=[]
iterations=5000
Nmax = 15
for i in range(3):
    dt = 0.1
    for epoch in range(iterations):
        nr=500
        ni=128
        nb=42

        # define initial loss
        ix = torch.linspace(-1, 1, steps=ni).view(ni, 1)
        ix = autograd.Variable(ix.float(), requires_grad=True).to(device)
        it = i * dt * np.ones([ni, 1], dtype=np.float32)
        it = autograd.Variable(torch.from_numpy(it).float(), requires_grad=True).to(device)

        lossr = lossr_f(1)
        lossi = lossi_f(1)
        lossb = lossb_f(1)
        #总loss
        loss=lossr*1+lossb*1+lossi*100
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # 梯度归0
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(epoch, "Traning Loss:", loss.data)
            if epoch != 0:
                lk = losses[-2] - losses[-1]
                lk1 = losses[-3] - losses[-2]
                lk2 = losses[-4] - losses[-3]
                Lk = (max(abs(lk), abs(lk1), abs(lk2)))
                print("Lk is :", Lk, "num nr is:", nr, "num ni is:", ni)
                judge = 1 if Lk >= 0.0005 else 0

    it1 = (i+1) * dt * np.ones([ni, 1], dtype=np.float32)
    it1 = autograd.Variable(torch.from_numpy(it1).float(), requires_grad=True).to(device)
    initial_rel_result1 = net(torch.cat((ix, it1), dim=1)).to(device)
    initial_rel_result1= autograd.Variable(initial_rel_result1.float(), requires_grad=True).to(device)

    x=torch.linspace(-1, 1, steps=200).view(200,1).to(device)
    t=torch.ones_like(x).to(device)
    t=t*(i+1)*dt
    u=net(torch.cat((x,t),dim=1))
    plt.figure(i+1)
    plt.plot(x.cpu().detach().numpy(),u.cpu().detach().numpy())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"执行时间：{elapsed_time} 秒")
plt.show()

