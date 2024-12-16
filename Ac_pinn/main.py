import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
from torch import autograd
import scipy.io



class Net(nn.Module):
    def __init__(self, inport_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inport_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            # 多项式f(x)用线性变化，用x即可
        )

    def forward(self, x):
        return self.net(x)

#设备
device = torch.device('cuda')
net = Net(1, 200, 101)
net.to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error 均方误差求
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)  # 优化器



#100阶Rk
q=100
tmp = np.float32(np.loadtxt('D:\python_code\Ac_pinn\Butcher_IRK%d.txt' % (q), ndmin=2))
IRK_weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
IRK_weights=torch.tensor(IRK_weights)
IRK_weights=autograd.Variable(IRK_weights.float(), requires_grad=True).to(device)
print(IRK_weights.size())


# 加载 .mat 文件
data1 = scipy.io.loadmat(r'D:\python_code\Ac_pinn\x.mat')
data2 = scipy.io.loadmat(r'D:\python_code\Ac_pinn\uu.mat')
# 从字典中提取实际的数据，假设变量名为 'x' 和 'u'
x = data1['x']  # 取出 x 对应的 numpy 数组
uu = data2['uu']  # 取出 u 对应的 numpy 数组

# 转换为 PyTorch Tensor
x = torch.tensor(x).view(-1,1)
x = autograd.Variable(x.float(), requires_grad=True).to(device)

uu_tensor = torch.tensor(uu)
u_initial=uu_tensor[:,160].view(-1,1)
u_initial=torch.cat(([u_initial] * (q+1)), dim=1)
u_initial = autograd.Variable(u_initial.float(), requires_grad=True).to(device)
u_exact=uu_tensor[:,180].view(-1,1)
u_exact = autograd.Variable(u_exact.float(), requires_grad=True).to(device)
data=torch.cat((x,u_initial),dim=1)

#随机采点
data= autograd.Variable(data.float(), requires_grad=True).to(device)
random_indices = torch.randperm(data.size(0))[:200]  # tensor.size(0) == 512
    # 根据随机索引选择 200 行
random_sample = data[random_indices]
random_sample= autograd.Variable(random_sample.float(), requires_grad=True).to(device)
x1 = random_sample[:,0].view(-1,1)
x1=autograd.Variable(x1.float(), requires_grad=True).to(device)

'''
def N(x):
    u=net(x)
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return (0.0001*u_xx-5*(u**3)+5*u)[:,0:q]
'''

def ssen(u=1):
    # ssen
    u_output = net(x1)
    u_x = autograd.grad(u_output, x1, grad_outputs=torch.ones_like(u_output), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x1, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_output_n1 = (0.0001*u_xx-5*(u_output**3)+5*u_output)[:,0:q].to(device)
    u_output_n1 = (u_output_n1 @ (IRK_weights.T)) * dt
    u_ni = u_output-u_output_n1
    ssen =torch.sum((u_ni - random_sample[:,1:202])**2)
    return ssen


def sseb(u=1):
    # sseb
    # 初始化变量
    x_1 = torch.tensor([1], requires_grad=True, dtype=torch.float32).to(device)
    x_m1 = torch.tensor([-1], requires_grad=True, dtype=torch.float32).to(device)
    # 网络输出
    u_1 = net(x_1)
    u_m1 = net(x_m1)
    # 计算梯度
    ux_1 = autograd.grad(u_1, x_1, grad_outputs=torch.ones_like(u_1), create_graph=True)[0]
    ux_m1 = autograd.grad(u_m1, x_m1, grad_outputs=torch.ones_like(u_m1), create_graph=True)[0]
    # 损失函数计算
    loss_ux = torch.sum((ux_1 - ux_m1)**2)
    loss_u = torch.sum((u_1 - u_m1)**2)
    # 总损失
    sseb = loss_ux + loss_u
    return sseb

dt=0.1
iterative=10000
#losses=[]
time=1
for epoch in range(iterative):
    # 总损失
    lossb=sseb(1)
    lossn=ssen(1)
    loss=lossn+lossb
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  # 梯度归0
    #losses.append(loss.item())
    if epoch % 10 == 0:
        print(epoch, "Traning Loss:", loss.data)

u_pred = net(x)[:, q]
plt.figure(1)
plt.plot(x.cpu().detach().numpy(), u_pred.cpu().detach().numpy())
plt.figure(2)
plt.plot(x.cpu().detach().numpy(), u_exact.cpu().detach().numpy())
plt.show()
plt.figure(3)
plt.plot(x.cpu().detach().numpy(), u_initial.cpu().detach().numpy())
plt.show()


a=torch.tensor([0]).to(device)
a=a.float()
print('net(a) is',net(a)[-1])
print('u_exact[256,1] is',u_exact[256])
print('0 loss is',(net(a)[-1]-u_exact[256]))
norm_diff = np.linalg.norm(u_pred.cpu().detach().numpy() - u_exact.cpu().detach().numpy(), 2)
norm_exact = np.linalg.norm(u_exact.cpu().detach().numpy(), 2)

# 计算相对误差
relative_error = norm_diff / norm_exact
print('error is',relative_error)

'''
error1=(mse_cost_function(u_pred, u_exact)*(101))**(1/2)
ze=torch.zeros_like(u_m1)
print(error1)
error2=(mse_cost_function(u_exact, ze)*(101))**(1/2)
error=error1/error2
print(error2)
print('error is:',error)
'''