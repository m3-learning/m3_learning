import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
import numpy as np


# extended from Zheng Shi zhs310@lehigh.edu and Majid Jahani maj316@lehigh.edu
# https://github.com/Optimization-and-Machine-Learning-Lab/TRCG
# BSD 3-Clause License

# Copyright (c) 2023, Optimization-and-Machine-Learning-Lab

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



class TRCG(Optimizer):


    def __init__(self, model, 
                 radius, device,
                   closure_size = 1,  # specifies how many parts the 
#                  lr=required, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False, *, maximize: bool = False,
#                  foreach: Optional[bool] = None,
                 cgopttol=1e-7,c0tr=0.2,c1tr=0.25,c2tr=0.75,t1tr=0.25,t2tr=2.0,
                 radius_max=5.0,
                 radius_initial=0.1,
                 differentiable: bool = False
                ):
        
        
        self.model = model
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = 60
        
        
        
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.closure_size = closure_size
    
    
        defaults = dict(
#             lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov,
#                         maximize=maximize, foreach=foreach,
                        differentiable=differentiable
                       )
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.params = list(model.parameters())


        super(TRCG, self).__init__(self.params, defaults)

    def findroot(self,x,p):
        aa = 0.0; bb = 0.0; cc = 0.0
        for pi, xi in zip(p,x):
            aa += (pi*pi).sum()
            bb += (pi*xi).sum()
            cc += (xi*xi).sum()
        bb = bb*2.0
        cc = cc - self.radius**2
        alpha = (-2.0*cc)/(bb+(bb**2-(4.0*aa*cc)).sqrt())
        return alpha.data.item()    
    
    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
         
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                 
                state = self.state[p]
                if 'pk' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['ph'])

          
        
    def CGSolver(self,loss_grad,cnt_compute, closure):
    
        cg_iter = 0 # iteration counter
        x0 = [] # define x_0 as a list
        for i in self.model.parameters():
            x0.append(torch.zeros(i.shape).to(self.device))
    
        r0 = [] # set initial residual to gradient
        p0 = [] # set initial conjugate direction to -r0
        self.cgopttol = 0.0
        
        for i in loss_grad:
            r0.append(i.data+0.0)     
            p0.append(0.0-i.data)
            self.cgopttol+=torch.norm(i.data)**2
        
        self.cgopttol = self.cgopttol.data.item()**0.5
        self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
    
        cg_term = 0
        j = 0

        while 1:
            j+=1
    
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j=j-1
                p1 = x0
                print ('\n\nCG has issues !!!\n\n')
                break
            # hessian vector product
            
            
            
            Hp = self.computeHessianVector(closure, p0)            
            cnt_compute+=1
            
            
            pHp = self.computeDotProduct(Hp, p0) # quadratic term
    
            # if nonpositive curvature detected, go for the boundary of trust region
            if pHp.data.item() <= 0:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 1
                break
            
            # if positive curvature
            # vector product
            rr0 = 0.0
            for i in r0:
                rr0 += (i*i).sum()
            
            # update alpha
            alpha = (rr0/pHp).data.item()
        
            x1 = []
            norm_x1 = 0.0
            for e in range(len(x0)):
                x1.append(x0[e]+alpha*p0[e])
                norm_x1 += torch.norm(x0[e]+alpha*p0[e])**2
            norm_x1 = norm_x1**0.5
            
            # if norm of the updated x1 > radius
            if norm_x1.data.item() >= self.radius:
                tau = self.findroot(x0,p0)
                p1 = []
                for e in range(len(x0)):
                    p1.append(x0[e]+tau*p0[e])
                cg_term = 2
                break
    
            # update residual
            r1 = []
            norm_r1 = 0.0
            for e in range(len(r0)):
                r1.append(r0[e]+alpha*Hp[e])
                norm_r1 += torch.norm(r0[e]+alpha*Hp[e])**2
            norm_r1 = norm_r1**0.5
    
            if norm_r1.data.item() < self.cgopttol:
                p1 = x1
                cg_term = 3
                break
    
            rr1 = 0.0
            for i in r1:
                rr1 += (i*i).sum()
    
            beta = (rr1/rr0).data.item()
    
            # update conjugate direction for next iterate
            p1 = []
            for e in range(len(r1)):
                p1.append(-r1[e]+beta*p0[e])
    
            p0 = p1
            x0 = x1
            r0 = r1
    

        cg_iter = j
        d = p1

        return d,cg_iter,cg_term,cnt_compute        
        
        
        
    def computeHessianVector(self, closure, p):

        
        with torch.enable_grad():
            if self.closure_size == 1 and self.gradient_cache is not None:
                # we reuse the gradient computation 
                 
                Hpp = torch.autograd.grad(self.gradient_cache,
                                              self.params,
                                              grad_outputs=p, 
                                              retain_graph=True) # hessian-vector in tuple
                Hp = [Hpi.data+0.0 for Hpi in Hpp]
                    
            
                
            
            else:
        
                for part in range(self.closure_size):
                    loss = closure(part,self.closure_size, self.device)
                    loss_grad_v = torch.autograd.grad(loss,self.params,create_graph=True) 
                    Hpp = torch.autograd.grad(loss_grad_v,
                                              self.params,
                                              grad_outputs=p, 
                                              retain_graph=False) # hessian-vector in tuple
                    if part == 0:
                        Hp = [Hpi.data+0.0 for Hpi in Hpp]
                    else:
                        for Hpi, Hppi in zip(Hp, Hpp):
                            Hpi.add_(Hppi)
                    
                    
        return Hp
        
    def computeLoss(self, closure):
        lossVal = 0.0
        with torch.no_grad():
            for part in range(self.closure_size):
                loss = closure(part,self.closure_size, self.device)
                lossVal+= loss.item()
                    
                    
        return lossVal        

        
    def computeGradientAndLoss(self, closure):
        lossVal = 0.0
        with torch.enable_grad():
            for part in range(self.closure_size):
                loss = closure(part,self.closure_size, self.device)
                lossVal+= loss.item()
                if  self.closure_size == 1 and self.gradient_cache is None:
                     
                    loss_grad = torch.autograd.grad(loss,self.params,retain_graph=True,create_graph=True) 
                    self.gradient_cache =  loss_grad
                else:
                    
                    loss_grad = torch.autograd.grad(loss,self.params,create_graph=False) 
                
                if part == 0:
                    grad = [p.data+0.0 for p in loss_grad]
                else:
                    for gi, gip in zip(grad, loss_grad):
                        gi.add_(gip) 
                    
                    
        return lossVal, grad        
        
    def computeGradient(self, closure):
        return self.computeGradientAndLoss(closure)[1]
                    
                    
        return grad
        
    def computeDotProduct(self,v,z):
        return torch.sum(torch.vstack([ (vi*zi).sum() for vi, zi in zip(v, z)  ]))
        
    def computeNorm(self,v):
        return torch.sqrt(torch.sum(torch.vstack([ (p**2).sum() for p in v])))
        
    @_use_grad_for_differentiable
    def step(self, closure):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        self.gradient_cache = None
        
        # store the initial weights 
        wInit = [w+0.0 for w in self.params]
        
        update = 2
        
        lossInit, loss_grad = self.computeGradientAndLoss(closure)
        NormG = self.computeNorm(loss_grad)
        
        cnt_compute=1

        
       
        # Conjugate Gradient Method
        d, cg_iter, cg_term, cnt_compute = self.CGSolver(loss_grad,cnt_compute, closure)

        
        Hd = self.computeHessianVector(closure, d)
        dHd = self.computeDotProduct(Hd, d)
        
        
        # update solution
        for wi, di in zip(self.params, d):
            with torch.no_grad():
                wi.add_(di)
        
        loss_new = self.computeLoss(closure)
        numerator = lossInit - loss_new

        gd = self.computeDotProduct(loss_grad, d)

        norm_d = self.computeNorm(d)
        
        denominator = -gd.data.item() - 0.5*(dHd.data.item())

        # ratio
        rho = numerator/denominator

        
        outFval = loss_new
        if rho < self.c1tr: # shrink radius
            self.radius = self.t1tr*self.radius
            update = 0
        elif rho > self.c2tr and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            self.radius = min(self.t2tr*self.radius,self.radius_max)
            update = 1
        # otherwise, radius remains the same
        if rho <= self.c0tr or numerator < 0: # reject d
            update = 3
            self.radius = self.t1tr*self.radius
            for wi, di in zip(self.params, d):
                with torch.no_grad():
                    wi.sub_(di)  
            outFval = lossInit
        return outFval, self.radius, cnt_compute, cg_iter    