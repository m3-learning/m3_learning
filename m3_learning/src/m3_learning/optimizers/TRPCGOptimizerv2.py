"""
Created on Sun Jan 24 16:34:00 2021
@author: Martin Takac
"""

import numpy as np
import tensorflow as tf
import torch


class TRPCGOptimizerv2:

    cgopttol = 1e-7
    c0tr = 0.2
    c1tr = 0.25
    c2tr = 0.75  # when to accept
    t1tr = 0.75
    t2tr = 2.0
    radius_max = 5.0  # max radius
    radius_initial = 1.0
    radius = radius_initial

    @tf.function
    def computeHessianProduct(self, x, y, v):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                out = self.model(x)
                loss = tf.keras.losses.mean_squared_error(out, y)
                loss = tf.reduce_mean(loss)
            grad = tape2.gradient(loss, self.model.trainable_variables)

            gradSum = tf.reduce_sum([tf.reduce_sum(g*p0i)
                                     for g, p0i in zip(grad, v)])
        Hp = tape.gradient(gradSum, self.model.trainable_variables)
        return Hp

    def __init__(self, model, radius, precondition,
                 cgopttol=1e-7, c0tr=0.0001, c1tr=0.1, c2tr=0.75, t1tr=0.25, t2tr=2.0, radius_max=2.0,
                 radius_initial=0.1):

        self.model = model
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = sum([tf.size(w).numpy()
                              for w in self.model.trainable_weights])
        self.d = self.cgmaxiter
        self.cgmaxiter = min(120, self.cgmaxiter)
        self.iterationCounterForAdamTypePreconditioning = 0
        self.precondition = precondition
        if self.precondition != 0:
            self.DiagPrecond = [w.data*0.0 for w in self.model.parameters()]
            self.DiagScale = 0.0

    def findroot(self, x, p):
        aa = 0.0
        bb = 0.0
        cc = 0.0
        for e in range(len(x)):
            aa += tf.reduce_sum(p[e]*p[e])
            bb += tf.reduce_sum(p[e]*x[e])
            cc += tf.reduce_sum(x[e]*x[e])
        bb = bb*2.0
        cc = cc - self.radius**2
        alpha = (-2.0*cc)/(bb + tf.sqrt(bb**2-(4.0*aa*cc)))

        return alpha

    def computeListNorm(self, lst):
        return np.sum([tf.reduce_sum(ri*ri) for ri in lst])**0.5

    def computeListNormSq(self, lst):
        return np.sum([tf.reduce_sum(ri*ri) for ri in lst])

    def computeDotProducts(self, u, v):
        return tf.reduce_sum(tf.stack([tf.reduce_sum(ui * vi) for ui, vi in zip(u, v)], 0))

    def normOfVar(self, x):
        return tf.sqrt(self.computeDotProducts(x, x))

    def CGSolver(self, loss_grad, x, y):
        cg_iter = 0  # iteration counter
        x0 = [w.numpy()*0.0 for w in self.model.trainable_weights]
        if self.precondition == 0:
            r0 = [i+0.0 for i in loss_grad]  # set initial residual to gradient
            normGrad = self.normOfVar(r0)
            # set initial conjugate direction to -r0
            p0 = [-i+0.0 for i in loss_grad]
            self.cgopttol = self.computeListNormSq(loss_grad)
            self.cgopttol = self.cgopttol**0.5
            self.cgopttol = (min(0.5, self.cgopttol**0.5))*self.cgopttol
        else:
            r0 = [(i.data+0.0)*pr.data for i,
                  pr in zip(loss_grad, self.SquaredPreconditioner)]
            p0 = [-(i.data+0.0)*pr.data for i,
                  pr in zip(loss_grad, self.SquaredPreconditioner)]
            self.cgopttol = self.computeListNormSq(r0)
            self.cgopttol = self.cgopttol.data.item()**0.5
            self.cgopttol = (min(0.5, self.cgopttol**0.5))*self.cgopttol

        cg_term = 0
        j = 0

        while 1:
            j += 1
            self.CG_STEPS_TOOK = j
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j = j-1
                p1 = x0
                print('\n\nCG has issues !!!\n\n')
                break
            # hessian vector product
            if self.precondition == 0:
                Hp = self.computeHessianProduct(x, y, p0)
            else:
                loss_grad_direct \
                    = np.sum([(gi*(si*pr.data)).sum() for gi, si, pr in zip(loss_grad, p0, self.SquaredPreconditioner)])
                Hp = torch.autograd.grad(loss_grad_direct, self.model.parameters(
                ), retain_graph=True)  # hessian-vector in tuple
                Hp = [g*pr.data for g,
                      pr in zip(Hp, self.SquaredPreconditioner)]

            pHp = tf.reduce_sum([tf.reduce_sum(Hpi*p0i)
                                 for Hpi, p0i in zip(Hp, p0)])

            # if nonpositive curvature detected, go for the boundary of trust region
            if pHp <= 0:
                tau = self.findroot(x0, p0)
                p1 = [xi+tau*p0i for xi, p0i in zip(x0, p0)]
                cg_term = 1
                break

            # if positive curvature
            # vector product
            rr0 = self.computeListNormSq(r0)
            # update alpha
            alpha = (rr0/pHp)

            x1 = [xi+alpha*pi for xi, pi in zip(x0, p0)]
            norm_x1 = self.computeListNorm(x1)

            if norm_x1 >= self.radius:
                tau = self.findroot(x0, p0)

                p1 = [xi+tau*pi for xi, pi in zip(x0, p0)]
                cg_term = 2
                break

            # update residual
            r1 = [ri+alpha*Hpi for ri, Hpi in zip(r0, Hp)]
            norm_r1 = self.computeListNorm(r1)

            if norm_r1 < self.cgopttol:
                p1 = x1
                cg_term = 3
                break

            rr1 = self.computeListNormSq(r1)
            beta = (rr1/rr0)

            # update conjugate direction for next iterate
            p1 = [-ri+beta*pi for ri, pi in zip(r1, p0)]

            p0 = p1
            x0 = x1
            r0 = r1

        cg_iter = j
        if self.precondition != 0:
            p1 = [pi*pr.data for pi, pr in zip(p1, self.SquaredPreconditioner)]

        d = p1

        return d, cg_iter, cg_term

    def assignToModel(self, newX):
        for w, nw in zip(self.model.trainable_weights, newX):
            w.assign(nw)

    def addToModel(self, d):
        for w, di in zip(self.model.trainable_weights, d):
            w.assign_add(di)

    def computeLoss(self, x, y):
        out = self.model(x)
        loss = tf.keras.losses.mean_squared_error(out, y)
        loss = tf.reduce_mean(loss)
        return loss

    def computeLossAndGrad(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.computeLoss(x, y)
        grad = tape.gradient(loss, self.model.trainable_variables)
        return loss, grad

    def step(self, x, y):

        loss, grad = self.computeLossAndGrad(x, y)
        w0 = [w.numpy()+0.0 for w in self.model.trainable_weights]
        update = 3

        while update == 3:
            update = 2
            # Conjugate Gradient Method

            d, cg_iter, cg_term = self.CGSolver(grad, x, y)
            Hd = self.computeHessianProduct(x, y, d)
            dHd = tf.reduce_sum([tf.reduce_sum(Hdi*di)
                                 for Hdi, di in zip(Hd, d)])
            gd = tf.reduce_sum([tf.reduce_sum(gi*di)
                                for gi, di in zip(grad, d)])
            norm_d = self.computeListNorm(d)

            denominator = -gd - 0.5*(dHd)
            self.addToModel(d)
            loss_new = self.computeLoss(x, y)
            numerator = loss - loss_new

            # ratio
            rho = numerator/denominator

            if rho < self.c1tr:  # shrink radius
                self.radius = self.t1tr*self.radius
                update = 0
            # and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            if rho > self.c2tr:
                self.radius = min(self.t2tr*self.radius, self.radius_max)
                update = 1
            # otherwise, radius remains the same
            if rho <= self.c0tr:  # reject d
                update = 3

                self.assignToModel(w0)

                lossTMP, grad = self.computeLossAndGrad(x, y)

                print('rejecting .... radius: %1.6e   FVALNew %1.6e,  DeltaF %1.6e ' % (
                    self.radius, lossTMP, numerator))
            if self.radius < 1e-15:
                break

        return loss, d, rho, update, cg_iter, cg_term, grad, norm_d, numerator, denominator, self.radius

    def stepMAE(self, loss, MAE, Coor, AtomTypes, Grid, Label):

        update = 3
        w0 = [a.data+0.0 for a in self.model.parameters()]
        loss_grad = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True, retain_graph=True)
        if self.precondition == 1:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale+(1-self.DiagScale)*gi*gi)
                di.data[di.data == 0] += 1.0
            self.DiagScale = 0.95
        if self.precondition == 2:  # Martens paper
            self.DiagScale = 0.001  # set lambda to what value?
            self.exponent = 0.75  # based on paper
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_((gi*gi + self.DiagScale)**self.exponent)
        if self.precondition == 3:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(1.0-self.DiagScale+self.DiagScale*gi*gi)
            self.DiagScale = 1e-2
        if self.precondition == 4:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale+(1-self.DiagScale)*gi*gi)
                di.data[di.data == 0] += 1.0
            self.DiagScale = 0.99
        if self.precondition == 5:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale+(1-self.DiagScale)*gi*gi)
                di.data[di.data == 0] += 1.0
            self.DiagScale = 0.90
        if self.precondition == 6:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale +
                             (1-self.DiagScale)*torch.abs(gi))
                di.data[di.data == 0] += 1.0
            self.DiagScale = 0.95

        if self.precondition == 6:
            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale +
                             (1-self.DiagScale)*torch.abs(gi))
                di.data[di.data == 0] += 1.0
            self.DiagScale = 0.95

        if self.precondition in [7, 8, 9]:
            if self.precondition == 7:
                self.DiagScale = 0.99
            if self.precondition == 8:
                self.DiagScale = 0.95
            if self.precondition == 9:
                self.DiagScale = 0.90

            self.iterationCounterForAdamTypePreconditioning += 1

            for gi, di in zip(loss_grad, self.DiagPrecond):
                di.data.set_(di.data*self.DiagScale +
                             (1-self.DiagScale)*torch.abs(gi))
                di.data[di.data == 0] += 1.0

        while update == 3:
            update = 2
            # Conjugate Gradient Method
            d, cg_iter, cg_term = self.CGSolver(loss_grad)

            for wi, di in zip(self.model.parameters(), d):
                wi.data.set_(wi.data+0.0+di)

            # MSE loss plus penalty term
            with torch.no_grad():
                loss_new = Projection_Error(XData, YData, idx, n_steps)

            numerator = loss.data.item() - loss_new.data.item()

            loss_grad_direct = np.sum([(gi*di).sum()
                                       for gi, di in zip(loss_grad, d)])

            Hd = torch.autograd.grad(loss_grad_direct, self.model.parameters(
            ), retain_graph=True)  # hessian-vector in tuple

            dHd = np.sum([(Hdi*di).sum() for Hdi, di in zip(Hd, d)])

            gd = np.sum([(gi*di).sum() for gi, di in zip(loss_grad, d)])

            norm_d = self.computeListNorm(d)

            denominator = -gd.data.item() - 0.5*(dHd.data.item())

            # ratio
            rho = numerator/denominator

            if rho < self.c1tr:  # shrink radius
                self.radius = self.t1tr*self.radius
                update = 0
            # and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            if rho > self.c2tr:
                self.radius = min(self.t2tr*self.radius, self.radius_max)
                update = 1
            # otherwise, radius remains the same
            if rho <= self.c0tr:  # reject d
                update = 3
                for wi, w0i in zip(self.model.parameters(), w0):
                    wi.data.set_(w0i.data)

        return d, rho, update, cg_iter, cg_term, loss_grad, norm_d, numerator, denominator