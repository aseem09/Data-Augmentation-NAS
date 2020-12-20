import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from classifier_model import Resnet18
from generator import Generator

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Architect(object):

    def __init__(self, model_gen, model_disc, model, network_params, criterion, adversarial_loss, num_classes, args, gamma=0.01):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.model_gen = model_gen
        self.model_disc = model_disc
        self.network_params = network_params
        self.criterion = criterion
        self.gamma = gamma
        self.args = args
        self.num_classes = num_classes
        self.adversarial_loss = adversarial_loss

    def step(self, input_train, target_train, input_valid, target_valid, lr, lr_gen, lr_disc, lr_model, optimizer_disc, optimizer_model, optimizer_gen, optimizer_a):
        optimizer_a.zero_grad()

        self._backward_step_unrolled(
            input_train, target_train, input_valid, target_valid, lr, lr_model, optimizer_model, lr_gen, optimizer_gen)

        nn.utils.clip_grad_norm_(self.model_disc.module.arch_parameters(), self.args.grad_clip)
        optimizer_a.step()

    def _compute_unrolled_model_classifier(self, input, target, eta, network_optimizer, unrolled_gen):
        loss = self.compute_loss(input, target, unrolled_gen)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(0.9)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + 5e-4*theta
        unrolled_model = self._construct_model_from_theta_classifier(theta.sub(eta, moment+dtheta))
        return unrolled_model
    
    def _construct_model_from_theta_classifier(self, theta):
        model_new = Resnet18()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()


    def _compute_unrolled_model_gen(self, input, target, eta, network_optimizer):
        loss = self.compute_loss_gen(input, target)
        theta = _concat(self.model_gen.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model_gen.parameters()).mul_(self.args.b1)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model_gen.parameters())).data + 0*theta
        unrolled_model = self._construct_model_from_theta_gen(theta.sub(eta, moment+dtheta))
        return unrolled_model
    
    def _construct_model_from_theta_gen(self, theta):
        model_new = Generator(100)
        model_dict = self.model_gen.state_dict()

        params, offset = {}, 0
        for k, v in self.model_gen.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, lr_model, optimizer_model, lr_gen, optimizer_gen):

        unrolled_model_gen = self._compute_unrolled_model_gen(input_train, target_train, lr_gen, optimizer_gen)
        unrolled_model_classifier = self._compute_unrolled_model_classifier(input_train, target_train, lr_model, optimizer_model, unrolled_model_gen)

        # COMPUTE FD 1

        logits = unrolled_model_classifier(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()

        #should be unrolled
        vector = [v.grad.data for v in unrolled_model_classifier.parameters()]

        grads_p, grads_n = self._hessian_vector_product_classifier(vector, input_train, target_train, unrolled_model_gen)
        
        implicit_grads_p = self._hessian_vector_product_gen(grads_p, input_train, target_train)
        implicit_grads_n = self._hessian_vector_product_gen(grads_n, input_train, target_train)

        implicit_grads = [(x-y) for x, y in zip(implicit_grads_p, implicit_grads_n)]

        for index, param in enumerate(self.model_disc.module.arch_parameters()):
            param.grad = implicit_grads[index] * lr

        # for index, param in enumerate(self.model_disc.module.arch_parameters()):
        #     print(str(index) + str(param.grad))

    def _hessian_vector_product_classifier(self, vector, input, target, unrolled_gen, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # loss should use unrolled gen version
        loss = self.compute_loss(input, target, unrolled_gen)
        grads_p = torch.autograd.grad(loss, unrolled_gen.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target, unrolled_gen)
        grads_n = torch.autograd.grad(loss, unrolled_gen.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        grads_p = [x.div_(2*R) for x in grads_p]
        grads_n = [x.div_(2*R) for x in grads_n]
        
        return grads_p, grads_n

    def _hessian_vector_product_gen(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model_gen.parameters(), vector):
            p.data.add_(R, v)
        loss = self.compute_loss_gen(input, target)
        grads_p = torch.autograd.grad(loss, self.model_disc.module.arch_parameters())

        for p, v in zip(self.model_gen.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.compute_loss_gen(input, target)
        grads_n = torch.autograd.grad(loss, self.model_disc.module.arch_parameters())

        for p, v in zip(self.model_gen.parameters(), vector):
            p.data.add_(R, v)
        
        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    # should use unrolled gen version
    def compute_loss(self, input, target, unrolled_gen):

        logits_1 = self.model(input)
        m_loss_1 = self.criterion(logits_1, target)

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (self.args.batch_size, self.args.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, self.num_classes, self.args.batch_size)))

        # Generate a batch of images
        gen_imgs = unrolled_gen(z, gen_labels)

        logits_2 = self.model(gen_imgs)
        m_loss_2 = self.gamma * self.criterion(logits_2, gen_labels)

        m_loss = m_loss_1 + m_loss_2

        return m_loss

    def compute_loss_gen(self, input, target):

        # Adversarial ground truths
        valid = Variable(FloatTensor(self.args.batch_size, 1).fill_(1.0), requires_grad=False)

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (self.args.batch_size, self.args.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, 10, self.args.batch_size)))

        # Generate a batch of images
        gen_imgs = self.model_gen(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = self.model_disc(gen_imgs, gen_labels)
        g_loss = self.adversarial_loss(validity, valid)

        return g_loss