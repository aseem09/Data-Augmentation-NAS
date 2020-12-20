import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Architect(object):

    def __init__(self, model_gen, model_disc, model, network_params, criterion, args, gamma=0.01):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.model_gen = model_gen
        self.model_disc = model_disc
        self.network_params = network_params
        self.criterion = criterion
        self.gamma = gamma
        self.args = args

    # def _compute_unrolled_classifier(self, input, target, eta, network_optimizer):
    #     logits = self.model(input)
    #     loss = self.criterion(logits, target)
    #     theta = _concat(self.model.parameters()).data
    #     try:
    #         moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    #     except:
    #         moment = torch.zeros_like(theta)
    #     dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    #     unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    #     return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, lr, lr_gen, lr_disc, lr_model, optimizer_model, optimizer_gen, optimizer_a):
        optimizer_a.zero_grad()

        self._backward_step_unrolled(
            input_train, target_train, input_valid, target_valid, lr, lr_gen, lr_disc, lr_model, optimizer_model, optimizer_gen, optimizer_a)

        nn.utils.clip_grad_norm_(self.model_disc.module.arch_parameters(), self.args.grad_clip)
        optimizer_a.step()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, lr_gen, lr_disc, lr_model, optimizer_model, optimizer_gen, optimizer_a):
        # unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, optimizer_a)

        # COMPUTE FD 1

        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()

        dalpha = [v.grad for v in self.model_disc.module.arch_parameters()]
        vector = [v.grad.data for v in self.model.parameters()]
        grads_p, grads_n = self._hessian_vector_product_classifier(vector, input_train, target_train)
        
        implicit_grads_p = self._hessian_vector_product_gen(grads_p, input_train, target_train)
        implicit_grads_n = self._hessian_vector_product_gen(grads_n, input_train, target_train)

        implicit_grads = [(x-y) for x, y in zip(implicit_grads_p, implicit_grads_n)]

        for index, param in enumerate(self.model_disc.module.arch_parameters()):
            param.grad = implicit_grads[index]

        # for index, param in enumerate(self.model_disc.module.arch_parameters()):
        #     print(str(index) + str(param.grad))

        # for g, ig in zip(dalpha, implicit_grads):
        #     if ig is not None:
        #         g.data.sub_(lr, ig.data)

        # for v, g in zip(self.model_disc.module.arch_parameters(), dalpha):
        #     if v.grad is None:
        #         v.grad = Variable(g.data)
        #     else:
        #         v.grad.data.copy_(g.data)

        # COMPUTE FD 2


        # STEP ONE
        # logits = self.model(input_valid)
        # unrolled_loss = self.criterion(logits, target_valid)
        # unrolled_loss.backward()
        # vector = [v.grad.data for v in self.model.parameters()]

        # # STEP TWO
        # m_loss, gen_loss = self.compute_loss(input_train, target_train)
        # grad_s = torch.autograd.grad(m_loss, self.model.parameters(), create_graph=True)
        # grad_s = torch.cat([x.view(-1) for x in grad_s])
        # grad2_g = None
        # for grads in grad_s:
        #     s_grads = torch.autograd.grad(grads, self.model_gen.parameters(), retain_graph=True)
        #     if grad2_g is None:
        #         grad2_g = s_grads
        #     else:
        #         grad2_g += s_grads

        # # STEP THREE 
        # grad3 =  torch.autograd.grad(gen_loss, self.model_gen.parameters(), create_graph=True)
        # grad3 = torch.cat([x.view(-1) for x in grad3])
        # grad3_a = None
        # for grads in grad3:
        #     s_grads = torch.autograd.grad(grads, self.model_disc.module.arch_parameters(), retain_graph=True)
        #     if grad3_a is None:
        #         grad3_a = s_grads
        #     else:
        #         grad3_a += s_grads

        # grad3_a = torch.autograd.grad(grad3, self.model_disc.module.arch_parameters())


        # for index, param in enumerate(self.model_disc.module.arch_parameters()):
        #     print(str(index) + " " + str(param))

        # for index, param in enumerate(self.model_disc.module.arch_parameters()):
        #     print(str(index) + str(param.grad))
        

#   def _construct_model_from_theta(self, theta):
#     model_new = self.model.new()
#     model_dict = self.model.state_dict()

#     params, offset = {}, 0
#     for k, v in self.model.named_parameters():
#       v_length = np.prod(v.size())
#       params[k] = theta[offset: offset+v_length].view(v.size())
#       offset += v_length

#     assert offset == len(theta)
#     model_dict.update(params)
#     model_new.load_state_dict(model_dict)
#     return model_new.cuda()

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
    
    def _hessian_vector_product_classifier(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model_gen.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model_gen.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        grads_p = [x.div_(2*R) for x in grads_p]
        grads_n = [x.div_(2*R) for x in grads_n]
        
        return grads_p, grads_n

    def compute_loss(self, input, target):
        # logits_1 = self.model(input)
        # m_loss_1 = self.criterion(logits_1, target)

        # # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (self.args.batch_size, self.args.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, 10, self.args.batch_size)))

        # # Generate a batch of images
        # gen_imgs = self.model_gen(z, gen_labels)

        # logits_2 = self.model(gen_imgs)
        # m_loss_2 = self.gamma * self.criterion(logits_2, gen_labels)

        # m_loss = m_loss_1 + m_loss_2


        adversarial_loss = nn.MSELoss()
        adversarial_loss.cuda()

        # Adversarial ground truths
        # valid = Variable(FloatTensor(self.args.batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(self.args.batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        # real_imgs = Variable(input.type(FloatTensor))
        # labels = Variable(target.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (self.args.batch_size, self.args.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, 10, self.args.batch_size)))

        # Generate a batch of images
        # gen_imgs = self.model_gen(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        # validity = self.model_disc(gen_imgs, gen_labels)
        # g_loss = self.gamma * adversarial_loss(validity, valid)

        # g_loss.backward()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss for real images
        # validity_real = self.model_disc(real_imgs, labels)
        # d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        # validity_fake = self.model_disc(gen_imgs.detach(), gen_labels)
        # d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        # d_loss = (d_real_loss + d_fake_loss) / 2

        # d_loss.backward()
            
        # nn.utils.clip_grad_norm_(self.network_params, self.args.grad_clip)

        # ---------------------
        #  Train Model
        # ---------------------

        logits_1 = self.model(input)
        m_loss_1 = self.criterion(logits_1, target)

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.args.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, CIFAR_CLASSES, batch_size)))

        # Generate a batch of images
        gen_imgs = self.model_gen(z, gen_labels)

        logits_2 = self.model(gen_imgs)
        m_loss_2 = self.gamma * self.criterion(logits_2, gen_labels)

        m_loss = m_loss_1 + m_loss_2

        return m_loss

    def compute_loss_gen(self, input, target):

        adversarial_loss = nn.MSELoss()
        adversarial_loss.cuda()

        # Adversarial ground truths
        valid = Variable(FloatTensor(self.args.batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(self.args.batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        # real_imgs = Variable(input.type(FloatTensor))
        # labels = Variable(target.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (self.args.batch_size, self.args.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, 10, self.args.batch_size)))

        # Generate a batch of images
        gen_imgs = self.model_gen(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = self.model_disc(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        return g_loss