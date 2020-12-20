import torch
import torch.nn as nn
import numpy as np

img_shape = (3, 32, 32)

# class Generator(nn.Module):
#     def __init__(self, latent_dim, num_classes=10):
#         super(Generator, self).__init__()

#         self.label_emb = nn.Embedding(num_classes, num_classes)

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(latent_dim + num_classes, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, noise, labels):
#         # Concatenate label embedding and image to produce input
#         gen_input = torch.cat((noise, self.label_emb(labels)), -1)
#         img = self.model(gen_input)
#         img = img.view(img.size(0), *img_shape)
#         return img

# Generator model

# # Generator(G_input_dim, label_dim, num_filters, G_output_dim)
# image_size = 32

# # Number of channels in the training images. For color images this is 3
# nc = 3

# # Size of z latent vector (i.e. size of generator input)
# nz = 100 + 10

# # Size of feature maps in generator
# ngf = 64

# class Generator(nn.Module):
#     def __init__(self, latent_dim, num_classes=10):
#         super(Generator, self).__init__()
#         self.ngpu = 1
#         # self.label_emb = nn.Embedding(num_classes, num_classes)

#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(latent_dim+ 1, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.LeakyReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.LeakyReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.LeakyReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             # nn.BatchNorm2d(ngf),
#             # nn.ReLU(True),
#             # state size. (nc) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 3, 1, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, noise, labels):
#         labels = labels.view(labels.size(0), -1)
#         # print(labels.size())
#         # print(noise.size())
#         gen_input = torch.cat([noise, labels], 1)
#         # print(gen_input.size())
#         gen_input = gen_input
#         gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
#         img = self.main(gen_input)
#         # print(img.size())
#         # img = img.view(img.size(0), *img_shape)
#         return img

# building generator
class Generator(nn.Module): 
	def __init__(self, latent_dim):
		super(Generator, self).__init__()
		self.label_embed = nn.Embedding(10, 100)
		self.depth=128

		def init(input, output, normalize=True): 
			layers = [nn.Linear(input, output)]
			if normalize: 
				layers.append(nn.BatchNorm1d(output, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers 

		self.generator = nn.Sequential(
			*init(latent_dim+100, self.depth), 
			*init(self.depth, self.depth*2), 
			*init(self.depth*2, self.depth*4), 
			*init(self.depth*4, self.depth*8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()
			)

	# torchcat needs to combine tensors 
	def forward(self, noise, labels): 
		gen_input = torch.cat((self.label_embed(labels), noise), -1)
		img = self.generator(gen_input)
		img = img.view(img.size(0), *img_shape)
		return img
