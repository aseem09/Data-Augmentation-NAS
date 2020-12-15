from classifier_model import Resnet18
from generator import Generator

c_model = Resnet18()
print(c_model)

print('\n')

gen = Generator(100)
print(gen)