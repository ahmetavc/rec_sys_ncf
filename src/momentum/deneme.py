import matplotlib.pyplot as plt
import pickle

with open('neumf_mom_0_loss', 'rb') as f:
	neumf_mom_0_loss = pickle.load(f)

with open('neumf_mom_03_loss', 'rb') as f:
        neumf_mom_03_loss = pickle.load(f)

with open('neumf_mom_07_loss' ,'rb') as f:
        neumf_mom_07_loss = pickle.load(f)

print(neumf_mom_0_loss)
print()
print(neumf_mom_03_loss)
print()
print(neumf_mom_07_loss)
