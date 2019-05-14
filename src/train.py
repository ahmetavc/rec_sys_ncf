import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import pickle
import csv

#Adam was standard I think

gmf_config_adadelta = {'alias': 'gmf_factor8neg4-implict-adadelta',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adadelta',
              'adadelta_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adadelta.model'}

mlp_config_adadelta = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-adadelta',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'adadelta',
              'adadelta_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-adadelta_Epoch19_HR0.1010_NDCG0.0457_adadelta.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adadelta.model'}

gmf_config_adagrad = {'alias': 'gmf_factor8neg4-implict-adagrad',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adagrad',
              'adagrad_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adagrad.model'}

mlp_config_adagrad = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-adagrad',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'adagrad',
              'adagrad_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-adagrad_Epoch19_HR0.1066_NDCG0.0479_adagrad.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adagrad.model'}

gmf_config_adam = {'alias': 'gmf_factor8neg4-implict-adam',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adam.model'}

mlp_config_adam = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-adam',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-adam_Epoch19_HR0.6025_NDCG0.3427_adam.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adam.model'}

gmf_config_adamax = {'alias': 'gmf_factor8neg4-implict-adamax',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adamax',
              'adamax_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adamax.model'}

mlp_config_adamax = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-adamax',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'adamax',
              'adamax_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-adamax_Epoch19_HR0.2422_NDCG0.1265_adamax.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_adamax.model'}

gmf_config_asgd = {'alias': 'gmf_factor8neg4-implict-asgd',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'asgd',
              'asgd_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_asgd.model'}

mlp_config_asgd = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-asgd',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'asgd',
              'asgd_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-asgd_Epoch19_HR0.0949_NDCG0.0418_asgd.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_asgd.model'}

gmf_config_rmsprop = {'alias': 'gmf_factor8neg4-implict-rmsprop',
              'num_epoch': 20,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'rmsprop',
              'rmsprop_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_rmsprop.model'}

mlp_config_rmsprop = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-rmsprop',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'rmsprop',
              'rmsprop_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              #'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-rmsprop_Epoch19_HR0.5689_NDCG0.3182_rmsprop.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_rmsprop.model'}

gmf_config_sgd = {'alias': 'gmf_factor8neg4-implict-sgd',
              'num_epoch': 20,
              'batch_size': 1024,
              'optimizer': 'sgd',
              'sgd_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_sgd.model'}

mlp_config_sgd = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001-sgd',
              'num_epoch': 20,
              'batch_size': 1024,  
              'optimizer': 'sgd',
              'sgd_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict-sgd_Epoch19_HR0.0944_NDCG0.0426_sgd.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}_sgd.model'}


# mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
#               'num_epoch': 200,
#               'batch_size': 256,  # 1024,
#               'optimizer': 'adam',
#               'adam_lr': 1e-3,
#               'num_users': 6040,
#               'num_items': 3706,
#               'latent_dim': 8,
#               'num_negative': 4,
#               'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
#               'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
#               'use_cuda': True,
#               'device_id': 7,
#               'pretrain': True,
#               'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
#               'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.xxxx_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.xxxx_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
# config_adadelta = gmf_config_adadelta
# engine_adadelta = GMFEngine(config_adadelta)
# config_adagrad = gmf_config_adagrad
# engine_adagrad = GMFEngine(config_adagrad)
# config_adam = gmf_config_adam
# engine_adam = GMFEngine(config_adam)
# config_adamax = gmf_config_adamax
# engine_adamax = GMFEngine(config_adamax)
# config_asgd= gmf_config_asgd
# engine_asgd = GMFEngine(config_asgd)
# config_rmsprop= gmf_config_rmsprop
# engine_rmsprop = GMFEngine(config_rmsprop)
# config_sgd= gmf_config_sgd
# engine_sgd = GMFEngine(config_sgd)

# config = mlp_config
# engine = MLPEngine(config)
config_adadelta = mlp_config_adadelta
engine_adadelta = MLPEngine(config_adadelta)
config_adagrad = mlp_config_adagrad
engine_adagrad = MLPEngine(config_adagrad)
config_adam = mlp_config_adam
engine_adam = MLPEngine(config_adam)
config_adamax = mlp_config_adamax
engine_adamax = MLPEngine(config_adamax)
config_asgd= mlp_config_asgd
engine_asgd = MLPEngine(config_asgd)
config_rmsprop= mlp_config_rmsprop
engine_rmsprop = MLPEngine(config_rmsprop)
config_sgd= mlp_config_sgd
engine_sgd = MLPEngine(config_sgd)

# config = neumf_config
# engine = NeuMFEngine(config)

#1 Adadelta Optimization
adadelta_hr = []
adadelta_ndcg = []
adadelta_loss = []

#2 Adagrad Optimization
adagrad_hr = []
adagrad_ndcg = []
adagrad_loss = []

#3 Adam Optimization
adam_hr = []
adam_ndcg = []
adam_loss = []

#4 Adamax Optimization
adamax_hr = []
adamax_ndcg = []
adamax_loss = []

#5 ASGD Optimization
asgd_hr = []
asgd_ndcg = []
asgd_loss = []

#6 RMSprop Optimization
rmsprop_hr = []
rmsprop_ndcg = []
rmsprop_loss = []

#7 SGD Optimization
sgd_hr = []
sgd_ndcg = []
sgd_loss = []

for x in range (7): #Number of optimizations
    for epoch in range(config_adadelta['num_epoch']):
        
#         if x == 0: #1 Adadelta Optimization
#             print('Epoch Adadelta {} starts !'.format(epoch))
#             print('-' * 80)
#             train_loader_adadelta = sample_generator.instance_a_train_loader(config_adadelta['num_negative'], config_adadelta['batch_size'])
#             loss = engine_adadelta.train_an_epoch(train_loader_adadelta, epoch_id=epoch)
#             hit_ratio, ndcg = engine_adadelta.evaluate(evaluate_data, epoch_id=epoch)
#             engine_adadelta.save(config_adadelta['alias'], epoch, hit_ratio, ndcg)
#             adadelta_hr.append(hit_ratio) 
#             adadelta_ndcg.append(ndcg)
#             adadelta_loss.append(loss)   
#             
#             if (epoch+1) == config_adadelta['num_epoch']:   
#                 with open('optimization/gmf_adadelta.csv', 'w') as f:
#                     csvlogger = csv.writer(f, dialect='excel', delimiter=';')
#                     csvlogger.writerow(zip(adadelta_hr, adadelta_ndcg, adadelta_loss))
#                 
#         elif x == 1: #2 Adagrad Optimization
#             print('Epoch Adagrad {} starts !'.format(epoch))
#             print('-' * 80)
#             train_loader_adagrad = sample_generator.instance_a_train_loader(config_adagrad['num_negative'], config_adagrad['batch_size'])
#             loss = engine_adagrad.train_an_epoch(train_loader_adagrad, epoch_id=epoch)
#             hit_ratio, ndcg = engine_adagrad.evaluate(evaluate_data, epoch_id=epoch)
#             engine_adagrad.save(config_adagrad['alias'], epoch, hit_ratio, ndcg)
#             adagrad_hr.append(hit_ratio) 
#             adagrad_ndcg.append(ndcg)
#             adagrad_loss.append(loss)   
#             
#             if (epoch+1) == config_adagrad['num_epoch']:   
#                 with open('optimization/gmf_adagrad.csv', 'w') as f:
#                     csvlogger = csv.writer(f, dialect='excel', delimiter=';')
#                     csvlogger.writerow(zip(adagrad_hr, adagrad_ndcg, adagrad_loss))
#                 
#         elif x == 2: #3 Adam Optimization
#             print('Epoch Adam {} starts !'.format(epoch))
#             print('-' * 80)
#             train_loader_adam = sample_generator.instance_a_train_loader(config_adam['num_negative'], config_adam['batch_size'])
#             loss = engine_adam.train_an_epoch(train_loader_adam, epoch_id=epoch)
#             hit_ratio, ndcg = engine_adam.evaluate(evaluate_data, epoch_id=epoch)
#             engine_adam.save(config_adam['alias'], epoch, hit_ratio, ndcg)
#             adam_hr.append(hit_ratio) 
#             adam_ndcg.append(ndcg)
#             adam_loss.append(loss)   
#             
#             if (epoch+1) == config_adam['num_epoch']:   
#                 with open('optimization/gmf_adam.csv', 'w') as f:
#                     csvlogger = csv.writer(f, dialect='excel', delimiter=';')
#                     csvlogger.writerow(zip(adam_hr, adam_ndcg, adam_loss))
#                 
#                 
#         elif x == 3: #4 Adamax Optimization
#             print('Epoch Adamax {} starts !'.format(epoch))
#             print('-' * 80)
#             train_loader_adamax = sample_generator.instance_a_train_loader(config_adamax['num_negative'], config_adamax['batch_size'])
#             loss = engine_adamax.train_an_epoch(train_loader_adamax, epoch_id=epoch)
#             hit_ratio, ndcg = engine_adamax.evaluate(evaluate_data, epoch_id=epoch)
#             engine_adamax.save(config_adamax['alias'], epoch, hit_ratio, ndcg)
#             adamax_hr.append(hit_ratio) 
#             adamax_ndcg.append(ndcg)
#             adamax_loss.append(loss)   
#             
#             if (epoch+1) == config_adamax['num_epoch']:   
#                 with open('optimization/gmf_adamax.csv', 'w') as f:
#                     csvlogger = csv.writer(f, dialect='excel', delimiter=';')
#                     csvlogger.writerow(zip(adamax_hr, adamax_ndcg, adamax_loss))
                
#         if x == 4: #5 ASGD Optimization
#             print('Epoch ASGD {} starts !'.format(epoch))
#             print('-' * 80)
#             train_loader_asgd = sample_generator.instance_a_train_loader(config_asgd['num_negative'], config_asgd['batch_size'])
#             loss = engine_asgd.train_an_epoch(train_loader_asgd, epoch_id=epoch)
#             hit_ratio, ndcg = engine_asgd.evaluate(evaluate_data, epoch_id=epoch)
#             engine_asgd.save(config_asgd['alias'], epoch, hit_ratio, ndcg)
#             asgd_hr.append(hit_ratio) 
#             asgd_ndcg.append(ndcg)
#             asgd_loss.append(loss)   
#             
#             if (epoch+1) == config_asgd['num_epoch']:   
#                 with open('optimization/gmf_asgd.csv', 'w') as f:
#                     csvlogger = csv.writer(f, dialect='excel', delimiter=';')
#                     csvlogger.writerow(zip(asgd_hr, asgd_ndcg, asgd_loss))
                
        if x == 5: #6 RMSprop Optimization
            print('Epoch RMSprop {} starts !'.format(epoch))
            print('-' * 80)
            train_loader_rmsprop = sample_generator.instance_a_train_loader(config_rmsprop['num_negative'], config_rmsprop['batch_size'])
            loss = engine_rmsprop.train_an_epoch(train_loader_rmsprop, epoch_id=epoch)
            hit_ratio, ndcg = engine_rmsprop.evaluate(evaluate_data, epoch_id=epoch)
            engine_rmsprop.save(config_rmsprop['alias'], epoch, hit_ratio, ndcg)
            rmsprop_hr.append(hit_ratio) 
            rmsprop_ndcg.append(ndcg)
            rmsprop_loss.append(loss)   
            
            if (epoch+1) == config_rmsprop['num_epoch']:   
                with open('optimization/gmf_rmsprop.csv', 'w') as f:
                    csvlogger = csv.writer(f, dialect='excel', delimiter=';')
                    csvlogger.writerow(zip(rmsprop_hr, rmsprop_ndcg, rmsprop_loss))

                
        elif x == 6: #7 SGD Optimization
            print('Epoch SGD {} starts !'.format(epoch))
            print('-' * 80)
            train_loader_sgd = sample_generator.instance_a_train_loader(config_sgd['num_negative'], config_sgd['batch_size'])
            loss = engine_sgd.train_an_epoch(train_loader_sgd, epoch_id=epoch)
            hit_ratio, ndcg = engine_sgd.evaluate(evaluate_data, epoch_id=epoch)
            engine_sgd.save(config_sgd['alias'], epoch, hit_ratio, ndcg)
            sgd_hr.append(hit_ratio) 
            sgd_ndcg.append(ndcg)
            sgd_loss.append(loss)   
            
            if (epoch+1) == config_sgd['num_epoch']:   
                with open('optimization/gmf_sgd.csv', 'w') as f:
                    csvlogger = csv.writer(f, dialect='excel', delimiter=';')
                    csvlogger.writerow(zip(sgd_hr, sgd_ndcg, sgd_loss))
