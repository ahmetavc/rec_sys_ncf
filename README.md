# Recommendation System with Neural Collaborative Filtering
Group 82
```bash
Ahmet Avci
Laurenz Gutleder
Levent Guner
```

Classical recommender systems make predictions based on users historical behaviors like most machine learning techniques. The two most popular approaches are content-based and collaborative filtering and their goal is to predict user preferences for a set of items based on past experiences. This project is based on the paper ‘Neural Collaborative Filtering (NCF)’ [1] which investigated the performance based new deep learning approach for recommender systems. It was published by National University of Singapore, Columbia University, Shandong University and Texas AM University in 2017 and it utilizes the flexibility, complexity, and non-linearity of Neural Network to build a recommender system. We have repeated some of the experiments in the paper with different algorithm optimization techniques and furthermore focused on hyperparameter optimization

Our work is based on a pytorch implementation [2] of the paper. We just focused on the training process in the files train.py and utils.py. There are different branches created by each of us for different parts of the project:

- Branch 'ahmet': hyperparameter optimizations (learning rate, momentum, regularization)
- Branch 'laurenz': optimization algorithms (Adadelta, Adagrad, Adam, Adamax, ASGD, RMSprop, SGD)
- Branch 'levent': hyperparameter optimizations (neural network depth)

[1] He, X., Liao, L., Zhang, H., Liqiang, N., Hu, X. and Tat-Seng, C. (2017) Neural Collaborative Filtering. In
Proceedings of the 26th International Conference on World Wide Web, Perth. Link: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf

[2] https://github.com/LaceyChen17/neural-collaborative-filtering
