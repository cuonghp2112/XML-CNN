from gensim.models import KeyedVectors
import scipy
from scipy import sparse
import numpy as np
from header import *
# import sys
# sys.path.append('models')
# from xmlCNN import xmlCNN

# x_tr = sparse.load_npz(params.data_path + '/x_train.npz')
# y_tr = sparse.load_npz(params.data_path + '/y_train.npz')
x_te = sparse.load_npz(r"..\datasets\vlsp_data\x_test.npz")
y_te = sparse.load_npz(r"..\datasets\vlsp_data\y_test.npz")

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
vocabulary = np.load(r"..\datasets\vlsp_data\vocab.npy").item()
np.load = np_load_old
vocabulary_inv = np.load(r"..\datasets\vlsp_data\vocab_inv.npy")

embedding_model = KeyedVectors.load_word2vec_format(r"..\embedding_weights\baomoi.vn.model.bin", binary=True)

embedding_weights = [embedding_model[w] if w in embedding_model
                        else np.random.uniform(-0.25, 0.25, 300)
                        for w in vocabulary_inv]
embedding_weights = np.array(embedding_weights).astype('float32')

model = xmlCNN(params, embedding_weights)
model = load_model(model, r"..\saved_models\Gen_data_CNN_Z_dim-100_mb_size-20_hidden_dims-512_preproc-0_loss-BCELoss_sequence_length-500_embedding_dim-300_params.vocab_size=30000\model_best_test")

# x_te, _ = load_batch_cnn(x_te, y_te, params, batch=False)

x_te = x_te.todense()
# y_te = y_te.todense()
dtype_i = ""
if(torch.cuda.is_available()):
    dtype_f = torch.cuda.FloatTensor
    dtype_i = torch.cuda.LongTensor
    model = model.cuda()
else:
    dtype_f = torch.FloatTensor
    dtype_i = torch.LongTensor
x_te = Variable(torch.from_numpy(x_tr.astype('int')).type(dtype_i))
# y_te = Variable(torch.from_numpy(y_tr.astype('float')).type(params.dtype_f))

Y2 = np.zeros(y_te.shape)
rem = x_te.shape[0]%20
for i in range(0,x_te.shape[0] - rem,20):
    e_emb = model.embedding_layer.forward(x_te[i:i+20].view(20, x_te.shape[1]))
    Y2[i:i+20,:] = model.classifier(e_emb).cpu().data
if(rem):
    e_emb = model.embedding_layer.forward(x_te[-rem:].view(rem, x_te.shape[1]))
    Y2[-rem:,:] = model.classifier(e_emb).cpu().data

y_te = y_te.todense()
print(y_te)
print(Y2)