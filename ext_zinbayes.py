import zinbayes.zinbayes as zb
import pandas as pd
import numpy as np
import edward as ed
import itertools 
import re
import os


class CSVDataset:
	"""counts_file,batches_file, labels_file must be csv files 
	- It extracts cell lables if cells are identified as <Label>_<Number>"""

	def __init__(self, folder,counts_file, batches_file=None, labels_file=None):
		X = pd.read_csv(os.path.join(folder, counts_file))
		if labels_file	is None:
			cell_names = X.columns.values[1:] #column 0 is not a cell name
			labels = [re.sub(r'_[0-9]+', '', cell) for cell in cell_names]
			self.cell_types, self.labels = np.unique(labels,return_inverse=True)

		else:
			labels = pd.read_csv(os.path.join(folder, labels_file), header=0, index_col=0)
			self.cell_types, self.labels = np.unique(labels.values,return_inverse=True)

		self.batches,self.batch_names = None,None
		if batches_file is not None:
			batches	 = pd.read_csv(os.path.join(folder, batches_file),header=0, index_col=0)
			self.batches, self.batch_idx = np.unique(batches.values,return_inverse=True)
		
		self.X = np.array(X)
		self.gene_names = self.X[:,0]
		self.X = self.X[:,1:].astype(np.float32).T
	




class ExtZINBayes(zb.ZINBayes):
	def __init__(self, n_components=10, n_mc_samples=1, gene_dispersion=True, zero_inflation=True, scalings=True, batch_correction=False,
	 test_iterations=100, optimizer=None, minibatch_size=None, validation=False, X_test=None):
		super(ExtZINBayes, self).__init__(n_components=n_components, n_mc_samples=n_mc_samples, gene_dispersion=gene_dispersion,
		zero_inflation=zero_inflation, scalings=scalings, batch_correction=batch_correction,
		test_iterations=test_iterations, optimizer=optimizer, minibatch_size=minibatch_size, validation=validation, X_test=X_test)
		


	def sample_rho_values(self, cells1, cells2, n_genes, n_samples=10): #n_samples -> number of mc samples
		n_t1, n_t2 = (np.where(cells1)[0]).shape[0], (np.where(cells2)[0]).shape[0]
		all_rhos1 = np.array([np.zeros(shape=(n_t1, n_genes)) for _ in range(n_samples)])
		all_rhos2 = np.array([np.zeros(shape=(n_t2, n_genes)) for _ in range(n_samples)])
		sample_z = self.qz.sample(n_samples).eval()
		sample_w = self.qW0.sample(n_samples).eval()
		for s in range(n_samples):
			rhos = self.sess.run(self.rho, feed_dict={self.z: sample_z[s], self.W0: sample_w[s]})
			all_rhos1[s] = rhos[cells1,:]
			all_rhos2[s] = rhos[cells2,:]
		return all_rhos1, all_rhos2


	def ensemble_pairs_by_bacthes(self, t1_cells, t2_cells, batches_names, batches,n_pairs=None):
		pairs, pairs_per_batch = [], []
		n_batches = batches_names.shape[0]
		b_t1, b_t2 = batches[t1_cells], batches[t2_cells]
		for b in range(0,n_batches):
			b_t1_ix, b_t2_ix = np.where(b_t1 == b)[0], np.where(b_t2 == b)[0]
			b_pairs = [b_t1_ix, b_t2_ix]
			pairs_per_batch.append(np.array(list(itertools.product(*b_pairs))))
		
		if n_pairs is not None:
			for b in range(0,n_batches):
				prop = np.count_nonzero(batches == b)/batches.shape[0]
				b_pairs = np.random.choice(np.arange(pairs_per_batch[b].shape[0]),size=round(n_pairs*prop), replace=False)
				pairs += pairs_per_batch[b][b_pairs,:].tolist()
		else:
			for b in range(0,n_batches):
				pairs += pairs_per_batch[b].tolist()
		return np.array(pairs)



	def ensemble_pairs(self, t1_cells, t2_cells, n_pairs=None):
		t1_ix, t2_ix = np.arange((np.where(t1_cells)[0]).shape[0]), np.arange((np.where(t2_cells)[0]).shape[0])
		pairs = [t1_ix, t2_ix]
		pairs = np.array(list(itertools.product(*pairs)))
		if n_pairs is not None:
			sub_pairs = np.random.choice(np.arange(pairs.shape[0]),size=n_pairs, replace=False)
			pairs = pairs[sub_pairs,:]
		return pairs



	def differential_expression_scores(self, types, labels, type1, type2=None, batches_names=None, batches=None, n_samples=10, n_pairs=None):
		types = list(types)
		t1 = types.index(type1)
		cells_t1 = labels == t1
		if type2 is not None:
			t2 = types.index(type2)
			cells_t2 = labels == t2
		else:
			cells_t2 = labels != t1

		if batches_names is None:
			pairs = self.ensemble_pairs(cells_t1, cells_t2, n_pairs)
		else:
			print("DE considering batches")
			pairs = self.ensemble_pairs_by_bacthes(cells_t1, cells_t2, batches_names, batches, n_pairs)
		n_pairs = len(pairs)
		n_genes = self.W0.shape[1]
		rho1_values, rho2_values = self.sample_rho_values(cells_t1, cells_t2, n_genes, n_samples)		
		print("rho1_values shape: " + str(rho1_values.shape)) # tem 3 dim (n_samples, n_cells, n_genes)
		print("rho2_values shape: " + str(rho2_values.shape)) # tem 3 dim (n_samples, n_cells, n_genes)

		bayes_factors =  np.zeros(shape=(n_pairs, n_genes)) #[]
		curr_pair = 0 
		cell1_rho, cell2_rho = np.zeros(shape=(n_samples,n_genes)), np.zeros(shape=(n_samples,n_genes))
		for pair in pairs:
			for i in range(n_samples): #rho1_values.shape[0] & rho2_values.shape[0] = n_samples
				cell1_rho[i] = rho1_values[i,pair[0],:] #get rho samples for cell of type 1
				cell2_rho[i] = rho2_values[i,pair[1],:] #get rho samples for cell of type 2
			h1 = cell1_rho > cell2_rho # compare n_samples
			means = np.mean(h1, axis=0)# means.shape = (1,genes)
			pair_bayes = np.log(means + 1e-8) - np.log(1-means + 1e-8) 
			bayes_factors[curr_pair] = pair_bayes
			curr_pair += 1

		means = np.mean(bayes_factors,axis=0) #bayes_factors matrix with shape (M_pairs, n_genes)
		return means #np.log(means + 1e-8) - np.log(1-means + 1e-8)



	def differentially_expressed_genes(self, gene_names, scores, threshold=0):
		abs_factors = np.abs(scores)
		df = pd.DataFrame(data={'Gene': gene_names,'factor': abs_factors})
		res = df.loc[df['factor'] >= threshold]
		res = res.sort_values(by=['factor'], ascending=False)
		return	res