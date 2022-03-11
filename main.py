"""
2022-03-08
Noé Aubin-Cadot

Goal:
- Do a dimensional reduction of the Criterion movies collection.
- Find the movie whose synopsis is the closest to a given query.

Installation :
- pip3 install sentence-transformers
- pip3 install scikit-learn
- pip3 install umap-learn

Sources :
- BERT : https://pypi.org/project/sentence-transformers/
- UMAP : https://pypi.org/project/umap-learn/
- TSNE : https://pypi.org/project/scikit-learn/

Data :
- Criterion : https://www.kaggle.com/ikarus777/criterion-movies-collection
- IMDB      : kaggle ?

"""

################################################################################
################################################################################
# Import libraries

import sys
import pandas as pd
import numpy as np

np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(precision=1)
np.set_printoptions(precision=5)
#np.set_printoptions(precision=20)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 10)
pd.set_option('max_colwidth', 20)
pd.set_option('display.width', 1000)

################################################################################
################################################################################
# Define functions

def read_labels_and_corpus(input_file,col_label,col_text):
	df = pd.read_csv(input_file)
	columns = list(df.columns)
	if col_label not in columns:
		error_message = 'ERROR : the specified column « '+col_label+' » not found in df. QUIT.'
		sys.exit(error_message)
	if col_text not in columns:
		error_message = 'ERROR : the specified column « '+col_text+' » not found in df. QUIT.'
		sys.exit(error_message)
	labels = list(df[col_label].fillna(''))
	corpus = list(df[col_text].fillna(''))
	return labels,corpus

def embed_the_corpus(input_file,col_label,col_text,output_file,sentence_transformer_model):
	print('Import data...')
	labels,corpus = read_labels_and_corpus(input_file,col_label,col_text)
	print('Import sentence_transformers...')
	from sentence_transformers import SentenceTransformer
	print('Define the embedding...')
	embedder = SentenceTransformer(sentence_transformer_model)
	print('Embed the corpus... (please wait, this can be long...)')
	X_embedding = embedder.encode(corpus)
	print('Format the embedding in a Pandas DataFrame...')
	df_embedding = pd.DataFrame(X_embedding,index=labels)
	columns = list(df_embedding.columns)
	df_embedding.index.name = 'label'
	df_embedding['text'] = corpus
	df_embedding = df_embedding[['text']+columns] # reorder columns
	print('Export the data...')
	df_embedding.to_csv(output_file)
	print('Done.')

def embed_the_queries(input_file,output_file,sentence_transformer_model):
	print('Import data...')
	df_queries   = pd.read_csv(input_file)
	queries      = list(df_queries['query'])
	print('Import sentence_transformers...')
	from sentence_transformers import SentenceTransformer
	print('Define the embedding...')
	embedder = SentenceTransformer(sentence_transformer_model)
	print('Embed the queries... (please wait, this can be long...)')
	X_embedding = embedder.encode(queries)
	print('Format the embedding in a Pandas DataFrame...')
	df_embedding = pd.DataFrame(X_embedding,index=queries)
	df_embedding.index.name = 'query'
	print('Export the data...')
	df_embedding.to_csv(output_file)
	print('Done.')

def normalize_matrix_along_axis(X,axis=1):
	X_norms = np.sqrt((X*X).sum(axis=1)).reshape(-1,1)
	X /= X_norms
	return X

def embedding_to_similarities(X,similarity_method='euclidean'):
	if similarity_method=='cosine':
		X = normalize_matrix_along_axis(X)
		# cos(theta)
		# = (v.w)/(|v|*|w|)   where "." is the dot product between the vectors v and w.
		# =  v.w              because v and w were normalized above    
		X_similarities = X.dot(X.T)
	elif similarity_method=='euclidean':
		# Not memory efficient and pretty slow:
		#m,n=X.shape
		#X_ver = X.reshape(m,n,1)
		#X_hor = X.T.reshape(1,n,m)
		#X_distances = np.sqrt(((X_ver-X_hor)**2).sum(axis=1))
		#X_similarities = 1/(1+X_distances)
		# Memory efficient and much faster:
		from sklearn.metrics.pairwise import euclidean_distances
		X_distances    = euclidean_distances(X,X)
		X_similarities = 1/(1+X_distances)
	return X_similarities

def similarities_to_dissimilarities(X):
	X[X>1]  =  1
	X[X<-1] = -1
	# Choose a dissimilarity :
	#X_dissimilarities = 1-X
	#X_dissimilarities = np.sqrt(1-X)
	X_dissimilarities = (1-X)/(1+X)
	return X_dissimilarities

def dissimilarities_to_dimensional_reduction(X,model_type='TSNE'):
	if model_type=='TSNE':
		from sklearn.manifold import TSNE
		model = TSNE(n_components=2, random_state=0, metric='precomputed', init='random', learning_rate='auto', square_distances=True)
	elif model_type=='UMAP':
		from umap import UMAP
		model = UMAP(metric='precomputed',random_state=0)
	X_2d = model.fit_transform(X)
	return X_2d

def embedding_to_dimensional_reduction(X,model_type='TSNE'):
	if model_type=='TSNE':
		from sklearn.manifold import TSNE
		model = TSNE(n_components=2, random_state=0, init='random', learning_rate='auto', square_distances=True)
	elif model_type=='UMAP':
		from umap import UMAP
		model = UMAP(random_state=0)
	X_2d = model.fit_transform(X)
	return X_2d


def colorful_scatter_plot(X_2d,labels,model_type,do_cosine_similarity):
	import random
	random.seed(0)
	import matplotlib.pyplot as plt
	fig,ax = plt.subplots(figsize=(18,9))
	plt.scatter(X_2d[:,0],X_2d[:,1], marker='o', s=10, edgecolor='None')
	do_add_labels=1
	if do_add_labels:
		for i in range(len(labels)):
			x,y=X_2d[i,:]
			s=labels[i]
			# Can be very pale:
			#R = random.random()
			#G = random.random()
			#B = random.random()
			# Can be too dark:
			#R = random.random()/2. + 0.25
			#G = random.random()/2. + 0.25
			#B = random.random()/2. + 0.25
			# Seems like a good balance:
			R = random.random()/2. + 0.35
			G = random.random()/2. + 0.35
			B = random.random()/2. + 0.35
			color = (R,G,B)
			plt.text(x,y,s,color=color,size=7)
	title = 'Dimensional reduction ('+model_type+')'
	if do_cosine_similarity:
		title = title + ' of the cosines similarities'
	plt.title(title,size=18)
	plt.tight_layout()
	plt.show()

def scatterplot_dimensional_reduction_of_the_embedding(input_file,model_type,do_cosine_similarity):
	print('Import data...')
	df_embedding = pd.read_csv(input_file)
	labels       = list(df_embedding['label'])
	corpus       = list(df_embedding['text'])
	X_embedding  = df_embedding.drop(columns=['label','text']).values
	if do_cosine_similarity:
		print('Compute similarities...')
		X_similarities = embedding_to_similarities(X_embedding)
		print('Compute dissimilarities...')
		X_dissimilarities = similarities_to_dissimilarities(X_similarities)
		print('Compute dimensional reduction...')
		X_2d = dissimilarities_to_dimensional_reduction(X_dissimilarities,model_type)
	else:
		print('Compute dimensional reduction...')
		X_2d = embedding_to_dimensional_reduction(X_embedding,model_type)
	print('Scatter plot the dimensional reduction...')
	colorful_scatter_plot(X_2d,labels,model_type,do_cosine_similarity)

def export_pairwise_similarity(input_file,output_file):
	df_embedding = pd.read_csv(input_file)
	labels       = list(df_embedding['label'])
	corpus       = list(df_embedding['text'])
	X_embedding  = df_embedding.drop(columns=['label','text']).values
	print('Compute similarities...')
	X_similarities  = embedding_to_similarities(X_embedding)
	df_similarities = pd.DataFrame(X_similarities,index=labels,columns=labels).round(6)
	df_similarities.index.name = 'label'
	print('Export similarities...')
	df_similarities.to_csv(output_file)
	print('Done.')

def compute_pairwise_similarity_between_X_and_Y(X,Y,similarity_method='euclidean'):
	if similarity_method=='cosine': # via cosine similarity
		X = normalize_matrix_along_axis(X,axis=1)
		Y = normalize_matrix_along_axis(Y,axis=1)
		XY_similarities = X.dot(Y.T)
	elif similarity_method=='euclidean': # via Euclidean distance
		from sklearn.metrics.pairwise import euclidean_distances
		XY_distances    = euclidean_distances(X,Y)
		XY_similarities = 1/(1+XY_distances)

		#m_X,n_X=X.shape
		#m_Y,n_Y=Y.shape
		#if n_X!=n_Y:
		#	error_message = 'ERROR : n_X != n_Y. QUIT.'
		#	sys.exit(error_message)
		#else:
		#	n = n_X
		#X = X.reshape(m_X,n,1)
		#Y = Y.T.reshape(1,n,m_Y)
		#XY_distances = np.sqrt((X-Y**2).sum(axis=1))
		#XY_similarities = 1/(1+XY_distances)
	return XY_similarities

def find_closest_texts_from_queries(input_file_embedding_corpus,
									input_file_embedding_queries,
									similarity_method):
	print('Import corpus embedding data...')
	df_corpus  = pd.read_csv(input_file_embedding_corpus)
	labels     = list(df_corpus['label'])
	corpus     = list(df_corpus['text'])
	X_corpus   = df_corpus.drop(columns=['label','text']).values
	print('Import queries embedding data...')
	df_queries = pd.read_csv(input_file_embedding_queries)
	queries    = list(df_queries['query'])
	X_queries  = df_queries.drop(columns=['query']).values
	print('Compute pairwise distance between queries and texts...')
	XY_similarities = compute_pairwise_similarity_between_X_and_Y(X_corpus,X_queries,similarity_method)
	print('Find the closest element to each query...')
	for i in range(len(queries)):
		query = queries[i]
		print('\n---------------------\n\nQuery =',query)
		similarities_from_query = XY_similarities[:,i]
		max_index_row = np.argmax(similarities_from_query)
		print('\nThe closest element found is the text number '+str(max_index_row)+'.')
		print('Matching score = '+str(round(100*similarities_from_query[max_index_row],2))+'%')
		print('\n'+labels[max_index_row])
		print('\n'+corpus[max_index_row])

################################################################################
################################################################################
# Use the functions


if __name__ == '__main__':

	# Choose a sentence transformer model from here :
	# https://www.sbert.net/docs/pretrained_models.html
	# https://huggingface.co/sentence-transformers
	sentence_transformer_model = 'bert-base-nli-mean-tokens' # deprecated but faster on macbook 13" without a GPU
	#sentence_transformer_model = 'all-mpnet-base-v2' # results less good and takes forever to compute

	# ---------------------------------------------------------
	# Embedding the corpus and the queries

	# Criterion movies corpus (1250 movies)
	# Run this once to embed the Criterion movies corpus
	# For bert-base-nli-mean-tokens :
	# 13" Macbook without a GPU  : between 5 to 6 minutes
	# Google Colab Pro with GPUs : 17 seconds
	do_embed_the_corpus_criterion=0
	if do_embed_the_corpus_criterion:
		input_file  = 'input/Criterion_movies_clean.csv'
		col_label   = 'Title'
		col_text    = 'Description'
		output_file = 'output/'+sentence_transformer_model+'/Criterion_movies_clean_embedded.csv'
		embed_the_corpus(input_file=input_file,
						col_label=col_label,
						col_text=col_text,
						output_file=output_file,
						sentence_transformer_model=sentence_transformer_model)

	# IMDB corpus (85855 movies)
	# Run this once to embed the Criterion movies corpus
	# For bert-base-nli-mean-tokens :
	# Macbook 13" without GPU   : about 7 hours
	# Google Colab Pro with GPU : about 3 minutes
	do_embed_the_corpus_imdb=0
	if do_embed_the_corpus_imdb:
		input_file  = 'input/IMDB_movies.csv'
		col_label   = 'title'
		col_text    = 'description'
		output_file = 'output/'+sentence_transformer_model+'/IMDB_movies_embedded.csv'
		embed_the_corpus(input_file=input_file,
						col_label=col_label,
						col_text=col_text,
						output_file=output_file,
						sentence_transformer_model=sentence_transformer_model)

	# Run this to embed the queries
	do_embed_the_queries=0
	if do_embed_the_queries:
		input_file  = 'input/queries.csv'
		output_file = 'output/'+sentence_transformer_model+'/queries_embedded.csv'
		embed_the_queries(input_file=input_file,
						output_file=output_file,
						sentence_transformer_model=sentence_transformer_model)

	# ---------------------------------------------------------
	# Having fun with the embedding

	# To get the pairwise similarities between the Criterion movies
	do_export_pairwise_similarity=0
	if do_export_pairwise_similarity:
		input_file  = 'output/'+sentence_transformer_model+'/Criterion_movies_clean_embedded.csv'
		output_file = 'output/'+sentence_transformer_model+'/Criterion_movies_clean_similarities.csv'
		export_pairwise_similarity(input_file,output_file)

	# To get a 2d scatter plot of the dimensional reduction of the corpus embedding
	do_scatterplot_dimensional_reduction_of_the_embedding=0
	if do_scatterplot_dimensional_reduction_of_the_embedding:
		input_file = 'output/'+sentence_transformer_model+'/Criterion_movies_clean_embedded.csv'
		#model_type = 'TSNE'
		model_type = 'UMAP'
		do_cosine_similarity = 0
		#do_cosine_similarity = 1
		scatterplot_dimensional_reduction_of_the_embedding(input_file=input_file,
															model_type=model_type,
															do_cosine_similarity=do_cosine_similarity)

	# ---------------------------------------------------------
	# Finding the perfect movie for tonight

	# To get the closest texts from the queries (Criterion movies)
	do_find_closest_texts_from_queries=0
	if do_find_closest_texts_from_queries:
		input_file_embedding_corpus  = 'output/'+sentence_transformer_model+'/Criterion_movies_clean_embedded.csv'
		input_file_embedding_queries = 'output/'+sentence_transformer_model+'/queries_embedded.csv'
		similarity_method            = 'euclidean'
		#similarity_method            = 'cosine'
		find_closest_texts_from_queries(input_file_embedding_corpus=input_file_embedding_corpus,
										input_file_embedding_queries=input_file_embedding_queries,
										similarity_method=similarity_method)

	# To get the closest texts from the queries (IMDB movies)
	do_find_closest_texts_from_queries=0
	if do_find_closest_texts_from_queries:
		input_file_embedding_corpus  = 'output/'+sentence_transformer_model+'/IMDB_movies_embedded.csv'
		input_file_embedding_queries = 'output/'+sentence_transformer_model+'/queries_embedded.csv'
		similarity_method            = 'euclidean'
		#similarity_method            = 'cosine'
		find_closest_texts_from_queries(input_file_embedding_corpus=input_file_embedding_corpus,
										input_file_embedding_queries=input_file_embedding_queries,
										similarity_method=similarity_method)









