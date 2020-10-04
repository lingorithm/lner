import llck
import re
import os
import json
import numpy as np
import keras as k
from keras.utils import to_categorical
from keras.models import Model, Input, Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from future.utils import iteritems

class NER:
	"""Named Entity Recognition Class
	"""

	def __init__(self, lang, sentences, entities, **kwargs):
		"""init TC

		Args:
			lang (str): language for the classifier used for llck
			data (list): list of tuples 
		"""
		self.lang = lang
		self.nlp = llck(self.lang, {'tokenize': 'tokenize'})
		self.entities = entities
		self.sentences = sentences
		self.verbose = kwargs.get('verbose', False)
		self.none_tag = kwargs.get('none_tag', 'O')
		self.unk_tag = kwargs.get('unk_tag', 'UNK')
		self.data = []
		self.tags_value = {}
		self._uniqe_tags = set([])
		self.words = set([self.unk_tag])
		self.tags = set([self.none_tag])
		self.word_tag = {}
		
	def process(self):
		"""Pre pcrocess data before training 
		"""
		self.__pre_processing_tags()
		self.__pre_processing_words()
		self.__pre_processing()
		if self.verbose:
			self.data_summary()
		
	def __pre_processing_words(self):
		# get all tags value
		for sentence in self.sentences:
			_data = []
			for token in self.nlp.process(sentence[0]).sentences[0].tokens:
				for entity in sentence[1]:
					if token.start >= entity[0] and token.end <= entity[1]:
						token.tag = self.word_tag.get(
							token.data, self.none_tag)
				_data.append((token.data, token.tag))
				self.words.add(token.data)
			self.data.append(_data)

		self.n_words = len(self.words)
		self.word2idx = {w: i for i, w in enumerate(self.words)}
		self._max_len = max([len(s) for s in self.data])

	def __pre_processing_tags(self):
		# convert all entities to B- and I-
		for entity in self.entities:
			for ent in self.entities[entity]:
				for value in ent['values']:
					_value = value.split(' ')
					for v in range(len(_value)):
						if v == 0:
							self.tags.add('B-'+entity)
							self.word_tag[_value[v]] = 'B-'+entity
						else:
							self.tags.add('I-'+entity)
							self.word_tag[_value[v]] = 'I-'+entity
					if entity not in self.tags_value:
						self.tags_value[entity] = {}
					self.tags_value[entity][value] = {
						"key": ent["key"], "meta": ent.get("meta", None)}

		# create 
		self.n_tags = len(self.tags)
		self.tag2idx = {t: i for i, t in enumerate(self.tags)}
		self.idx2tag = {v: k for k, v in iteritems(self.tag2idx)}

	def __pre_processing(self): 
		x = []
		y = []
		# TODO: Optmization
		for sent in self.data:
			_x = []
			_y = []
			for w in sent:
				if len(w) == 2:
					_x.append(self.word2idx[w[0]])
					_y.append(self.tag2idx[w[1]])
			x.append(_x)
			y.append(_y)
		
		self.x = self.__pre_processing_x(x)
		self.y = self.__pre_processing_y(y)
	   

	def __pre_processing_x(self, x):
		return pad_sequences(maxlen=self._max_len, sequences=x,
							   padding="post", value=self.word2idx[self.unk_tag])

	def __pre_processing_y(self, y):
		y = pad_sequences(maxlen=self._max_len, sequences=y, padding="post", value=self.tag2idx[self.none_tag])
		return [to_categorical(i, num_classes=self.n_tags) for i in y]


	def data_summary(self):
		"""Print data summary
		"""
		print("Data Summary")
		print("Sentence Size = %d and Maximum length = %d" %
			  (len(self.data), self._max_len))
		print("Word Size = %d and Tags Size = %d" %
			  (self.n_words, self.n_tags))
		print("Tags = ", self.tags)

	def __pre_train(self):
		"""Split X and Y into training and validation data
		"""
		self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(
			self.x, self.y, shuffle=True, test_size=0.2)
		if self.verbose:
			print("Training Data")
			print("Shape of train_X = %s and train_Y = %s" %
				  (len(self.train_X), len(self.train_Y)))
			print("Shape of val_X = %s and val_Y = %s" %
				  (len(self.val_X), len(self.val_Y)))

	def train(self, epochs, batch_size, model_name):
		"""Traing the model

		Args:
			epochs (int): number of epochs
			batch_size (int): batch size
			model_name (str): model name
		"""
		self.__pre_train()
		self.__create_model()
		self.model_name = model_name
		checkpoint = ModelCheckpoint(
			"checkpoint/cp-%s-{epoch:04d}.ckpt" % (model_name), monitor='val_acc', verbose=1, save_best_only=True, mode='min')

		self.model.fit(self.train_X, np.array(self.train_Y), epochs=epochs, batch_size=batch_size,
					   validation_data=(self.val_X, np.array(self.val_Y)), callbacks=[checkpoint])
		self.save_model()

	def save_model(self):
		"""Save model to a file
		"""
		self.model.save("%s.h5" % (self.model_name))

	def __create_model(self):
		"""Create model layers
		"""
		input = Input(shape=(self._max_len,))
		word_embedding_size = self._max_len

		# Embedding Layer
		model = Embedding(input_dim=self.n_words,
						  output_dim=word_embedding_size, input_length=self._max_len)(input)

		# BI-LSTM Layer
		model = Bidirectional(LSTM(units=word_embedding_size,
								   return_sequences=True,
								   dropout=0.5, recurrent_dropout=0.5,
								   kernel_initializer=k.initializers.he_normal()))(model)
		model = LSTM(units=word_embedding_size * 2, dropout=0.5,
					 recurrent_dropout=0.5,
					 return_sequences=True,
					 kernel_initializer=k.initializers.he_normal())(model)
		# TimeDistributed Layer
		model = TimeDistributed(Dense(self.n_tags, activation="relu"))(model)

		# CRF Layer
		crf = CRF(self.n_tags)

		out = crf(model)  # output

		self.model = Model(input, out)
		#Optimiser
		adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

		# Compile model
		self.model.compile(optimizer=adam, loss=crf.loss_function, metrics=[
						   crf.accuracy, 'accuracy'])

		self.model.summary()

	def load(self, path):
		"""Load model from a path 

		Args:
			path (str): model file path 
		"""
		adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
		crf = CRF(self.n_tags)
		self.model = load_model(path, compile=False,
								custom_objects={'CRF': CRF})
		self.model.compile(optimizer=adam, loss=crf.loss_function, metrics=[
						   crf.accuracy, 'accuracy'])
		self.model.summary()


	def __tokenize(self, input):
		return [[token.get() for token in sent.tokens]
				for sent in self.nlp.process(input).sentences]

	def __predictions(self, text):
		"""Prediect function

		Args:
			text (str): input text

		Returns:
			list: list of predected classes number 
		"""
		tokens = self.__tokenize(text)
		X = self.__pre_processing_x(
			[[self.word2idx.get(w, self.word2idx[self.unk_tag]) for w in t] for t in tokens])
		return tokens, self.model.predict(X)

	def __get_final_output(self, pred):
		"""match predected number with actual classes

		Args:
			pred (list): list of predected classes number 

		Returns:
			list: list of tuples ordered by most matching class
		"""
		out = []
		for pred_i in pred:
			out_i = []
			for p in pred_i:
				p_i = np.argmax(p)
				out_i.append(self.idx2tag[p_i])
			out.append(out_i)
		return out


	def __pretty(self, tags, tokens): 
		entities = []
		_entity = []
		_ent_name = ''
		for i in range(len(tags)):
			if tags[i].startswith('B-'):
				_entity.append(tokens[i])
				_ent_name = tags[i].replace('B-', '')
			elif tags[i].startswith('I-') and len(_entity) >= 1:
				_entity.append(tokens[i])
			elif tags[i] == 'O':
				if len(_entity):
					_value = " ".join(_entity)
					entities.append(
						{**self.tags_value[_ent_name][_value], **{'value': _value, 'entity': _ent_name}})
				_entity = []
				_ent_name = ''
		return entities

	def recognize(self, text):
		"""Classify funtion

		Args:
			text (str): input text

		Returns:
			list: list of tuples ordered by most matching class
		"""
		tokens, pred = self.__predictions(text)
		tags = self.__get_final_output(pred)
		out = []
		for i in range(len(tags)):
			out.append(self.__pretty(tags[i], tokens[i]))
		return out, tokens

	def F1(self):
		test_pred = self.model.predict(self.val_X, verbose=1)
		pred_labels = self.__get_final_output(test_pred)
		test_labels = self.__get_final_output(self.val_Y)
		print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
