from tensorboardX import SummaryWriter
import datetime
import os
import torch

class Logger():
	def __init__(self):
		self.m_tensor_writer = None
		self.m_io_writer = None
	
	def addIOWriter(self, args):
		myhost = os.uname()[1]
		file_time = datetime.datetime.now().strftime('%H_%M_%d_%m')
		output_file = myhost+"_"+file_time
		
		hidden_size = args.hidden_size
		batch_size = args.batch_size
		embedding_dim = args.batch_size
		optimizer_type = args.optimizer_type
		lr = args.lr
		window_size = args.window_size
		shared_embedding = args.shared_embedding
		data_name = args.data_name

		output_file = output_file+"_"+str(hidden_size)+"_"+str(batch_size)+"_"+str(embedding_dim)+"_"+str(optimizer_type)+"_"+str(lr)+"_"+str(window_size)+"_"+str(shared_embedding)

		output_file = output_file+"_"+str(data_name)
		print("output_file", output_file)
		self.m_io_writer = open(output_file, "w")

		model_name = args.model_name
		tensor_file_name = myhost+"_"+file_time
		self.m_tensor_writer = SummaryWriter("../tensorboard/"+model_name+"/"+tensor_file_name)

	def addOutput2IO(self, outputString):
		print(outputString)
		self.m_io_writer.write(outputString+"\n")
		self.m_io_writer.flush()

	def addScalar2Tensorboard(self, name, value, index):
		self.m_tensor_writer.add_scalar('data/'+name, value, index)

	def addHistogram2Tensorboard(self, name, value, index):
		print("----"*10)
		print(torch.max(value), torch.min(value))
		print(torch.sum(value))
		print("----"*10)

		self.m_tensor_writer.add_histogram('data/'+name, value, index, 'tensorflow')
	
	def closeIOWriter(self):
		self.m_io_writer.close()

	def closeTensorWriter(self):
		self.m_tensor_writer.close()