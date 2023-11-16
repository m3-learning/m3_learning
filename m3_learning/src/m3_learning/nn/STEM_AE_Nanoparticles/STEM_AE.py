import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from ..Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
from util.file_IO import make_folder
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import warnings 
warnings.filterwarnings("ignore")
from viz.layout import find_nearest
import time
from datetime import date
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from viz.nn import get_theta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




import sys
import os
# sys.path.append(os.path.abspath("./../STEM_AE")) # or whatever the name of the immediate parent folder is

class ConvAutoencoder():
    """builds the convolutional autoencoder
    """

    def __init__(self,
                 encoder_step_size,
                 pooling_list,
                 decoder_step_size,
                 upsampling_list,
                 embedding_size,
                 conv_size,
                 device,
                 checkpoint = None,
                 learning_rate=3e-5,
                 emb_h5_path = './Combined_all_samples/embeddings.h5',
                 gen_h5_path = './Combined_all_samples/generated.h5',
                 ):
        """Initialization function

        Args:
            encoder_step_size (list): sets the size of the encoder
            pooling_list (list): sets the pooling list to define the pooling layers
            decoder_step_size (list): sets the size of the decoder
            upsampling_list (list): sets the size for upsampling
            embedding_size (int): number of embedding channels
            conv_size (int): sets the number of convolutional neurons in the model
            device (torch.device): set the device to run the model
            learning_rate (float, optional): sets the learning rate for the optimizer. Defaults to 3e-5.
        """
        self.encoder_step_size = encoder_step_size
        self.pooling_list = pooling_list
        self.decoder_step_size = decoder_step_size
        self.upsampling_list = upsampling_list
        self.embedding_size = embedding_size
        self.conv_size = conv_size
        self.device = device
        self.learning_rate = learning_rate

        self.checkpoint = checkpoint
        # self.train = train

        self.emb_h5_path = emb_h5_path
        self.gen_h5_path = gen_h5_path

        # complies the network
        self.compile_model()

    def open_embedding_h(self):
        return h5py.File(self.emb_h5_path)
    
    def open_generated_h(self):
        return h5py.File(self.gen_h5_path)
        # check = checkpoint.split('/')[-1][:-4]
        # return h[check]
    
    def compile_model(self):
        """function that complies the neural network model
        """
        # builds the encoder
        self.encoder = Encoder(
            original_step_size=self.encoder_step_size,
            pooling_list=self.pooling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            device=self.device
        ).to(self.device)

        # builds the decoder
        self.decoder = Decoder(
            original_step_size=self.decoder_step_size,
            upsampling_list=self.upsampling_list,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            pooling_list=self.pooling_list,
        ).to(self.device)

        # builds the autoencoder
        self.autoencoder = AutoEncoder(
            self.encoder, self.decoder, self.embedding_size,
            device=self.device).to(self.device)

        # sets the optimizers
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(), lr=self.learning_rate
        )

        # sets the datatype of the model to float32
        self.autoencoder.type(torch.float32)

    ## TODO: implement embedding calculation/unshuffler
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              save_emb_every=10):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
            save_emb_every (int, optional): 
        """
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)

        # builds the dataloader
        self.DataLoader_ = DataLoader(
            data, batch_size=batch_size, shuffle=True)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_+1

        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            if epoch % save_emb_every ==0: # tell loss function to give embedding
                print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
                print('.............................')
                fill_embeddings = self.get_embedding(data, train=True,check='temp')


            train = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, ln_parm,
                fill_embeddings=fill_embeddings)
            train_loss = train
            train_loss /= len(self.DataLoader_)
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')

          #  schedular.step()
            if best_train_loss > train_loss:
                best_train_loss = train_loss
                checkpoint = {
                    "net": self.autoencoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch,
                    "encoder": self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                if epoch >= 0:
                    lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
                    file_path = folder_path + f'/{save_date}_' +\
                        f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                        f'_trainloss:{train_loss:.4f}.pkl'
                    self.checkpoint = file_path
                    torch.save(checkpoint, file_path)

            if epoch%save_emb_every==0:
                with h5py.File(self.emb_h5_path,'r+') as h:
                    check = file_path.split('/')[-1][:-4]
                    h[f'embedding_{check}'] = h[f'embedding_temp']
                    h[f'scaleshear_{check}'] = h[f'scaleshear_temp']
                    h[f'rotation_{check}'] = h[f'rotation_temp'] 
                    h[f'translation_{check}'] = h[f'translation_temp']
                    del h[f'embedding_temp']         
                    del h[f'scaleshear_temp']          
                    del h[f'rotation_temp']            
                    del h[f'translation_temp']
                            
            if scheduler is not None:
                scheduler.step()

    ## TODO: why unshuffler taking so long? (just use h100 instead of titan)
    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)
        
        for idx,x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
            # tic = time.time()
            sorted, indices = torch.sort(idx)
            sorted = sorted.detach().numpy()

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x = self.autoencoder(x)
            else: embedding, sd, mn, predicted_x = self.autoencoder(x)

            reg_loss_1 = coef*torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5

            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)

            # reconstruction loss
            mask = (predicted_x!=0)
            loss = F.mse_loss(x, predicted_x, reduction='mean');
            loss = (loss*mask.float()).sum()
            loss /= mask.sum()

            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            train_loss += loss.item()

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()
            
            # fill embedding if the correct epoch
            if fill_embeddings:
                scale_shear,rotation,translation = self.autoencoder.temp_affines
                self.embedding[sorted] = embedding[indices].cpu().detach().numpy()
                self.scale_shear[sorted] = scale_shear[indices].cpu().reshape((-1,6)).detach().numpy()
                self.rotation[sorted] = rotation[indices].reshape((-1,6)).cpu().detach().numpy()
                self.translation[sorted] = translation[indices].reshape((-1,6)).cpu().detach().numpy()

        return train_loss

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['net'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        check = path_checkpoint.split('/')[-1][:-4]

        try:
            h = h5py.File(self.emb_h5_path,'r+')
            self.embedding = h[f'embedding_{check}']
            self.scale_shear = h[f'scaleshear_{check}']
            self.rotation = h[f'rotation_{check}']                    
            self.translation = h[f'translation_{check}']
        except Exception as error:
            print(error)
            print('Embedding and affines not opened')

        try:
            h = h5py.File(self.gen_h5_path,'r+')
            self.generated = h[check]
        except Exception as error:
            print(error)
            print('Generated not opened')

    def get_embedding(self, data, batch_size=32,train=True,check=None):
        """extracts embeddings from the data

        Args:
            data (torch.tensor): data to get embeddings from
            batch_size (int, optional): batchsize for inference. Defaults to 32.

        Returns:
            torch.tensor: predicted embeddings
        """

        # builds the dataloader
        dataloader = DataLoader(data, 
            batch_size, shuffle=False)

        try:
            try: 
                h = h5py.File(self.emb_h5_path,'r+')
            except: 
                h = h5py.File(self.emb_h5_path,'w')
                print('creating new embeddings.h5 file')

            if check==None: check = self.checkpoint.split('/')[-1][:-4]
            try: 
                embedding_ = h[f'embedding_{check}']
                scale_shear_ = h[f'scaleshear_{check}']
                rotation_ = h[f'rotation_{check}']                    
                translation_ = h[f'translation_{check}']
            except:
                embedding_ = h.create_dataset(f'embedding_{check}', data = np.zeros([data.shape[0], self.embedding_size]))
                scale_shear_ = h.create_dataset(f'scaleshear_{check}', data = np.zeros([data.shape[0],6]))
                rotation_ = h.create_dataset(f'rotation_{check}', data = np.zeros([data.shape[0],6]))
                translation_ = h.create_dataset(f'translation_{check}', data = np.zeros([data.shape[0],6]))
                print('creating new embedding and affine h5 datasets')

            self.embedding = embedding_
            self.scale_shear = scale_shear_
            self.rotation = rotation_
            self.translation = translation_

        except Exception as error:
            print(error) 
            assert self.train,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
             
        h.close()
                
        if train: 
            print('Created empty h5 embedding datasets to fill during training')
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            for i, (_,x) in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                with torch.no_grad():
                    with h5py.File(self.emb_h5_path,'r+') as h:       
                        value = x
                        test_value = Variable(value.to(self.device))
                        test_value = test_value.float()
                        embedding,scale_shear,rotation,translation = self.encoder(test_value)                        
                        h[f'embedding_{check}'][i*batch_size:(i+1)*batch_size, :] = embedding.cpu().detach().numpy()
                        h[f'scaleshear_{check}'][i*batch_size:(i+1)*batch_size, :] = scale_shear.reshape(-1,6).cpu().detach().numpy()
                        h[f'rotation_{check}'][i*batch_size:(i+1)*batch_size, :] = rotation.reshape(-1,6).cpu().detach().numpy()
                        h[f'translation_{check}'][i*batch_size:(i+1)*batch_size, :] = translation.reshape(-1,6).cpu().detach().numpy()
        h.close()

    def stack_emb_affines(self):
        h = self.open_embedding_h()
        check = self.checkpoint.split('/')[-1][:-4]
        scaler = StandardScaler()
        emblist = [h[f'embedding_{check}'][:,i].reshape(-1,1) for i in range(32)]
        scaled_embs = list(map( scaler.fit_transform,emblist) )
        rotation = scaler.fit_transform(get_theta(h[f'rotation_{check}']).reshape(-1,1))
        translationx = scaler.fit_transform(h[f'translation_{check}'][:,2].reshape(-1,1))
        translationy = scaler.fit_transform(h[f'translation_{check}'][:,5].reshape(-1,1))
        scalex = scaler.fit_transform(h[f'scaleshear_{check}'][:,0].reshape(-1,1))
        shearx = scaler.fit_transform(h[f'scaleshear_{check}'][:,1].reshape(-1,1))
        scaley = scaler.fit_transform(h[f'scaleshear_{check}'][:,4].reshape(-1,1))
        shearx = scaler.fit_transform(h[f'scaleshear_{check}'][:,3].reshape(-1,1))
        scaled_embs+=[rotation,translationx,translationy,scalex,shearx,scaley,shearx]
        return np.stack(scaled_embs,axis=1).squeeze()

    def get_clusters(self,dset,scaled_array,n_components=None,n_clusters=None):
        if n_components == None:
            print('Getting scree plot...')
            pca = PCA()
            pca.fit(scaled_array)
            plt.clf()
            plt.plot(pca.explained_variance_ratio_.cumsum(),marker='o')
            plt.show()
            n_components = int(input("Choose number of PCA components: "))
            
        print(f'PCA with {n_components} components...')
        pca = PCA(n_components)
        transformed = pca.fit_transform(scaled_array)
        
        if n_clusters == None:
            print('Getting elbow plot...')
            wcss = []
            for i in tqdm(range(10,self.embedding_size+7)):
                kmeans_pca = KMeans(n_clusters=i,init='k-means++',random_state=42)
                kmeans_pca.fit(transformed[::100])
                wcss.append(kmeans_pca.inertia_)
            plt.clf()
            plt.plot(range(10,self.embedding_size+7),wcss,marker='o')
            plt.show()
            n_clusters = int(input('Choose number of clusters: '))
            
        print(f'Clustering with {n_clusters} clusters...')
        kmeans_pca = KMeans(n_clusters=n_clusters,init='k-means++',random_state=42)
        kmeans_pca.fit(transformed)
        cluster_list = []
        for i,particle in enumerate(dset.meta['particle_list']):
            img = kmeans_pca.labels_[dset.meta['particle_inds'][i]:dset.meta['particle_inds'][i+1] ].reshape(dset.meta['shape_list'][i][0],dset.meta['shape_list'][i][1])
            cluster_list.append(img)
        print('Done')
        
        return cluster_list,kmeans_pca.labels_

    def generate_by_range(self,meta,checkpoint,
                         ranges=None,
                         generator_iters=50,
                         averaging_number=100,
                         overwrite=False,
                         with_affine=False,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space.
        Saves to embedding h5 dataset

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        assert not self.train, 'set self.train to False if calculating manually'
        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        channels = np.arange(self.embedding_size)
        # sets the channels to use in the object
        if "channels" in kwargs:
            channels = kwargs["channels"]
        if "ranges" in kwargs:
            ranges = kwargs["ranges"]

        # gets the embedding if a specific embedding is not provided
        try:
            embedding = self.embedding
        except Exception as error:
            print(error)
            assert False, 'Make sure model is set to appropriate embeddings first'

        try: # try opening h5 file
            try: # make new file
                h = h5py.File(self.gen_h5_path,'w')
            except: # open existing file
                h = h5py.File(self.gen_h5_path,'r+')

            check = checkpoint.split('/')[-1][:-4]
            try: # make new dataset
                if overwrite and check in h: del h[check]
                self.generated = h.create_dataset(check,
                                            data=np.zeros( [len(meta['particle_list']),
                                                            generator_iters,
                                                            len(channels),
                                                            128,128] ) )
                self.generated.attrs['clustered']=False    

            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        for p,p_name in enumerate(meta['particle_list']): # each sample
            print(p, p_name)
            # shape 128*128, 32
            data=self.embedding[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
                
            # loops around the number of iterations to generate
            for i in tqdm(range(generator_iters)):

                # loops around all of the embeddings
                for j, channel in enumerate(channels):

                    if ranges is None: # span this range when generating
                        ranges = np.stack((np.min(data, axis=0),
                                        np.max(data, axis=0)), axis=1)

                    # linear space values for the embeddings
                    value = np.linspace(ranges[j][0], ranges[j][1],
                                        generator_iters)

                    # finds the nearest points to the value and then takes the average
                    # average number of points based on the averaging number
                    # len: averaging number
                    idx = find_nearest(
                        data[:,channel],
                        value[i],
                        averaging_number)

                    # computes the mean of the selected index to yield (embsize) length verctor
                    # shape 32
                    gen_value = np.mean(data[idx], axis=0)

                    # specifically updates the value of the mean embedding image to visualize 
                    # based on the linear spaced vector
                    # shape
                    gen_value[channel] = value[i]

                    # generates diffraction pattern
                    self.generated[meta['particle_inds'][p]: meta['particle_inds'][p+1],i,j] =\
                        self.generate_spectra(gen_value).squeeze()       
        h.close()

        # return self.generated
    
    def generate_by_labels(self,meta,labels,
                           checkpoint=None,
                         channels=None,
                        #  averaging_number=100,
                         overwrite=False,
                         with_affine=False,
                         **kwargs
                         ):
        """Generates images as the variables traverse the latent space.
        Saves to embedding h5 dataset

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """

        assert not self.train, 'set self.train to False'
        
        # sets the kwarg values
        for key, value in kwargs.items():
            exec(f'{key} = value')

        if channels==None: channels = np.arange(self.embedding_size)

        # gets the embedding if a specific embedding is not provided
        try:
            embedding = self.embedding
        except Exception as error:
            print(error)
            assert False, 'Make sure model is set to appropriate embeddings first'

        try: # try opening h5 file
            try: # make new file
                h = h5py.File(self.gen_h5_path,'r+')
            except: # open existing file
                h = h5py.File(self.gen_h5_path,'w')

            if checkpoint==None: check = self.checkpoint.split('/')[-1][:-4]
            else: check = checkpoint
            
            try: # make new dataset
                if overwrite and check in h: del h[check]
                self.generated = h.create_dataset(check,
                                            data=np.zeros( [len(meta['particle_list']),
                                                            len(np.unique(labels)),
                                                            len(channels),
                                                            128,128] ) )
                self.generated.attrs['clustered']=True      
                print(check)
            except: # open existing dataset for checkpoint
                self.generated = h[check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        # for p,p_name in enumerate(meta['particle_list']): # each sample
        #     print(p, p_name)
        #     data=self.embedding[meta['particle_inds'][p]:\
        #                         meta['particle_inds'][p+1]]
        for p,p_name in enumerate(meta['particle_list']): # each sample
            print(p, p_name)
            # shape 128*128, 32
            data=self.embedding[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
            data_labels=labels[meta['particle_inds'][p]:\
                                meta['particle_inds'][p+1]]
            inds=None
            # loops around the number of iterations to generate
            for i in tqdm(np.unique(labels)):
                inds = np.argwhere(data_labels==i)
            
                # # specifically updates the value of the mean embedding image to visualize 
                # # based on the linear spaced vector
                # gen_value[channel] = value[i]

                # # generates diffraction pattern
                # for ind in inds:
                #     self.generated[ind,i] = self.generate_spectra(embedding[ind,i,j]).squeeze()
                # gen_values

                # generate spectrum at each index
                gen_value = np.atleast_2d(data[inds]).mean(axis=0).squeeze()
                
                self.generated[p,i] = self.generate_spectra(gen_value)
                # time.sleep(0.5)
        h.close()
           
    def generate_spectra(self, embedding,with_affine=False):
        """generates spectra from embeddings

        Args:
            embedding (torch.tensor): predicted embeddings to decode

        Returns:
            torch.tensor: decoded spectra
        """

        embedding = torch.from_numpy(np.atleast_2d(embedding)).to(self.device)
        embedding = self.decoder(embedding.float())
        if with_affine:
            embedding = self.transformer(embedding)
        embedding = embedding.cpu().detach().numpy()
        return embedding


class ConvBlock(nn.Module):
    """Convolutional Block with 3 convolutional layers, 1 layer normalization layer with ReLU and ResNet

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, t_size, n_step):
        """Initializes the convolutional block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(ConvBlock, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_2 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_3 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov1d_1(x)
        out = self.cov1d_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_4(out)
        out = out.add(x_input)

        return out


class IdentityBlock(nn.Module):

    """Identity Block with 1 convolutional layers, 1 layer normalization layer with ReLU"""

    def __init__(self, t_size, n_step):
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(IdentityBlock, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class Affine_Transform(nn.Module):
    def __init__(self,
                 device,
                 scale = True,
                 shear = True,
                 rotation = True,
                 translation = True,
                 Symmetric = True,
                 mask_intensity = True,
                 scale_limit = 0.05,
                 shear_limit = 0.1,
                 rotation_limit = 0.1,
                 trans_limit = 0.15,
                 adj_mask_para=0
                 ):
        super(Affine_Transform,self).__init__()
        self.scale = scale
        self.shear = shear
        self.rotation = rotation
        self.translation = translation
        self.Symmetric = Symmetric
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.rotation_limit = rotation_limit
        self.trans_limit = trans_limit
        self.adj_mask_para = adj_mask_para
        self.mask_intensity = mask_intensity
        self.device = device
        self.count = 0

    def forward(self,out,rotate_value = None):

        if self.scale:
            scale_1 = self.scale_limit*nn.Tanh()(out[:,self.count])+1
            scale_2 = self.scale_limit*nn.Tanh()(out[:,self.count+1])+1
            self.count +=2
        else:
            scale_1 = torch.ones([out.shape[0]]).to(self.device)
            scale_2 = torch.ones([out.shape[0]]).to(self.device)

        if self.rotation:
            if rotate_value!=None:
                # use large mask no need to limit to too small range
                rotate = rotate_value.reshape(out[:,self.count].shape) + self.rotation_limit*nn.Tanh()(out[:,self.count])
                self.count+=1
            else:
                rotate = nn.ReLU()(out[:,self.count])
                self.count+=1
        else:
            rotate = torch.zeros([out.shape[0]]).to(self.device)

        if self.shear:
            if self.Symmetric:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = shear_1
                self.count+=1
            else:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = self.shear_limit*nn.Tanh()(out[:,self.count+1])
                self.count+=2
        else:
            shear_1 = torch.zeros([out.shape[0]]).to(self.device)
            shear_2 = torch.zeros([out.shape[0]]).to(self.device)
        # usually the 4d-stem has symetric shear value, we make xy=yx, that's the reason we don't need shear2

        if self.translation:
            trans_1 = self.trans_limit*nn.Tanh()(out[:,self.count])
            trans_2 = self.trans_limit*nn.Tanh()(out[:,self.count+1])
            self.count +=2
        else:
            trans_1 = torch.zeros([out.shape[0]]).to(self.device)
            trans_2 = torch.zeros([out.shape[0]]).to(self.device)
  
        if self.mask_intensity:
            mask_parameter = self.adj_mask_para*nn.Tanh()(out[:,self.embedding_size:self.embedding_size+1])+1
        else:
            # this project doesn't need mask parameter to adjust value intensity in mask region, so we make it 1 here.
            mask_parameter = torch.ones([out.shape[0]])

        self.count = 0
 
        a_1 = torch.cos(rotate)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones([out.shape[0]]).to(self.device)
        a_5 = torch.zeros([out.shape[0]]).to(self.device)

       # combine shear and strain together
        c1 = torch.stack((scale_1,shear_1), dim=1).squeeze()
        c2 = torch.stack((shear_2,scale_2), dim=1).squeeze()
        c3 = torch.stack((a_5,a_5), dim=1).squeeze()
        scaler_shear = torch.stack((c1, c2, c3), dim=2)

        # Add the rotation after the shear and strain
        b1 = torch.stack((a_1,a_2), dim=1).squeeze()
        b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
        b3 = torch.stack((a_5,a_5), dim=1).squeeze()
        rotation = torch.stack((b1, b2, b3), dim=2)

        d1 = torch.stack((a_4,a_5), dim=1).squeeze()
        d2 = torch.stack((a_5,a_4), dim=1).squeeze()
        d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
        translation = torch.stack((d1, d2, d3), dim=2)

        return scaler_shear, rotation, translation, mask_parameter


class Encoder(nn.Module):
    """Encoder block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, original_step_size, pooling_list, embedding_size, conv_size, device):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image
            pooling_list (List): the list of parameter for each 2D MaxPool layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
        """

        super(Encoder, self).__init__()
        self.device = device
        blocks = []

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]

        number_of_blocks = len(pooling_list)

        blocks.append(ConvBlock(t_size=conv_size,
                                n_step=original_step_size))
        blocks.append(IdentityBlock(
            t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(
            pooling_list[0], stride=pooling_list[0]))

        for i in range(1, number_of_blocks):
            original_step_size = [
                original_step_size[0] // pooling_list[i - 1],
                original_step_size[1] // pooling_list[i - 1],
            ]
            blocks.append(ConvBlock(t_size=conv_size,
                                    n_step=original_step_size))
            blocks.append(
                IdentityBlock(t_size=conv_size, n_step=original_step_size)
            )
            blocks.append(nn.MaxPool2d(
                pooling_list[i], stride=pooling_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        original_step_size = [
            original_step_size[0] // pooling_list[-1],
            original_step_size[1] // pooling_list[-1],
        ]
        input_size = original_step_size[0] * original_step_size[1]

        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        self.relu_1 = nn.ReLU()

        self.dense = nn.Linear(input_size, embedding_size)
        self.affine_dense = nn.Linear(embedding_size,7) # TODO: save in (nx6) instead of (nx2x3)
        self.affine = Affine_Transform(self.device, Symmetric = False,
                                    mask_intensity = False)

    def forward(self, x):
        """Forward pass of the encoder

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        selection = self.relu_1(out)

        # get affine transforms from dense
        affine_dense = self.affine_dense(selection) 
        scaler_shear, rotation, translation, mask_parameter = self.affine(affine_dense)

        return selection,scaler_shear, rotation, translation


class Decoder(nn.Module):
    """Decoder class

    Args:
        nn (nn.module): base class for all neural network modules
    """

    def __init__(
        self,
        original_step_size,
        upsampling_list,
        embedding_size,
        conv_size,
        pooling_list,
    ):
        """Decoder block

        Args:
            original_step_size (Int): the x and y size of input image
            upsampling_list (Int): the list of parameter for each 2D upsample layer
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block
            pooling_list (List): the list of parameter for each 2D MaxPool layer
        """

        super(Decoder, self).__init__()
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(
            embedding_size, original_step_size[0] * original_step_size[1]
        )
        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        blocks = []
        number_of_blocks = len(upsampling_list)
        blocks.append(ConvBlock(t_size=conv_size,
                                n_step=original_step_size))
        blocks.append(IdentityBlock(
            t_size=conv_size, n_step=original_step_size))
        for i in range(number_of_blocks):
            blocks.append(
                nn.Upsample(
                    scale_factor=upsampling_list[i],
                    mode="bilinear",
                    align_corners=True,
                )
            )
            original_step_size = [
                original_step_size[0] * upsampling_list[i],
                original_step_size[1] * upsampling_list[i],
            ]
            blocks.append(ConvBlock(t_size=conv_size,
                                    n_step=original_step_size))
            blocks.append(
                IdentityBlock(t_size=conv_size, n_step=original_step_size)
            )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        self.output_size_0 = original_step_size[0]
        self.output_size_1 = original_step_size[1]

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        out = self.dense(x)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)

        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        output = out.view(-1, self.output_size_0, self.output_size_1)

        return output

## TODO: adjust transformer class so we can apply transforms during decoding
class Transformer(nn.Module):
#     def __init__(self,device):
#         """AutoEncoder model

#         Args:
#             enc (nn.Module): Encoder block
#             dec (nn.Module): Decoder block
#         """
#         super().__init__()
#         self.device = device

#     def forward(self, x):
#         """Forward pass of the autoencoder applies and affine grid to the decoder value

#         Args:
#             x (Tensor): Input (training data)

#        Returns:
#             embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
#         """
#         predicted,affines = x
#         scaler_shear,rotation,translation = affines
        
#         # sample and apply affine grids
#         size_grid = torch.ones([predicted.shape[0], 1, 
#                                 predicted.shape[-2], 
#                                 predicted.shape[-1]])

#         grid_1 = F.affine_grid(scaler_shear.to(self.device), 
#                                size_grid.size()).to(self.device) # scale shear
#         grid_2 = F.affine_grid(rotation.to(self.device), 
#                                size_grid.size()).to(self.device) # rotation
#         grid_3 = F.affine_grid(translation.to(self.device), 
#                                size_grid.size()).to(self.device) # translation
        
#         predicted = F.grid_sample(predicted,grid_3) # translation first to center
#         predicted = F.grid_sample(predicted,grid_1)
#         predicted = F.grid_sample(predicted,grid_2)

#         return predicted
    pass
        

class AutoEncoder(nn.Module):
    def __init__(self, enc, dec,emb_size,device,
                 training=True,generator_mode='affine'):
        """AutoEncoder model

        Args:
            enc (nn.Module): Encoder block
            dec (nn.Module): Decoder block
            emb_size ():
            device():
            train (bool): 
            mode (): {'affine','no_affine'}
        """
        super().__init__()
        self.device = device
        self.enc = enc
        self.dec = dec
        self.temp_affines = None # saves for filling embedding info
        self.training = training
        self.generator_mode = generator_mode
        # self.transformer = Transformer(self.device)

    def forward(self, x):
        """Forward pass of the autoencoder applies and affine grid to the decoder value

        Args:
            x (Tensor): Input (training data)

       Returns:
            embedding, predicted (Tuple: Tensor): embedding and generated image with affine transforms applied 
        """
        embedding,scaler_shear,rotation,translation = self.enc(x)
        predicted = self.dec(embedding).unsqueeze(1)
        self.temp_affines = scaler_shear,rotation,translation
        
        if self.generator_mode=='no_affine':
            if self.training: return embedding,predicted.squeeze()
            else: return predicted.squeeze()

        # sample and apply affine grids
        size_grid = torch.ones([x.shape[0],1,x.shape[-2],x.shape[-1]])

        grid_1 = F.affine_grid(scaler_shear.to(self.device), size_grid.size()).to(self.device) # scale shear
        grid_2 = F.affine_grid(rotation.to(self.device), size_grid.size()).to(self.device) # rotation
        grid_3 = F.affine_grid(translation.to(self.device), size_grid.size()).to(self.device) # translation

        predicted = F.grid_sample(predicted,grid_3) # translation first to center
        predicted = F.grid_sample(predicted,grid_1)
        predicted = F.grid_sample(predicted,grid_2)

        if self.generator_mode=='affine':
            if self.training: return embedding,predicted.squeeze()
            else: return predicted.squeeze()
        
        assert False, 'set train to True/False, and generator_mode to affine/no_affine'

def db_show_im(tensor):
    import matplotlib.pyplot as plt
    plt.imshow(tensor.squeeze().detach().cpu())
    plt.show()