require 'nn'

-- network-------------------------------------------------------
function _getModel(height,width,len,hidden,nb_dims)

   nbFilter=32
   Timnet = nn.Sequential()
   Timnet:add(nn.SpatialConvolution(3, nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(nbFilter))
   Timnet:add(nn.ReLU())	
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(nbFilter, 2*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(2*nbFilter)) 
   Timnet:add(nn.ReLU())	
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(2*nbFilter, 4*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(4*nbFilter)) 
   Timnet:add(nn.ReLU())	
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(4*nbFilter, 8*nbFilter, 3, 3))
   Timnet:add(nn.SpatialBatchNormalization(8*nbFilter)) 
   Timnet:add(nn.ReLU())	
   Timnet:add(nn.SpatialMaxPooling(2,2,2,2))

   Timnet:add(nn.SpatialConvolution(8*nbFilter, 1, 1, 1))
   Timnet:add(nn.SpatialBatchNormalization(1)) 
   Timnet:add(nn.ReLU())
   Timnet:add(nn.View(100))                
   Timnet:add(nn.Linear(100, 500))
   Timnet:add(nn.ReLU())                    
   Timnet:add(nn.Linear(500, nb_dims))
   
   Timnet:add(nn.Linear(nb_dims, hidden))
   Timnet:add(nn.ReLU())                    
   Timnet:add(nn.Linear(hidden, height*width*len))

   -- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
   local method = 'xavier'
   local Timnet = require('weight-init')(Timnet, method)
   print("Creating Model")
   print('Timnet\n' .. Timnet:__tostring());
   return Timnet
end

function getModel(height,width,len,hidden,nb_dims)

   numLast = 0
   
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
   encoder:add(pool1)

   encoder:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)
   encoder:add(pool2)
   
   encoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   encoder:add(nn.ReLU(true))

   local pool3 = nn.SpatialMaxPooling(2, 2, 2, 2)
   encoder:add(pool3)

   encoder:add(nn.View(8*25*25):setNumInputDims(3))
   encoder:add(nn.Linear(8*25*25,nb_dims))

   -- Create decoder
   decoder = nn.Sequential()
   decoder:add(nn.Linear(nb_dims,8*25*25))
   decoder:add(nn.View(8,25,25):setNumInputDims(1))
   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool3))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

   decoder:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool2))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

   decoder:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.ReLU(true))

   decoder:add(nn.SpatialMaxUnpooling(pool1))
   --decoder:add(nn.SpatialUpSamplingNearest(2))

   decoder:add(nn.SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1))
   decoder:add(nn.Sigmoid(true))
   decoder:add(nn.View(3, 200, 200))

   -- Create autoencoder
   autoencoder = nn.Sequential()
   autoencoder:add(encoder)
   autoencoder:add(decoder)

   return autoencoder
   
end


