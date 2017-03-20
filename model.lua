require 'nn'

-- network-------------------------------------------------------
function getModel(height,width,len,hidden,nb_dims)
	Auto_Timnet = nn.Sequential()
	Auto_Timnet:add(nn.View(height*width*len))                
	Auto_Timnet:add(nn.Linear(height*width*len, hidden))
	Auto_Timnet:add(nn.ReLU())                    
	Auto_Timnet:add(nn.Linear(hidden, nb_dims))
	Auto_Timnet:add(nn.ReLU())       
	Auto_Timnet:add(nn.Linear(nb_dims, hidden))
	Auto_Timnet:add(nn.ReLU())                    
	Auto_Timnet:add(nn.Linear(hidden, height*width*len))

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local Auto_Timnet = require('weight-init')(Auto_Timnet, method)
        print("Creating Model")
	print('Timnet\n' .. Auto_Timnet:__tostring());
	return Auto_Timnet
end



