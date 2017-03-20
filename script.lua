
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'
local cuda = pcall(require, 'cutorch') -- Use CUDA if available

require "Get_Baxter_Files"
require "function"
require "printing"


function AE_Training(model,batch)
   local LR= 0.1
	if opt.optimiser=="sgd" then  optimizer = optim.sgd end
	if opt.optimiser=="rmsprop" then  optimizer = optim.rmsprop end
	if opt.optimiser=="adam" then optimizer = optim.adam end
	model:cuda()
   

   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
		-- just in case:
		collectgarbage()
		--get new parameters
		if x ~= parameters then
		 parameters:copy(x)
		end
		--reset gradients
		gradParameters:zero()
		--criterion=nn.MSECriterion()
		criterion=nn.AbsCriterion()
		criterion=criterion:cuda()
		ouput=model:forward(batch)
		loss = criterion:forward(ouput, batch)
		gradInput=model:backward(batch, criterion:backward(ouput, batch))
		--print(gradInput:mean())
      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState) 

	if opt.execution== 'debug' then
		print(model:get(6).weight[1]:mean())
		print(torch.cmul(model:get(6).gradInput, model:get(6).gradInput):mean())
	end

   return loss[1],model.output
end


function train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


	local nbEpoch=50
	local nbIter=1
	local list_loss={}
	local list_corr={}
	local loss=0
	nbList= #list_folders_images
   local plot = true

	name_save="AE_model.t7"
	txt_test=list_txt[nbList]
	local truth=getTruth(txt_test)
	img_test=images_Paths(list_folders_images[nbList])

	for epoch=1, nbEpoch do
	loss=0
	print('--------------Epoch : '..epoch..' ---------------')
		for iter=1, nbIter do
			indice1=torch.random(1,nbList-1)
			indice1=1
			list=images_Paths(list_folders_images[indice1])

			Batch=torch.Tensor(BatchSize,200,200,3)
			nbBatch=math.floor(#list/BatchSize)-2
			for numImage=1,nbBatch do
				batch=load_batch(list,BatchSize,image_height,image_width,(numImage-1)*BatchSize+1)
				Batch=batch:cuda()
				loss_iter,output=AE_Training(model,Batch)
				loss = loss + loss_iter
			end
			xlua.progress(iter, nbIter)
		end
	xlua.progress(epoch, nbEpoch)
	table.insert(list_corr,corr)
	table.insert(list_loss,loss/(nbBatch*nbIter*BatchSize))
	print("Mean Loss : "..loss/(nbBatch*nbIter*BatchSize))
	display=torch.cat(Batch[1],output[1]:reshape(3,200,200),3)
	image.save("./Log/reconstruction/reconstruction_epoch_"..epoch..".jpg",display)
	--image.display(display)

	corr=Print_performance(model, img_test, "Test"..epoch, Log_Folder,truth, plot)
    print("Corr : "..corr)
	end
      save_model(model,name_save)
end

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-optimiser', 'sgd', 'Optimiser : adam|sgd|rmsprop')
cmd:option('-execution', 'release', 'execution : debug|release')
opt = cmd:parse(arg)


torch.manualSeed(1337)
LR=0.0001
local dataAugmentation=true
local Log_Folder='./Log/'
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt()
local nb_dims=1
local hidden=100
BatchSize=2
image_width=200
image_height=200

require('./model')
model = getModel(image_width,image_height,3,hidden,nb_dims)
model=model:cuda()
parameters,gradParameters = model:getParameters()

train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


imgs={} --memory is free!!!!!
