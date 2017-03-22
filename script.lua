
require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'
--local cuda = pcall(require, 'cutorch') -- Use CUDA if available

require "Get_Baxter_Files"
require "function"
require "printing"


function AE_Training(model,batch)
   local LR= 0.001
	if opt.optimiser=="sgd" then  optimizer = optim.sgd end
	if opt.optimiser=="rmsprop" then  optimizer = optim.rmsprop end
	if opt.optimiser=="adam" then optimizer = optim.adam end
	model:cuda()

	input=batch
	expected=batch

	if opt.model=="DAE" then
		noise=torch.rand(batch:size())
		noise=(noise-noise:mean())/(noise:std())
		noise=noise:cuda()
		input=input+noise
	end

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
		criterion=nn.AbsCriterion()
		criterion=criterion:cuda()
		ouput=model:forward(input)
		loss = criterion:forward(ouput, expected)
		gradInput=model:backward(input, criterion:backward(ouput, expected))
		--print(gradInput:mean())
      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState) 

	if opt.execution== 'debug' then
		print(model:get(6).weight[1]:mean())
		print(torch.cmul(model:get(6).gradInput, model:get(6).gradInput):mean())
	end

   return loss[1]
end


function train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


	local nbEpoch=50
	local nbIter=100
	local list_loss={}
	local list_corr={}
	local loss=0
	nbList= #list_folders_images
   local plot = true

	name_save="AE_model_3D.t7"
	txt_test=list_txt[nbList]
	local truth
	if opt.dimension=="1D" then
		truth=getTruth(txt_test)
	else
		nb_part=50
		part_test=1
		truth=getTruth_3D(txt_test, nb_part, part_test)
	end
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
					loss_iter=AE_Training(model,Batch)
					loss = loss + loss_iter
				end
				xlua.progress(iter, nbIter)
			end
		xlua.progress(epoch, nbEpoch)
		table.insert(list_loss,loss/(nbBatch*nbIter*BatchSize))
		print("Mean Loss : "..loss/(nbBatch*nbIter*BatchSize))
		print_list(list_loss, Log_Folder.."Mean_Loss.log","loss")
		corr=Print_performance(model, img_test, "Test"..epoch, Log_Folder,truth,epoch, plot)
		table.insert(list_corr,corr)
		if opt.dimension=="1D" then
			print_list(list_corr, Log_Folder.."corr.log","corr")
		end
		print("Corr : ")
		print(corr)
	end
      save_model(model,name_save)
end

-- Command-line options
local cmd = torch.CmdLine()
cmd:option('-optimiser', 'sgd', 'Optimiser : adam|sgd|rmsprop')
cmd:option('-execution', 'release', 'execution : debug|release')
cmd:option('-network', 'deep', 'network : deep|base')
cmd:option('-model', 'AE', 'model : AE|DAE')
cmd:option('-dimension', '1D', 'dimension : 1D|3D')
opt = cmd:parse(arg)


torch.manualSeed(1337)
LR=0.0001
local dataAugmentation=true

local Log_Folder='./Log/'
if opt.network=="deep" then Log_Folder='./Deep_Log/' end
local list_folders_images, list_txt=Get_HeadCamera_HeadMvt(opt.dimension)

local nb_dims=1
if opt.dimension=="3D" then nb_dims=3 end
local hidden=100
BatchSize=2
image_width=200
image_height=200

if opt.network=="deep" then require('./deep_model')
else require('./model') end
model = getModel(image_width,image_height,3,hidden,nb_dims)
model=model:cuda()
parameters,gradParameters = model:getParameters()
train_Epoch(list_folders_images,list_txt,Log_Folder,use_simulate_images,LR)


imgs={} --memory is free!!!!!
