function save_model(model,path)
   --print("Saved at : "..path)
   model.modules[2] = nil -- delete decoding part, no need to forward model
   model:float()
   parameters, gradParameters = model:getParameters()
   local lightModel = model:clone():float()
   lightModel:clearState()
   torch.save(path,lightModel)
end


function load_list(list)
   local im={}
   for i=1, #list do
      table.insert(im,getImage(list[i]))
   end 
   return im
end

function load_batch(list,batchSize,lenght, width, indice)
   local batch=torch.Tensor(batchSize,3,lenght, width)
   for i=indice, indice+batchSize-1 do
      batch[i-indice+1]=getImage(list[i])
   end 
	batch=batch-batch:mean()
	batch=batch/batch:std()
   return batch
end

function getImage(im)
   if im=='' or im==nil then return nil end
   local image1=image.load(im,3,'byte')
   local format="200x200"
   local img1_rsz=image.scale(image1,format)
   return img1_rsz:float()
end


