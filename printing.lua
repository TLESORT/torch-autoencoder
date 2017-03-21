function Print_performance(model, list, name, Log_Folder,truth, epoch, displayPlot)

   local list_learned_rep={}
	if opt.network=="deep" then
		id_rep=23
	else
		id_rep=4
	end
      
   for i=1, #list do
      Batch=load_batch(list,1,200,200,i)
      
      model:forward(Batch:cuda())
      local learned_rep=model:get(id_rep).output[1] 	

-- sample taken for visualisation
		if i<20 then
			concat=torch.cat(Batch[1]:double(),model.output:double():reshape(3,200,200),3)
			if i%5==1 then display=concat
			elseif i%5==0 and i==5 then 
				display=torch.cat(display,concat,2)
				patchwork = display 
			elseif i%5==0 then 
				display=torch.cat(display,concat,2)
				if i==5 then
					patchwork = display
				else 
					patchwork = torch.cat(patchwork,display ,3)
				end
			else display=torch.cat(display,concat,2)
			end
		end

      table.insert(list_learned_rep,learned_rep)
   end
	
	image.save("./Log/reconstruction/reconstruction_epoch_"..epoch..".jpg",patchwork)
   corr=ComputeCorrelation(truth,list_learned_rep)
   if displayPlot then
      show_figure(list_learned_rep, Log_Folder..'state'..name..'.log')
      show_figure_normalized(list_learned_rep,truth, Log_Folder..'stateNorm'..name..'.log',corr)
   end
   return corr
end

function ComputeCorrelation(truth,output)
   Truth=torch.Tensor(#truth)
   Output=torch.Tensor(#output)
   for i=1, #truth do
      Truth[i]=truth[i]
      Output[i]=output[i]
   end
   corr=torch.cmul((Truth-Truth:mean()),(Output-Output:mean())):mean()
   corr=corr/(Truth:std()*Output:std())
   return corr
end

function show_figure_normalized(output,truth, Name, corr)

   local Truth=torch.Tensor(#truth)
   local Output=torch.Tensor(#output)	
   local corr=corr or 1
   if corr<0 then 
      Variable_Truth='Normalized Truth (*-1)'
      corr=-1
   else Variable_Truth='Normalized Truth ' end
   local Variable_Output='Normalized State'

   for i=1, #truth do
      Truth[i]=truth[i]
      Output[i]=output[i]
   end
   Truth=corr*(Truth-Truth:mean())/Truth:std()
   Output=(Output-Output:mean())/Output:std()


   -- log results to files
   accLogger = optim.Logger(Name)

   for i=1, #output do
      -- update logger
      accLogger:add{[Variable_Output] = Output[i],[Variable_Truth] = Truth[i]}
   end
   -- plot logger
   accLogger:style{[Variable_Output] = '+',[Variable_Truth] = '+'}
   accLogger.showPlot = false
   accLogger:plot()
end

function show_figure(output, Name,point)
   local point=point or '+'
   local Variable_Output='State'
   local accLogger = optim.Logger(Name)
   for i=1, #output do accLogger:add{[Variable_Output] = output[i]}end
   accLogger:style{[Variable_Output] = '+'}
   accLogger.showPlot = false
   accLogger:plot()
end

function print_list(list_loss,Name_file, Name)
   local Variable_Output=Name
   local accLogger = optim.Logger(Name_file)
   for i=1, #list_loss do accLogger:add{[Variable_Output] = list_loss[i]}end
   accLogger:style{[Variable_Output] = '-'}
   accLogger.showPlot = false
   accLogger:plot()
end
