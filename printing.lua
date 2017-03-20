function Print_performance(model, list, name, Log_Folder,truth, displayPlot)

   local list_learned_rep={}
      
   for i=1, #list do
      Batch=load_batch(list,1,200,200,i)
      
      model:forward(Batch:cuda())
      local learned_rep=model:get(4).output[1] 	

      table.insert(list_learned_rep,learned_rep)
   end
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
