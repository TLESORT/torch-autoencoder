---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input (Path): path of a Folder which contained jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(Path)
   local listImage={}
   for file in paths.files(Path) do
      -- We only load files that match the extension
      if file:find('jpg' .. '$') then
         -- and insert the ones we care about in our table
         table.insert(listImage, paths.concat(Path,file))
      end
      
   end
   table.sort(listImage)
   return listImage
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function txt_path(Path)
   local txt=nil
   for file in paths.files(Path) do
      if file:find('txt' .. '$') then
         txt=paths.concat(Path,file)
      end
   end
   return txt
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folders(Path, including)
   local list= {}
   local list_txt={}
   for file in paths.files(Path) do
      if file:find(including) then
         Path_Folder= paths.concat(Path,file)
         table.insert(list,paths.concat(Path_Folder,"Images"))
         table.insert(list_txt, paths.concat(Path_Folder,"robot_joint_states.txt"))
      end
   end
   return list, list_txt
end

function Get_Folders_3D(Path, including, excluding,list)
   local list=list or {}
   local incl=including or ""
   local excl=excluding or "uyfouhjbhytfoughl" -- random motif
   for file in paths.files(Path) do
      -- We only load files that match 2016 because we know that there are the folder we are interested in
      if file:find(incl) and (not file:find(excl)) then
         -- and insert the ones we care about in our table
         table.insert(list, paths.concat(Path,file))
      end
   end
   return list
end

function txt_path(Path,including)
   local including=including or ""
   local txt=nil
   for file in paths.files(Path) do
      if file:find(including..'.txt' .. '$') then
         txt=paths.concat(Path,file)
      end
   end
   return txt
end

---------------------------------------------------------------------------------------
-- Function : Get_HeadCamera_HeadMvt(use_simulate_images)
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images 
-- Output (list_head_left): list of the images directories path
-- Output (list_txt):  txt list associated to each directories (this txt file contains the grundtruth of the robot position)
---------------------------------------------------------------------------------------
function Get_HeadCamera_HeadMvt(threeD)
   local Path
   local Paths_Folder={}
   local list_txt={}
   if threeD=="3D" then
      Path="../baxter_representation_learning_3D/simpleData3D"
      local Paths=Get_Folders_3D(Path,'record')
      for i=1, #Paths do
         Paths_Folder=Get_Folders_3D(Paths[i],'recorded','txt',Paths_Folder)
         table.insert(list_txt, txt_path(Paths[i],"endpoint_state"))
      end
   else
      Path="../baxter_representation_learning_1D/moreData/"
      Paths_Folder, list_txt=Get_Folders(Path,'head_pan')
   end

   table.sort(list_txt)
   table.sort(Paths_Folder)
   
   return Paths_Folder, list_txt
end


---------------------------------------------------------------------------------------
-- Function : tensorFromTxt(path)
-- Input (path) : path of a txt file which contain position of the robot
-- Output (torch.Tensor(data)): tensor with all the joint values (col: joint, lign : indice)
-- Output (labels):  name of the joint
---------------------------------------------------------------------------------------
function tensorFromTxt(path)
   local data, raw = {}, {}
   local rawCounter, columnCounter = 0, 0
   local nbFields, labels, _line = nil, nil, nil

   for line in io.lines(path)  do 
      local comment = false
      if line:sub(1,1)=='#' then  
         comment = true            
         line = line:sub(2)
      end 
      rawCounter = rawCounter +1      
      columnCounter=0
      raw = {}
      for value in line:gmatch'%S+' do
         columnCounter = columnCounter+1
         raw[columnCounter] = tonumber(value)
      end

      -- we check that every row contains the same number of data
      if rawCounter==1 then
         nbFields = columnCounter
      elseif columnCounter ~= nbFields then
         error("data dimension for " .. rawCounter .. "the sample is not consistent with previous samples'")
      end
      
      if comment then labels = raw else table.insert(data,raw) end 
   end
   return torch.Tensor(data), labels
end




---------------------------------------------------------------------------------------
-- Function : getTruth(txt,use_simulate_images)
-- Input (txt) : 
-- Input (use_simulate_images) : 
-- Input (arrondit) :
-- Output (truth): 
---------------------------------------------------------------------------------------
function getTruth(txt)
   local truth={}
   local head_pan_indice=2
   local tensor, label=tensorFromTxt(txt)
   
   for i=1, (#tensor[{}])[1] do
      table.insert(truth, tensor[i][head_pan_indice])
   end
   return truth
end

function getTruth_3D(txt_joint, nb_part, part)
   local x=2
   local y=3
   local z=4
   
   local tensor, label=tensorFromTxt(txt_joint)

   numImages = tensor:size(1)
   local list_truth={}

   for i=1, numImages do--(#tensor[{}])[1] do	
      local truth=torch.Tensor(3)
      truth[1]=tensor[i][x]
      truth[2]=tensor[i][y]
      truth[3]=tensor[i][z]
      table.insert(list_truth,truth)
   end
   return list_truth
end


---------------------------------------------------------------------------------------
-- Function : arrondit(value)
-- Input (tensor) : 
-- Input (head_pan_indice) : 
-- Output (tensor): 
---------------------------------------------------------------------------------------
function arrondit(value)
   floor=math.floor(value*10)/10
   ceil=math.ceil(value*10)/10
   if math.abs(value-ceil)>math.abs(value-floor) then result=floor
   else result=ceil end
   return result
end
