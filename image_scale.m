 cd GOPRO_Large
 cd train
 folders = dir('.');
 counter = 1;
 for idx = 3:length(folders)
     cd(folders(idx,1).name)
     files = dir('./blur/*.png');
     j = counter;
     for i = 1:length(files)
         filename = files(i,1).name;
         newfilename = sprintf('%06d.png', j);
         copyfile(['./blur/',filename], ['../../blur_all_train/', newfilename]);
         j = j+1;
     end
     files = dir('./sharp/*.png');
     j = counter;
     for i = 1:length(files)
         filename = files(i,1).name;
         newfilename = sprintf('%06d.png', j);
         copyfile(['./sharp/',filename], ['../../sharp_all_train/', newfilename]);
         j = j+1;
     end  
     counter = j;
     cd ..
 end
 cd ..

 cd test
 folders = dir('.');
 counter = 1;
 for idx = 3:length(folders)
     cd(folders(idx,1).name)
     files = dir('./blur/*.png');
     j = counter;
     for i = 1:length(files)
         filename = files(i,1).name;
         newfilename = sprintf('%06d.png', j);
         copyfile(['./blur/',filename], ['../../blur_all_test/', newfilename]);
         j = j+1;
     end
     files = dir('./sharp/*.png');
     j = counter;
     for i = 1:length(files)
         filename = files(i,1).name;
         newfilename = sprintf('%06d.png', j);
         copyfile(['./sharp/',filename], ['../../sharp_all_test/', newfilename]);
         j = j+1;
     end  
     counter = j;
     cd ..
 end
 cd ..

 cd sharp_all_train
 files = dir('*.png');
 for i = 1:length(files)
    fprintf('imgNo %d\n', i);
    imagefile = files(i,1).name;
    img = imresize(imread(imagefile), 1/2, 'cubic');
    imwrite(img, imagefile);
 end
 cd ..

 cd blur_all_train
 files = dir('*.png');
 for i = 1:length(files)
    fprintf('imgNo %d\n', i);
    imagefile = files(i,1).name;
    img = imresize(imread(imagefile), 1/2, 'cubic');
    imwrite(img, imagefile);
 end
 cd ..

 cd sharp_all_test
 files = dir('*.png');
 for i = 1:length(files)
    fprintf('imgNo %d\n', i);
    imagefile = files(i,1).name;
    img = imresize(imread(imagefile), 1/2, 'cubic');
    imwrite(img, imagefile);
 end
 cd ..

 cd blur_all_test
 files = dir('*.png');
 for i = 1:length(files)
    fprintf('imgNo %d\n', i);
    imagefile = files(i,1).name;
    img = imresize(imread(imagefile), 1/2, 'cubic');
    imwrite(img, imagefile);
 end
 cd ..

